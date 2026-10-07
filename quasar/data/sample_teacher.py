"""Sample teacher responses from an OpenAI-compatible ``/chat/completions`` server.

The input is a JSONL of prompts: ``quasar.data.prompts`` slices (``prompt``) or chat
rows (their first user turn); ``source`` and ``orig_row`` are carried through. Each
output line is

  {"index", "source", "orig_row", "finish_reason",
   "messages": [{"role": "user", ...}, {"role": "assistant", "content", "reasoning_content"}]}

where ``index`` is the input line number. Records are appended in completion order and
re-running the same command resumes: indices already in the output are skipped, so
failed requests are retried by the next run. ``max_tokens`` is clamped per prompt to
``max(256, min(--max_tokens, --context_length - prompt_tokens))``, counting the prompt
with the chat template and its generation prompt. The server's reasoning parser
returns the reasoning separately (``reasoning`` in vLLM, ``reasoning_content`` in
SGLang); it is stored as ``reasoning_content``, the field chat templates render.
Reasoning left inline in ``content`` is split off by ``build_distill``. No
per-request seed is sent.

Serve the teacher first, e.g. ``vllm serve Qwen/Qwen3-4B-Thinking-2507 --reasoning-parser
qwen3 --max-model-len 32768`` (one replica per GPU or data parallel), then run e.g.

  python -m quasar.data.sample_teacher --input opb_200k.jsonl --output samples.jsonl \\
      --model Qwen/Qwen3-4B-Thinking-2507 --context_length 32768 --max_tokens 16384 \\
      --temperature 0.6 --top_p 0.95 --top_k 20

Sampling settings of the paper corpora:

  Qwen3-4B-Thinking-2507:  --temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens 16384 --context_length 32768
  Llama-3.1-8B-Instruct:   --temperature 0.6 --top_p 0.9 --max_tokens 3584 --context_length 4096
  Qwen3-8B (thinking on):  --max_tokens 16384 --context_length 32768 and the model's generation_config
                           sampling (--temperature 0.6 --top_p 0.95 --top_k 20)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from quasar.data.common import iter_jsonl, prompt_text, token_len

MIN_COMPLETION_TOKENS = 256
TIMEOUT_S = 1800
RETRIES = 3


def post(url: str, payload: dict) -> dict:
    """POST JSON with a few retries: long sampling jobs see transient server errors."""
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(2 * (attempt + 1))


def sample_one(index: int, example: dict, tokenizer: Any, args: argparse.Namespace) -> dict:
    messages = [{"role": "user", "content": prompt_text(example)}]
    prompt_tokens = token_len(tokenizer, messages, add_generation_prompt=True)
    max_tokens = max(MIN_COMPLETION_TOKENS, min(args.max_tokens, args.context_length - prompt_tokens))
    payload = {"model": args.model, "messages": messages, "temperature": args.temperature, "max_tokens": max_tokens}
    payload.update({k: v for k, v in (("top_k", args.top_k), ("top_p", args.top_p)) if v is not None})
    choice = post(f"{args.base_url}/chat/completions", payload)["choices"][0]
    message = choice["message"]
    # vLLM's reasoning parser returns ``reasoning``; SGLang (and vLLM before the rename) ``reasoning_content``.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return {
        "index": index,
        "source": example.get("source"),
        "orig_row": example.get("orig_row"),
        "messages": messages
        + [{"role": "assistant", "content": message.get("content") or "", "reasoning_content": reasoning}],
        "finish_reason": choice.get("finish_reason"),
    }


def done_indices(path: str) -> set[int]:
    """Indices already written. A last line cut short by an interrupted run is ended with a
    newline, so new records start on a fresh line, and its index is sampled again."""
    done, line = set(), ""
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            with contextlib.suppress(ValueError):
                done.add(json.loads(line)["index"])
    if line and not line.endswith("\n"):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n")
    return done


def run(args: argparse.Namespace, tokenizer: Any) -> list[tuple[int, str]]:
    """Sample every pending input row, appending records; returns the failures."""
    examples = list(iter_jsonl(args.input))
    done = done_indices(args.output)
    todo = iter([i for i in range(len(examples)) if i not in done])
    print(f"{len(examples)} prompts, {len(done)} already sampled")

    errors, written, start = [], 0, time.time()
    with open(args.output, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures: dict = {}

        def submit_next() -> None:
            index = next(todo, None)
            if index is not None:
                futures[pool.submit(sample_one, index, examples[index], tokenizer, args)] = index

        for _ in range(args.concurrency):
            submit_next()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                index = futures.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    errors.append((index, str(exc)))
                    print(f"index={index} failed: {exc}")
                else:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += 1
                    if written % 100 == 0:
                        print(f"{written} written, {time.time() - start:.0f}s", flush=True)
                submit_next()
    print(f"wrote {written} records to {args.output}; {len(errors)} failed (re-run to retry)")
    return errors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL of prompt rows")
    ap.add_argument("--output", required=True, help="JSONL of response records (appended; resumable)")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="model name as served")
    ap.add_argument("--tokenizer", default=None, help="tokenizer of the served model (default: --model)")
    ap.add_argument("--context_length", type=int, required=True, help="prompt + completion budget")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    run(args, AutoTokenizer.from_pretrained(args.tokenizer or args.model))


if __name__ == "__main__":
    main()
