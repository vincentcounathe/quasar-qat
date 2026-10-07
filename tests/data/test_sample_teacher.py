import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from quasar.data import sample_teacher
from quasar.data.common import token_len


@pytest.fixture
def server():
    """Stub /chat/completions: echoes the prompt; prompts containing FAIL fail once."""
    state = {"payloads": [], "failed": set()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            if "FAIL" in prompt and prompt not in state["failed"]:
                state["failed"].add(prompt)
                self.send_response(500)
                self.end_headers()
                return
            state["payloads"].append(body)
            field = "reasoning" if "new-vllm" in prompt else "reasoning_content"
            message = {"role": "assistant", "content": f"answer: {prompt}", field: "some thoughts"}
            data = json.dumps(
                {
                    "choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", state
    httpd.shutdown()


def _args(tmp_path, base_url, **overrides):
    args = dict(
        input=str(tmp_path / "in.jsonl"),
        output=str(tmp_path / "out.jsonl"),
        base_url=base_url,
        model="teacher",
        context_length=600,
        max_tokens=300,
        temperature=0.6,
        top_p=0.95,
        top_k=None,
        concurrency=3,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _records(path):
    return sorted((json.loads(line) for line in Path(path).read_text().splitlines()), key=lambda r: r["index"])


def test_records_clamp_and_resume(server, tmp_path, toy_tokenizer, monkeypatch):
    base_url, state = server
    monkeypatch.setattr(sample_teacher, "RETRIES", 0)
    rows = [{"prompt": "word " * (i * 80) + f"q{i}", "source": "s", "orig_row": 100 + i} for i in range(8)]
    rows[2] = {"messages": [{"role": "user", "content": "new-vllm q2"}], "source": "t", "orig_row": 5}
    rows[5]["prompt"] = "FAIL q5"
    (tmp_path / "in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = _args(tmp_path, base_url)

    errors = sample_teacher.run(args, toy_tokenizer)
    assert [i for i, _ in errors] == [5]
    recs = _records(args.output)
    assert [r["index"] for r in recs] == [0, 1, 2, 3, 4, 6, 7]
    for payload in state["payloads"]:
        assert payload["top_p"] == 0.95 and "top_k" not in payload
        prompt_tokens = token_len(toy_tokenizer, payload["messages"], add_generation_prompt=True)
        assert payload["max_tokens"] == max(256, min(300, 600 - prompt_tokens))
    assert {256, 300} < {p["max_tokens"] for p in state["payloads"]}  # requested, clamped to the window, the floor
    for r in recs:
        assert r["messages"][1] == {
            "role": "assistant",
            "content": f"answer: {r['messages'][0]['content']}",
            "reasoning_content": "some thoughts",
        }
    assert recs[2]["source"] == "t" and recs[2]["orig_row"] == 5

    # An interrupted run left a cut-short line; the next run retries it and the failure.
    output = Path(args.output)
    lines = output.read_text().splitlines(keepends=True)
    cut = json.loads(lines[-1])["index"]
    output.write_text("".join(lines[:-1]) + lines[-1][:30])
    state["payloads"].clear()
    assert sample_teacher.run(args, toy_tokenizer) == []
    retried = sorted(int(p["messages"][0]["content"].split()[-1][1:]) for p in state["payloads"])
    assert retried == sorted([5, cut])
    good = [json.loads(line) for line in output.read_text().splitlines() if line.endswith("}")]
    assert sorted(r["index"] for r in good) == list(range(8))
    state["payloads"].clear()
    sample_teacher.run(args, toy_tokenizer)
    assert state["payloads"] == []
