"""Self-contained test inputs: a byte-level ChatML tokenizer, tiny random decoders and chat rows."""

import json
import random

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast, Qwen3Config

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.reasoning_content is defined %}<think>\n{{ m.reasoning_content }}\n</think>\n\n{% endif %}"
    "{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
WORDS = ["alpha", "beta", "gamma", "delta", "zeta", "theta", "kappa", "lambda", "sigma", "omega"]
# Every projection input is a multiple of the INT group (128); the vocabulary covers byte_tokenizer.
SIZES = dict(
    vocab_size=264,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=512,
)


def byte_tokenizer() -> PreTrainedTokenizerFast:
    """One token per byte plus the ChatML markers (259 ids): real chat templating and masks, no download."""
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    fast.chat_template = CHATML
    return fast


def tiny_model(family: str = "qwen3", seed: int = 0):
    """Random bf16 decoder, Qwen3 (tied embeddings) or Llama (untied), with a few outliers as in trained weights."""
    torch.manual_seed(seed)
    tied = family == "qwen3"
    config = (Qwen3Config if tied else LlamaConfig)(**SIZES, tie_word_embeddings=tied)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.view(-1)[torch.randperm(p.numel())[: p.numel() // 200]] *= 8
    return model


def save_base(model, path):
    """Save ``model`` and the byte tokenizer as a base checkpoint directory."""
    model.save_pretrained(path)
    byte_tokenizer().save_pretrained(path)
    return path


def chat_rows(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)

    def text(lo, hi):
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))

    return [
        {
            "messages": [
                {"role": "user", "content": text(2, 8)},
                {"role": "assistant", "content": text(2, 10), "reasoning_content": text(1, 6)},
            ]
        }
        for _ in range(n)
    ]


def write_jsonl(path, rows) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)
