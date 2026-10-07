import pytest


class ToyTokenizer:
    """Whitespace tokenizer with a ChatML-like template: enough to exercise length filters
    without downloading a model."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m.get('reasoning_content') or ''} {m['content']}<|im_end|>\n" for m in messages
        )
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


@pytest.fixture
def toy_tokenizer(monkeypatch):
    """A ToyTokenizer, also returned by ``AutoTokenizer.from_pretrained`` inside the CLIs."""
    import transformers

    tok = ToyTokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    return tok


_CACHE = {}


@pytest.fixture
def hf_tokenizer():
    """Load a real tokenizer, or skip when it is not available (offline, gated)."""

    def load(name):
        if name not in _CACHE:
            from transformers import AutoTokenizer

            try:
                _CACHE[name] = AutoTokenizer.from_pretrained(name)
            except OSError as exc:  # offline, gated or missing
                _CACHE[name] = exc
        if isinstance(_CACHE[name], Exception):
            pytest.skip(f"tokenizer {name} unavailable: {_CACHE[name]}")
        return _CACHE[name]

    return load
