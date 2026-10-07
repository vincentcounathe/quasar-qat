"""Byte-for-byte copies of a checkpoint's non-weight files.

Re-serializing a config through ``transformers`` can rewrite fields (rope
settings in particular) in a form other serving stacks misread, and reloading a
tokenizer can drop or rename files. Copying the source files verbatim keeps an
exported model's metadata identical to the model it was trained from.
"""

from __future__ import annotations

import shutil
from pathlib import Path

CONFIG_FILES = ("config.json", "generation_config.json")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
)


def _local_dir(path: str | Path, patterns) -> Path:
    """A local directory, or the files matching ``patterns`` of a Hugging Face Hub repo."""
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path), allow_patterns=list(patterns)))


def copy_model_files(out_dir: str | Path, model_path: str | Path, tokenizer_path: str | Path | None = None):
    """Copy config/generation config from ``model_path`` and tokenizer files from ``tokenizer_path``.

    ``tokenizer_path`` defaults to ``model_path``. Returns the list of copied file names.
    """
    out = Path(out_dir)
    copied = []
    for src, names in ((model_path, CONFIG_FILES), (tokenizer_path or model_path, TOKENIZER_FILES)):
        src = _local_dir(src, names)
        for name in names:
            if (src / name).is_file():
                shutil.copyfile(src / name, out / name)
                copied.append(name)
    if "config.json" not in copied:
        raise FileNotFoundError(f"no config.json under {model_path}")
    return copied
