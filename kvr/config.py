"""Paths, model IDs, and runtime configuration."""
import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_QWEN = "Qwen/Qwen2.5-7B-Instruct"
MODEL_QWEN3 = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PHI = "microsoft/Phi-3-mini-128k-instruct"
# Base (non-instruct-tuned) control for the instruct-tuning-artifact ablation.
# Same architecture + tokenizer as MODEL_LLAMA, without the instruct tuning.
MODEL_LLAMA_BASE = "meta-llama/Llama-3.1-8B"


def _path_from_env(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser()


@dataclass
class Config:
    blob_mount: Path = field(default_factory=lambda: _path_from_env("KVR_BLOB_MOUNT", "/mnt/kvr"))
    local_cache: Path = field(default_factory=lambda: _path_from_env("KVR_LOCAL_CACHE", "~/.cache/kvr"))
    results_dir: Path = field(default_factory=lambda: _path_from_env("KVR_RESULTS_DIR", "~/.cache/kvr/results"))

    @classmethod
    def from_env(cls) -> "Config":
        return cls()
