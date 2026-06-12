from pathlib import Path
from kvr.config import Config, MODEL_LLAMA, MODEL_QWEN


def test_default_config_paths_are_pathlib():
    cfg = Config()
    assert isinstance(cfg.blob_mount, Path)
    assert isinstance(cfg.local_cache, Path)
    assert isinstance(cfg.results_dir, Path)


def test_model_ids_are_huggingface_paths():
    assert MODEL_LLAMA == "meta-llama/Llama-3.1-8B-Instruct"
    assert MODEL_QWEN == "Qwen/Qwen2.5-7B-Instruct"


def test_config_env_override():
    import os
    os.environ["KVR_BLOB_MOUNT"] = "/tmp/test_blob"
    try:
        cfg = Config.from_env()
        assert cfg.blob_mount == Path("/tmp/test_blob")
    finally:
        del os.environ["KVR_BLOB_MOUNT"]
