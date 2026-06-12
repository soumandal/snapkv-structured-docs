"""Download Llama-3.1-8B and Qwen2.5-7B weights to the HF hub cache.

Run once per VM. Honors the standard HF env vars (HF_HUB_CACHE, then
HF_HOME/hub, then ~/.cache/huggingface/hub). On the A100 the intended
location is `/mnt/kvr/models/.hf_cache` once the blob mount is in place;
until then point HF_HUB_CACHE at a local dir with enough free space.
"""
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from kvr.config import MODEL_LLAMA, MODEL_QWEN


def main() -> None:
    print(f"Cache: {HF_HUB_CACHE}")
    for model_id in [MODEL_LLAMA, MODEL_QWEN]:
        print(f"Downloading {model_id}")
        path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"],
            ignore_patterns=["*.bin"],
        )
        print(f"  → {path}")
    print("Done.")


if __name__ == "__main__":
    main()
