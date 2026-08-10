"""Search mRNABERT on HF mirror."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)
from huggingface_hub import list_models
models = list(list_models(search="mRNABERT"))
for m in models[:10]:
    print(m.id)