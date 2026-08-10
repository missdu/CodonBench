"""Download mRNABERT - manual approach: download pytorch_model.bin, load MosaicBERT manually."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

import torch
from huggingface_hub import hf_hub_download
import json

model_id = "YYLY66/mRNABERT"

# 1. Download pytorch_model.bin
print("Downloading pytorch_model.bin...")
bin_path = hf_hub_download(model_id, "pytorch_model.bin")
print(f"Downloaded to: {bin_path}")

# 2. Download config
print("Downloading config.json...")
cfg_path = hf_hub_download(model_id, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
print(f"Config: {json.dumps(cfg, indent=2)}")

# 3. Load state dict
print("Loading state dict...")
state_dict = torch.load(bin_path, map_location="cpu")
print(f"State dict keys ({len(state_dict)}):")
for k in sorted(state_dict.keys())[:15]:
    print(f"  {k}: {state_dict[k].shape}")
print(f"  ...")
for k in sorted(state_dict.keys())[-5:]:
    print(f"  {k}: {state_dict[k].shape}")

# 4. Save to local path
save_path = "./models/mrnabert_raw"
os.makedirs(save_path, exist_ok=True)
torch.save(state_dict, os.path.join(save_path, "pytorch_model.bin"))
with open(os.path.join(save_path, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print(f"\nRaw model saved to {save_path}")
print("Will need custom loading code for MosaicBERT architecture.")
print("Done!")