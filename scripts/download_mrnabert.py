"""Download mRNABERT - use BertForMaskedLM and fix keys properly."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

from transformers import BertForMaskedLM, AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torch

model_id = "Taykhoom/mRNABERT"

print("Loading config...")
cfg = AutoConfig.from_pretrained(model_id)
print(f"Config: vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}")

print("Loading safetensors weights...")
sf_path = hf_hub_download(model_id, "model.safetensors")
raw_state = load_file(sf_path)
print(f"Raw keys ({len(raw_state)}):")
for k in sorted(raw_state.keys())[:10]:
    print(f"  {k}: {raw_state[k].shape}")
print(f"  ...")

print("\nCreating BertForMaskedLM from config...")
model = BertForMaskedLM(cfg)
model_keys = set(model.state_dict().keys())
print(f"Model expects {len(model_keys)} keys")

# Try loading with "bert." prefix (since BertForMaskedLM has "bert." prefix)
missing, unexpected = model.load_state_dict(raw_state, strict=False)
print(f"\nDirect load: Missing={len(missing)}, Unexpected={len(unexpected)}")
if not missing and not unexpected:
    print("Perfect match!")
else:
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    # Check if raw keys match model keys
    raw_set = set(raw_state.keys())
    matched = raw_set & model_keys
    print(f"\n  Matched keys: {len(matched)}/{len(model_keys)}")

# Verify forward pass
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
test = tok("ATG AAA GGG", return_tensors="pt")
with torch.no_grad():
    out = model(**{k: v for k, v in test.items()})
print(f"\nForward pass OK: logits shape={out.logits.shape}")

# Save the full model (with loaded weights)
save_path = "./models/mrnabert"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tok.save_pretrained(save_path)
print(f"Model + tokenizer saved to {save_path}")
print("Done!")
