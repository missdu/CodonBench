"""Try YYLY66/mRNABERT - likely standard BERT architecture."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

from transformers import BertModel, BertForMaskedLM, AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, list_repo_files
import torch

model_id = "YYLY66/mRNABERT"
print(f"=== {model_id} ===")

# Check repo files
files = list_repo_files(model_id)
print(f"Repo files: {files}")

print("\nLoading config...")
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"Config: vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}")

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f"Tokenizer OK: vocab_size={tok.vocab_size}")

# Test tokenizer
test = tok("ATG AAA GGG TTT CCC", return_tensors="pt")
print(f"Test encoding: {test['input_ids'].tolist()}")

print("\nLoading safetensors...")
sf_path = hf_hub_download(model_id, "model.safetensors")
raw_state = load_file(sf_path)
print(f"Raw keys ({len(raw_state)}):")
for k in sorted(raw_state.keys())[:10]:
    print(f"  {k}: {raw_state[k].shape}")

# Try BertForMaskedLM
print("\nTrying BertForMaskedLM...")
try:
    model = BertForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    print(f"SUCCESS! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    with torch.no_grad():
        out = model(**{k: v for k, v in test.items()})
    print(f"Forward pass OK: logits shape={out.logits.shape}")

    save_path = "./models/mrnabert"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"Saved to {save_path}")
except Exception as e:
    print(f"BertForMaskedLM failed: {e}")

    # Try BertModel
    print("\nTrying BertModel...")
    try:
        model = BertModel.from_pretrained(model_id, trust_remote_code=True)
        print(f"SUCCESS! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    except Exception as e2:
        print(f"BertModel also failed: {e2}")

        # Manual load
        print("\nManual load from safetensors...")
        model = BertForMaskedLM(cfg)
        missing, unexpected = model.load_state_dict(raw_state, strict=False)
        print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if not missing and not unexpected:
            print("Perfect match!")
            save_path = "./models/mrnabert"
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tok.save_pretrained(save_path)
            print(f"Saved to {save_path}")

print("Done!")