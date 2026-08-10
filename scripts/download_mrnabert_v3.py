"""Download mRNABERT - use AutoModel with trust_remote_code."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

from transformers import AutoModel, AutoTokenizer
import torch

# Try both model IDs
for model_id in ["YYLY66/mRNABERT", "Taykhoom/mRNABERT"]:
    print(f"\n{'='*60}")
    print(f"Trying {model_id}")
    print(f"{'='*60}")

    try:
        print("Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"Tokenizer OK: vocab_size={tok.vocab_size}")

        test = tok("ATG AAA GGG TTT CCC", return_tensors="pt")
        print(f"Test encoding: {test['input_ids'].tolist()}")

        print("Loading model with AutoModel + trust_remote_code...")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        print(f"Model OK: {type(model).__name__}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

        with torch.no_grad():
            out = model(**{k: v for k, v in test.items()})
        if hasattr(out, 'last_hidden_state'):
            print(f"Forward pass OK: last_hidden_state shape={out.last_hidden_state.shape}")
        else:
            print(f"Forward pass OK: output type={type(out)}")

        # Save locally
        save_path = f"./models/mrnabert_{model_id.split('/')[0].lower()}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tok.save_pretrained(save_path)
        print(f"Saved to {save_path}")
        break  # Success, stop trying

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")