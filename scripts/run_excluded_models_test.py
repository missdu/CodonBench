import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import traceback
from pathlib import Path
from datetime import datetime

EXCLUDED_MODELS = [
    {
        "model": "EnCodon-600M",
        "hf_id": "nvidia/NV-CodonFM-Encodon-600M",
        "architecture": "Custom (NVIDIA)",
        "expected_type": "cLM",
    },
    {
        "model": "EnCodon-1B",
        "hf_id": "nvidia/NV-CodonFM-Encodon-1B",
        "architecture": "Custom (NVIDIA)",
        "expected_type": "cLM",
    },
    {
        "model": "TE-80M",
        "hf_id": "nvidia/NV-CodonFM-TE-80M",
        "architecture": "Custom (NVIDIA)",
        "expected_type": "cLM",
    },
    {
        "model": "TE-600M",
        "hf_id": "nvidia/NV-CodonFM-TE-600M",
        "architecture": "Custom (NVIDIA)",
        "expected_type": "cLM",
    },
    {
        "model": "TE-1B",
        "hf_id": "nvidia/NV-CodonFM-TE-1B",
        "architecture": "Custom (NVIDIA)",
        "expected_type": "cLM",
    },
    {
        "model": "Cdwt-1B",
        "hf_id": "nvidia/NV-CodonFM-Cdwt-1B",
        "architecture": "Custom (NVIDIA)",
        "expected_type": "cLM",
    },
    {
        "model": "CodonTransformer",
        "hf_id": "adibvafa/CodonTransformer",
        "architecture": "BigBird",
        "expected_type": "cLM",
    },
    {
        "model": "Mistral-Codon-117M",
        "hf_id": "RaphaelMourad/Mistral-Codon-v1-117M",
        "architecture": "Mixtral MoE",
        "expected_type": "cLM",
    },
    {
        "model": "Mistral-Codon-16M",
        "hf_id": "RaphaelMourad/Mistral-Codon-v1-16M",
        "architecture": "Mixtral MoE",
        "expected_type": "cLM",
    },
    {
        "model": "Mistral-Codon-1M",
        "hf_id": "RaphaelMourad/Mistral-Codon-v1-1M",
        "architecture": "Mixtral MoE",
        "expected_type": "cLM",
    },
    {
        "model": "cdsBERT",
        "hf_id": "GleghornLab/cdsBERT",
        "architecture": "BERT (char tokenizer)",
        "expected_type": "cLM",
    },
    {
        "model": "cdsBERT-plus",
        "hf_id": "GleghornLab/cdsBERT-plus",
        "architecture": "BERT (char tokenizer)",
        "expected_type": "cLM",
    },
    {
        "model": "CodonT5",
        "hf_id": None,
        "architecture": "Custom (T5 variant)",
        "expected_type": "cLM",
    },
    {
        "model": "CodonBert-v2",
        "hf_id": "zluvolyote/CodonBert",
        "architecture": "Custom",
        "expected_type": "cLM",
    },
    {
        "model": "CaLM",
        "hf_id": None,
        "architecture": "Contrastive",
        "expected_type": "cLM",
    },
    {
        "model": "DeCodon",
        "hf_id": None,
        "architecture": "Encoder-Decoder",
        "expected_type": "cLM",
    },
    {
        "model": "HELM",
        "hf_id": None,
        "architecture": "Hierarchical",
        "expected_type": "cLM",
    },
    {
        "model": "Equi-mRNA",
        "hf_id": None,
        "architecture": "Equivariant",
        "expected_type": "cLM",
    },
    {
        "model": "CodonMoE",
        "hf_id": None,
        "architecture": "MoE",
        "expected_type": "cLM",
    },
    {
        "model": "SynCodonLM",
        "hf_id": None,
        "architecture": "Unknown",
        "expected_type": "cLM",
    },
]

def test_model_load(model_info):
    result = {
        "model": model_info["model"],
        "hf_id": model_info["hf_id"],
        "architecture": model_info["architecture"],
        "timestamp": datetime.now().isoformat(),
        "load_attempted": False,
        "load_success": False,
        "error_type": None,
        "error_message": None,
        "error_traceback": None,
        "tokenizer_test": None,
        "forward_test": None,
    }
    
    if model_info["hf_id"] is None:
        result["error_type"] = "NO_PUBLIC_WEIGHTS"
        result["error_message"] = "No HuggingFace model ID available; weights not publicly accessible"
        return result
    
    result["load_attempted"] = True
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_info["hf_id"], trust_remote_code=True)
        result["config_loaded"] = True
        result["model_type"] = getattr(config, 'model_type', 'UNKNOWN')
    except Exception as e:
        result["error_type"] = "CONFIG_LOAD_FAILED"
        result["error_message"] = str(e)[:200]
        result["error_traceback"] = traceback.format_exc()[:500]
        return result
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_info["hf_id"], trust_remote_code=True)
        
        test_codons = "ATG GCT TTT GAA CCG"
        tokens = tokenizer(test_codons, return_tensors="pt")
        decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        
        unk_count = sum(1 for t in decoded if t in ['[UNK]', '<unk>', '▁', ''])
        total = len(decoded)
        
        result["tokenizer_test"] = {
            "input": test_codons,
            "decoded_tokens": decoded[:20],
            "n_tokens": total,
            "n_unk": unk_count,
            "unk_ratio": round(unk_count / total, 3) if total > 0 else 1.0,
            "is_char_level": any(len(t) == 1 and t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>'] for t in decoded if t),
        }
        
        if unk_count / max(total, 1) > 0.5:
            result["error_type"] = "TOKENIZER_UNK"
            result["error_message"] = f"{unk_count}/{total} tokens are UNK ({unk_count/total:.1%})"
            return result
        
    except Exception as e:
        result["tokenizer_test"] = {"error": str(e)[:200]}
        result["error_type"] = "TOKENIZER_LOAD_FAILED"
        result["error_message"] = str(e)[:200]
        result["error_traceback"] = traceback.format_exc()[:500]
        return result
    
    try:
        from transformers import AutoModel
        import torch
        
        model = AutoModel.from_pretrained(model_info["hf_id"], trust_remote_code=True)
        
        with torch.no_grad():
            outputs = model(tokens["input_ids"][:1], attention_mask=tokens.get("attention_mask", torch.ones_like(tokens["input_ids"]))[:1])
        
        result["forward_test"] = {
            "output_shape": list(outputs.last_hidden_state.shape) if hasattr(outputs, 'last_hidden_state') else "N/A",
        }
        result["load_success"] = True
        result["error_type"] = None
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        result["error_type"] = "MODEL_LOAD_OR_FORWARD_FAILED"
        result["error_message"] = str(e)[:200]
        result["error_traceback"] = traceback.format_exc()[:500]
        result["forward_test"] = {"error": str(e)[:200]}
    
    return result

def main():
    import src.models.xformers_compat
    
    print("=" * 70)
    print("CodonBench: Systematic Model Exclusion Testing")
    print(f"Testing {len(EXCLUDED_MODELS)} excluded models")
    print("=" * 70)
    
    all_results = []
    
    for i, model_info in enumerate(EXCLUDED_MODELS):
        print(f"\n[{i+1}/{len(EXCLUDED_MODELS)}] Testing {model_info['model']}...")
        result = test_model_load(model_info)
        all_results.append(result)
        
        status = "✓ LOADED" if result["load_success"] else f"✗ {result['error_type']}"
        print(f"  Status: {status}")
        if result["error_message"]:
            print(f"  Error: {result['error_message'][:100]}")
        if result.get("tokenizer_test") and isinstance(result["tokenizer_test"], dict):
            tt = result["tokenizer_test"]
            if "unk_ratio" in tt:
                print(f"  UNK ratio: {tt['unk_ratio']:.1%}")
    
    n_success = sum(1 for r in all_results if r["load_success"])
    n_no_weights = sum(1 for r in all_results if r["error_type"] == "NO_PUBLIC_WEIGHTS")
    n_failed = len(all_results) - n_success - n_no_weights
    
    summary = {
        "total_tested": len(all_results),
        "n_successfully_loaded": n_success,
        "n_no_public_weights": n_no_weights,
        "n_failed_compatibility": n_failed,
        "success_rate": round(n_success / len(all_results), 3),
        "models": all_results,
    }
    
    out_path = Path("./results/excluded_models_detailed.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"\n{'='*70}")
    print(f"Summary: {n_success} loaded, {n_failed} failed, {n_no_weights} no weights")
    print(f"Success rate: {n_success}/{len(all_results)} = {n_success/len(all_results):.1%}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()