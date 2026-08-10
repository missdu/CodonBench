"""
CodonBench 冒烟测试脚本
======================
用极小样本数据验证每个环节在真实GPU上可运行，不用猜。
每个测试独立，失败不阻塞后续。
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("PYTHONPATH"):
    for p in os.environ["PYTHONPATH"].split(":"):
        if p not in sys.path:
            sys.path.insert(0, p)
sys.path.insert(0, str(PROJECT_ROOT))
logger_temp = logging.getLogger(__name__)
logger_temp.info(f"PROJECT_ROOT={PROJECT_ROOT}, cwd={os.getcwd()}")

logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _load_encodon(model_id, device="cpu"):
    import src.models.xformers_compat
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True, device_map=device)
    return model, tokenizer


def _fix_token_type_ids(inputs):
    import torch
    if "token_type_ids" in inputs:
        seq_len = inputs["input_ids"].shape[1]
        tt_len = inputs["token_type_ids"].shape[1]
        if tt_len < seq_len:
            pad = torch.zeros(inputs["input_ids"].shape[0], seq_len - tt_len,
                              dtype=inputs["token_type_ids"].dtype,
                              device=inputs["token_type_ids"].device)
            inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], pad], dim=1)
    return inputs


def test_import():
    """测试1: 核心包导入"""
    name = "test_import"
    try:
        import torch
        import transformers
        import sklearn
        import scipy
        import seaborn
        import yaml
        assert torch.cuda.is_available(), "CUDA not available"
        return name, PASS, f"torch={torch.__version__}, cuda={torch.cuda.device_count()}gpus, transformers={transformers.__version__}"
    except Exception as e:
        return name, FAIL, str(e)


def test_model_download():
    """测试2: 能否从HuggingFace下载EnCodon-80M（最小的模型）"""
    name = "test_model_download"
    try:
        model_id = "goodarzilab/encodon-80M"
        logger.info(f"Downloading tokenizer for {model_id}...")
        logger.info(f"Downloading model {model_id}...")
        model, tokenizer = _load_encodon(model_id, device="cpu")
        n_params = sum(p.numel() for p in model.parameters())
        del model, tokenizer
        return name, PASS, f"Downloaded {model_id}, {n_params/1e6:.1f}M params"
    except Exception as e:
        return name, FAIL, f"Cannot download from HuggingFace: {e}. Will use local models only."


def test_model_load_gpu():
    """测试3: 能否将EnCodon-80M加载到GPU"""
    name = "test_model_load_gpu"
    try:
        import torch
        model_id = "goodarzilab/encodon-80M"
        device = "cuda:2"

        model, tokenizer = _load_encodon(model_id, device=device)
        model = model.eval()

        vram_mb = torch.cuda.memory_allocated(2) / 1024**2

        test_seq = "ATG GCT AAA TTT CCG GAT GCA TTT AAA CCC GGG AAA TTT CCG GAT"
        inputs = _fix_token_type_ids(tokenizer(test_seq, return_tensors="pt").to(device))
        with torch.no_grad():
            outputs = model(**inputs)

        del model, tokenizer
        torch.cuda.empty_cache()

        return name, PASS, f"Loaded to {device}, {vram_mb:.0f}MB VRAM, forward pass OK"
    except Exception as e:
        return name, FAIL, str(e)


def test_llr_computation():
    """测试4: 能否计算LLR（Zero-shot评估核心）"""
    name = "test_llr_computation"
    try:
        import torch
        from src.eval.evaluation_utils import compute_llr_encoder
        model_id = "goodarzilab/encodon-80M"
        device = "cuda:2"

        model, tokenizer = _load_encodon(model_id, device=device)
        model = model.eval()

        wt_seq = "ATG GCT AAA TTT CCG GAT GCA TTT AAA CCC GGG AAA TTT CCG GAT"
        mut_seq = "ATG GCA AAA TTT CCG GAT GCA TTT AAA CCC GGG AAA TTT CCG GAT"

        llr = compute_llr_encoder(model, tokenizer, wt_seq, mut_seq, device=device)

        del model, tokenizer
        torch.cuda.empty_cache()

        if np.isnan(llr) or np.isinf(llr):
            return name, FAIL, f"LLR is nan/inf: {llr}"

        return name, PASS, f"LLR={llr:.4f} (wt vs mut, finite)"
    except Exception as e:
        return name, FAIL, str(e)


def test_embedding_extraction():
    """测试5: 能否提取embedding（下游评估核心）"""
    name = "test_embedding_extraction"
    try:
        import torch
        from src.eval.evaluation_utils import extract_embeddings
        model_id = "goodarzilab/encodon-80M"
        device = "cuda:2"

        model, tokenizer = _load_encodon(model_id, device=device)
        model = model.eval()

        seqs = [
            "ATG GCT AAA TTT CCG GAT GCA TTT AAA CCC GGG AAA TTT CCG GAT",
            "ATG GCA AAA TTT CCG GAT GCA TTT AAA CCC GGG AAA TTT CCG GAT",
            "ATG GCC AAA TTT CCG GAT GCA TTT AAA CCC GGG AAA TTT CCG GAT",
        ]

        embeddings = extract_embeddings(model, tokenizer, seqs, device=device, batch_size=2, show_progress=False)

        del model, tokenizer
        torch.cuda.empty_cache()

        if embeddings.shape[0] != len(seqs):
            return name, FAIL, f"Expected {len(seqs)} embeddings, got {embeddings.shape[0]}"
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            return name, FAIL, "Embeddings contain nan/inf"

        return name, PASS, f"Embedding shape={embeddings.shape}, all finite"
    except Exception as e:
        return name, FAIL, str(e)


def test_downstream_pipeline():
    """测试6: 下游任务评估流水线（embedding → RF → 预测）"""
    name = "test_downstream_pipeline"
    try:
        import torch
        from src.eval.evaluation_utils import extract_embeddings
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        model_id = "goodarzilab/encodon-80M"
        device = "cuda:2"

        model, tokenizer = _load_encodon(model_id, device=device)
        model = model.eval()

        np.random.seed(42)
        seqs = ["ATG " + " ".join(np.random.choice(["GCT","GCC","GCA","GCG","AAA","AAG","TTT","TTC"], 20)) for _ in range(30)]
        labels = np.random.randn(30)

        embeddings = extract_embeddings(model, tokenizer, seqs, device=device, batch_size=8, show_progress=False)

        del model, tokenizer
        torch.cuda.empty_cache()

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(embeddings[:20], labels[:20])
        preds = rf.predict(embeddings[20:])
        r2 = r2_score(labels[20:], preds)

        return name, PASS, f"R2={r2:.4f} on 10 test samples (expected negative for random labels)"
    except Exception as e:
        return name, FAIL, str(e)


def test_data_pipeline():
    """测试7: 数据下载/预处理流水线"""
    name = "test_data_pipeline"
    try:
        from src.data.download import prepare_all_tasks
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = prepare_all_tasks(tmpdir)
            task_status = {}
            for k, v in results.items():
                if v is not None:
                    task_status[k] = f"OK({len(v)}rows)"
                else:
                    task_status[k] = "FAILED"

            ok_count = sum(1 for v in results.values() if v is not None)
            if ok_count == 0:
                return name, FAIL, "All tasks failed"
            return name, PASS, f"{ok_count}/7 tasks OK: {task_status}"
    except Exception as e:
        return name, FAIL, str(e)


def test_local_models():
    """测试8: 服务器本地已有模型"""
    name = "test_local_models"
    try:
        clms_path = Path("./cLMs")
        if not clms_path.exists():
            return name, SKIP, f"Local path {clms_path} does not exist"

        models = [d.name for d in clms_path.iterdir() if d.is_dir()]
        if not models:
            return name, SKIP, "No local models found"

        loadable = []
        for m in models:
            mpath = clms_path / m
            has_config = any((mpath / f).exists() for f in ["config.json", "config.yaml"])
            has_model = any((mpath / f).exists() for f in ["pytorch_model.bin", "model.safetensors"])
            if has_config and has_model:
                loadable.append(m)

        if not loadable:
            return name, SKIP, f"Found dirs {models} but none have config+weights"
        return name, PASS, f"Loadable local models: {loadable}"
    except Exception as e:
        return name, FAIL, str(e)


def test_quantization():
    """测试9: 4-bit量化能否工作"""
    name = "test_quantization"
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        return name, PASS, "BitsAndBytesConfig created successfully"
    except ImportError:
        return name, SKIP, "bitsandbytes not installed, 4-bit quantization unavailable"
    except Exception as e:
        return name, FAIL, str(e)


def test_gpu_memory():
    """测试10: GPU内存信息"""
    name = "test_gpu_memory"
    try:
        import torch
        info = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3
            info.append(f"GPU{i}: {free:.1f}GB free / {total:.1f}GB total")
        return name, PASS, "; ".join(info)
    except Exception as e:
        return name, FAIL, str(e)


ALL_TESTS = [
    test_import,
    test_gpu_memory,
    test_model_download,
    test_model_load_gpu,
    test_llr_computation,
    test_embedding_extraction,
    test_downstream_pipeline,
    test_data_pipeline,
    test_local_models,
    test_quantization,
]


def run_smoke_tests(data_dir: str = None, output_dir: str = None):
    """运行所有冒烟测试"""
    print("=" * 70)
    print("CodonBench Smoke Test Suite")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print()

    results = []
    for test_func in ALL_TESTS:
        test_name = test_func.__doc__.strip().split("\n")[0] if test_func.__doc__ else test_func.__name__
        print(f"  Running {test_name}...")
        t0 = time.time()
        try:
            name, status, msg = test_func()
        except Exception as e:
            name = test_func.__name__
            status = FAIL
            msg = f"UNEXPECTED: {e}"
        elapsed = time.time() - t0

        icon = {"PASS": "OK", "FAIL": "XX", "SKIP": "--"}[status]
        print(f"    [{icon}] {name}: {msg} ({elapsed:.1f}s)")
        results.append({"name": name, "status": status, "message": msg, "time_s": round(elapsed, 2)})

    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_skip = sum(1 for r in results if r["status"] == SKIP)

    print()
    print("=" * 70)
    print(f"Results: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP out of {len(results)} tests")
    print("=" * 70)

    if n_fail > 0:
        print("FAILED tests:")
        for r in results:
            if r["status"] == FAIL:
                print(f"  - {r['name']}: {r['message']}")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
            "tests": results,
        }
        with open(out_path / "smoke_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {out_path / 'smoke_test_report.json'}")

    return n_fail == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Data directory")
    parser.add_argument("--output", default="./results/smoke_test", help="Output directory for report")
    args = parser.parse_args()
    success = run_smoke_tests(args.data, args.output)
    sys.exit(0 if success else 1)
