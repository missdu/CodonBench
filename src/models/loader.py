import os
import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List

import yaml
import torch
import numpy as np

logger = logging.getLogger(__name__)

HOME = Path(os.path.expanduser("~"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class ModelConfig:
    name: str
    hf_id: Optional[str] = None
    local_path: Optional[str] = None
    architecture: str = "encoder"
    params_M: int = 100
    pretrain_Mb: int = 100
    tokenizer_type: str = "codon"
    trust_remote_code: bool = True
    quantize_4bit: bool = False
    priority: int = 1
    is_mixtral: bool = False

    def get_model_path(self) -> str:
        if self.local_path:
            p = Path(os.path.expanduser(self.local_path))
            if p.exists():
                return str(p)
        if self.hf_id:
            return self.hf_id
        if self.local_path:
            return os.path.expanduser(self.local_path)
        raise FileNotFoundError(f"Cannot find model {self.name}")


def load_configs() -> Dict[str, ModelConfig]:
    cfg_path = CONFIGS_DIR / "models.yaml"
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    configs = {}
    for name, info in raw.get("models", {}).items():
        info_copy = dict(info)
        info_copy["name"] = name
        configs[name] = ModelConfig(**{k: v for k, v in info_copy.items() if k in ModelConfig.__dataclass_fields__})
    return configs


class CodonModelLoader:
    MODEL_CONFIGS: Dict[str, ModelConfig] = {}

    @classmethod
    def load_configs(cls):
        if not cls.MODEL_CONFIGS:
            cls.MODEL_CONFIGS = load_configs()
        return cls.MODEL_CONFIGS

    @classmethod
    def list_models(cls) -> List[str]:
        cfgs = cls.load_configs()
        return sorted(cfgs.keys(), key=lambda k: cfgs[k].priority)

    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        cfgs = cls.load_configs()
        if model_name not in cfgs:
            raise KeyError(f"Model '{model_name}' not in registry. Available: {list(cfgs.keys())}")
        return cfgs[model_name]

    @classmethod
    def load(
        cls,
        model_name: str,
        device: str = "cuda:0",
        quantize_override: Optional[bool] = None,
        max_retries: int = 3,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        config = cls.get_config(model_name)
        model_path = config.get_model_path()
        should_quantize = quantize_override if quantize_override is not None else config.quantize_4bit

        meta = {
            "model_name": model_name,
            "model_path": model_path,
            "architecture": config.architecture,
            "params_M": config.params_M,
            "quantized": should_quantize,
            "device": device,
            "load_time_s": 0,
            "vram_mb": 0,
            "success": False,
            "error": None,
        }

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Loading {model_name} (attempt {attempt}/{max_retries}) from {model_path}")
                t0 = time.time()

                model, tokenizer = cls._load_model_tokenizer(
                    model_path, config, should_quantize, device
                )

                elapsed = time.time() - t0
                vram = 0
                if device.startswith("cuda"):
                    vram = torch.cuda.memory_allocated(int(device.split(":")[1]) if ":" in device else 0) / 1024**2

                model.eval()
                meta.update({"load_time_s": round(elapsed, 2), "vram_mb": round(vram, 1), "success": True})
                logger.info(f"Loaded {model_name}: {elapsed:.1f}s, {vram:.0f}MB VRAM")
                return model, tokenizer, meta

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM loading {model_name} on attempt {attempt}")
                torch.cuda.empty_cache()
                if attempt == 1 and not should_quantize:
                    logger.info(f"Retrying with 4-bit quantization for {model_name}")
                    should_quantize = True
                    meta["quantized"] = True
                    continue
                if attempt < max_retries:
                    continue
                meta["error"] = f"OOM after {max_retries} attempts"

            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
                meta["error"] = str(e)
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                break

        return None, None, meta

    @classmethod
    def _load_model_tokenizer(
        cls, path: str, config: ModelConfig, quantize: bool, device: str
    ) -> Tuple[Any, Any]:
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, BitsAndBytesConfig
        import src.models.xformers_compat

        is_decoder = config.architecture in ("decoder",)
        is_mlm = "encodon" in config.name.lower() or "codonfm" in config.name.lower() or "codonbert" in config.name.lower() or "cdsbert" in config.name.lower()
        if config.is_mixtral:
            auto_model_cls = AutoModel
        elif is_decoder:
            auto_model_cls = AutoModelForCausalLM
        elif is_mlm:
            auto_model_cls = AutoModelForMaskedLM
        else:
            auto_model_cls = AutoModel

        tokenizer_kwargs = {"trust_remote_code": config.trust_remote_code}
        model_kwargs = {"trust_remote_code": config.trust_remote_code}

        if quantize:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        model = auto_model_cls.from_pretrained(path, **model_kwargs)

        if not quantize:
            model = model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    @classmethod
    def release(cls, model, device: str = "cuda:0"):
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        import gc
        gc.collect()
