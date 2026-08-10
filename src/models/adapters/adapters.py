"""
模型适配器：处理不同cLM的forward pass格式差异
每个适配器封装特定模型的tokenizer调用和输出提取逻辑
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class BaseModelAdapter:
    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def tokenize(self, sequence: str, max_length: int = 2048) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            sequence, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        with torch.no_grad():
            return self.model(**inputs)

    def get_logits(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        raise ValueError(f"Cannot extract logits from output type {type(outputs)}")

    def get_hidden_states(self, outputs: Any, layer: int = -2) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[layer]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        raise ValueError(f"Cannot extract hidden states from output type {type(outputs)}")

    def compute_llr(self, wt_seq: str, mut_seq: str, max_length: int = 2048) -> float:
        wt_inputs = self.tokenize(wt_seq, max_length)
        mut_inputs = self.tokenize(mut_seq, max_length)

        wt_outputs = self.forward(wt_inputs)
        mut_outputs = self.forward(mut_inputs)

        wt_logits = self.get_logits(wt_outputs)
        mut_logits = self.get_logits(mut_outputs)

        wt_labels = wt_inputs["input_ids"][:, 1:]
        mut_labels = mut_inputs["input_ids"][:, 1:]
        wt_mask = wt_inputs["attention_mask"][:, 1:]
        mut_mask = mut_inputs["attention_mask"][:, 1:]

        from src.eval.evaluation_utils import compute_sequence_log_likelihood
        wt_ll = compute_sequence_log_likelihood(wt_logits[:, :-1, :], wt_labels, wt_mask)
        mut_ll = compute_sequence_log_likelihood(mut_logits[:, :-1, :], mut_labels, mut_mask)

        return (mut_ll - wt_ll).item()

    def extract_embedding(self, sequence: str, layer: int = -2, max_length: int = 2048) -> np.ndarray:
        inputs = self.tokenize(sequence, max_length)
        try:
            outputs = self.forward({**inputs, "output_hidden_states": True})
        except TypeError:
            outputs = self.forward(inputs)

        hidden = self.get_hidden_states(outputs, layer)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.cpu().numpy().squeeze()


class EncoderAdapter(BaseModelAdapter):
    pass


class DecoderAdapter(BaseModelAdapter):
    def compute_llr(self, wt_seq: str, mut_seq: str, max_length: int = 2048) -> float:
        wt_inputs = self.tokenize(wt_seq, max_length)
        mut_inputs = self.tokenize(mut_seq, max_length)

        wt_outputs = self.forward(wt_inputs)
        mut_outputs = self.forward(mut_inputs)

        from src.eval.evaluation_utils import compute_sequence_log_likelihood
        wt_ll = compute_sequence_log_likelihood(
            wt_outputs.logits[:, :-1, :],
            wt_inputs["input_ids"][:, 1:],
            wt_inputs["attention_mask"][:, 1:],
        )
        mut_ll = compute_sequence_log_likelihood(
            mut_outputs.logits[:, :-1, :],
            mut_inputs["input_ids"][:, 1:],
            mut_inputs["attention_mask"][:, 1:],
        )
        return (mut_ll - wt_ll).item()


class CaLMAdapter(BaseModelAdapter):
    def tokenize(self, sequence: str, max_length: int = 2048) -> Dict[str, torch.Tensor]:
        if " " not in sequence:
            codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
            sequence = " ".join(codons)
        return super().tokenize(sequence, max_length)

    def get_logits(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            return outputs.logits
        if hasattr(outputs, "prediction_logits"):
            return outputs.prediction_logits
        if isinstance(outputs, tuple):
            for o in outputs:
                if isinstance(o, torch.Tensor) and o.dim() == 3:
                    return o
        return super().get_logits(outputs)


class CodonBERTAdapter(BaseModelAdapter):
    def get_logits(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            return outputs.logits
        if hasattr(outputs, "prediction_logits"):
            return outputs.prediction_logits
        return super().get_logits(outputs)


class CdsBERTAdapter(BaseModelAdapter):
    def tokenize(self, sequence: str, max_length: int = 2048) -> Dict[str, torch.Tensor]:
        if " " not in sequence:
            codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
            sequence = " ".join(codons)
        return super().tokenize(sequence, max_length)


class EquiMRNAAdapter(BaseModelAdapter):
    def get_hidden_states(self, outputs: Any, layer: int = -2) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[layer]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "encoder_hidden_states"):
            return outputs.encoder_hidden_states[layer]
        return super().get_hidden_states(outputs, layer)


class MoEAdapter(BaseModelAdapter):
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        with torch.no_grad():
            try:
                return self.model(**inputs)
            except TypeError:
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
                return self.model(input_ids=input_ids, attention_mask=attention_mask)


ADAPTER_REGISTRY = {
    "encodon-80m": EncoderAdapter,
    "encodon-200m": EncoderAdapter,
    "encodon-1b": EncoderAdapter,
    "decodon-200m": DecoderAdapter,
    "decodon-1b": DecoderAdapter,
    "calm": CaLMAdapter,
    "codonbert": CodonBERTAdapter,
    "cdsbert": CdsBERTAdapter,
    "helm": EncoderAdapter,
    "syncodonlm": DecoderAdapter,
    "codonmoe": MoEAdapter,
    "equi_mrna": EquiMRNAAdapter,
    "lifecode": EncoderAdapter,
    "biolangfusion": EncoderAdapter,
}


def get_adapter(model_name: str, model, tokenizer, device: str = "cuda:0") -> BaseModelAdapter:
    adapter_cls = ADAPTER_REGISTRY.get(model_name, EncoderAdapter)
    logger.info(f"Using {adapter_cls.__name__} for {model_name}")
    return adapter_cls(model, tokenizer, device)
