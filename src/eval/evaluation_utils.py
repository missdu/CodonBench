import torch
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def _fix_token_type_ids(inputs):
    if "token_type_ids" in inputs:
        seq_len = inputs["input_ids"].shape[1]
        tt_len = inputs["token_type_ids"].shape[1]
        if tt_len < seq_len:
            pad = torch.zeros(inputs["input_ids"].shape[0], seq_len - tt_len,
                              dtype=inputs["token_type_ids"].dtype,
                              device=inputs["token_type_ids"].device)
            inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], pad], dim=1)
    return inputs


def compute_sequence_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        token_ll = torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        token_ll = token_ll * attention_mask
        return token_ll.sum(dim=-1)


def compute_llr_encoder(
    model,
    tokenizer,
    wt_seq: str,
    mut_seq: str,
    device: str = "cuda:0",
    max_length: int = 2048,
) -> float:
    with torch.no_grad():
        wt_inputs = _fix_token_type_ids(tokenizer(
            wt_seq, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        ).to(device))
        mut_inputs = _fix_token_type_ids(tokenizer(
            mut_seq, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        ).to(device))

        wt_outputs = model(**wt_inputs, output_hidden_states=False)
        mut_outputs = model(**mut_inputs, output_hidden_states=False)

        wt_logits = wt_outputs.logits if hasattr(wt_outputs, "logits") else wt_outputs[0]
        mut_logits = mut_outputs.logits if hasattr(mut_outputs, "logits") else mut_outputs[0]

        wt_labels = wt_inputs["input_ids"][:, 1:]
        mut_labels = mut_inputs["input_ids"][:, 1:]
        wt_mask = wt_inputs["attention_mask"][:, 1:]
        mut_mask = mut_inputs["attention_mask"][:, 1:]
        wt_logits_shifted = wt_logits[:, :-1, :]
        mut_logits_shifted = mut_logits[:, :-1, :]

        wt_ll = compute_sequence_log_likelihood(wt_logits_shifted, wt_labels, wt_mask)
        mut_ll = compute_sequence_log_likelihood(mut_logits_shifted, mut_labels, mut_mask)

        return (mut_ll - wt_ll).item()


def compute_llr_decoder(
    model,
    tokenizer,
    wt_seq: str,
    mut_seq: str,
    device: str = "cuda:0",
    max_length: int = 2048,
) -> float:
    with torch.no_grad():
        wt_inputs = _fix_token_type_ids(tokenizer(
            wt_seq, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        ).to(device))
        mut_inputs = _fix_token_type_ids(tokenizer(
            mut_seq, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        ).to(device))

        wt_outputs = model(**wt_inputs)
        mut_outputs = model(**mut_inputs)

        wt_labels = wt_inputs["input_ids"]
        mut_labels = mut_inputs["input_ids"]
        wt_mask = wt_inputs["attention_mask"]
        mut_mask = mut_inputs["attention_mask"]

        wt_ll = compute_sequence_log_likelihood(wt_outputs.logits[:, :-1, :], wt_labels[:, 1:], wt_mask[:, 1:])
        mut_ll = compute_sequence_log_likelihood(mut_outputs.logits[:, :-1, :], mut_labels[:, 1:], mut_mask[:, 1:])

        return (mut_ll - wt_ll).item()


def extract_embeddings(
    model,
    tokenizer,
    sequences: List[str],
    device: str = "cuda:0",
    layer: int = -2,
    batch_size: int = 8,
    max_length: int = 2048,
    show_progress: bool = True,
) -> np.ndarray:
    from tqdm import tqdm

    all_embeddings = []

    def _collate(seqs):
        return _fix_token_type_ids(tokenizer(
            seqs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device))

    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting embeddings")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            inputs = _collate(batch_seqs)

            model_inputs = {k: v for k, v in inputs.items() if k in model.forward.__code__.co_varnames or k in ('input_ids', 'attention_mask')}

            try:
                outputs = model(**model_inputs, output_hidden_states=True)
            except TypeError:
                try:
                    outputs = model(**model_inputs)
                except TypeError:
                    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                if not hasattr(outputs, "hidden_states"):
                    if hasattr(outputs, "last_hidden_state"):
                        hidden = outputs.last_hidden_state
                    else:
                        hidden = outputs[0]
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                    all_embeddings.append(pooled.cpu().numpy())
                    continue

            hidden = outputs.hidden_states[layer]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)
