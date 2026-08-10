import types
import sys

try:
    import xformers
except ImportError:
    xformers_mock = types.ModuleType("xformers")
    xops_mock = types.ModuleType("xformers.ops")

    class _LowerTriangularMask:
        def materialize(self, shape=None, device=None, dtype=None):
            import torch
            if shape is None:
                return None
            seq_len = shape[-1]
            return torch.triu(torch.full(shape, float("-inf"), device=device, dtype=dtype), diagonal=1)

    xops_mock.LowerTriangularMask = _LowerTriangularMask

    def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
        import torch
        import math
        # xformers input: [batch, seq_q, heads, head_dim]
        # Need: [batch, heads, seq_q, head_dim] for standard attention
        q2 = q.transpose(1, 2)  # [batch, heads, seq_q, head_dim]
        k2 = k.transpose(1, 2)  # [batch, heads, seq_k, head_dim]
        v2 = v.transpose(1, 2)  # [batch, heads, seq_k, head_dim]

        if scale is None:
            scale = 1.0 / math.sqrt(q2.shape[-1])
        attn = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if attn_bias is not None:
            if isinstance(attn_bias, _LowerTriangularMask):
                causal_mask = torch.triu(
                    torch.full(attn.shape, float("-inf"), device=attn.device, dtype=attn.dtype),
                    diagonal=1
                )
                attn = attn + causal_mask
            elif hasattr(attn_bias, 'shape') and attn_bias.shape == attn.shape:
                attn = attn + attn_bias
            else:
                try:
                    materialized = attn_bias.materialize(attn.shape, device=attn.device, dtype=attn.dtype)
                    if materialized is not None:
                        attn = attn + materialized
                except Exception:
                    pass

        attn = torch.softmax(attn, dim=-1)
        if p > 0:
            attn = torch.nn.functional.dropout(attn, p=p, training=True)
        out = torch.matmul(attn, v2)  # [batch, heads, seq_q, head_dim]
        return out.transpose(1, 2)  # [batch, seq_q, heads, head_dim]

    xops_mock.memory_efficient_attention = _memory_efficient_attention
    xformers_mock.ops = xops_mock

    sys.modules["xformers"] = xformers_mock
    sys.modules["xformers.ops"] = xops_mock
