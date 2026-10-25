"""Mamba-1-style selective SSM with explicit control of the write term.

Math/interface reference: state-spaces/mamba mamba_simple.py and
selective_scan_interface.py (authors Albert Gu / Tri Dao).
This is a small independent implementation, not a patched installed package.
The reference backend executes the actual selective recurrence, not a dummy
sequence network. CUDA uses the installed mamba_ssm selective_scan_fn.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def scan_reference(u, delta, A, B, C, D, z):
    """All delta values already softplus-transformed; inputs float32."""
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    output = []
    for t in range(u.shape[-1]):
        decay = torch.exp(delta[:, :, t, None] * A[None, :, :])
        write = delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        state = decay * state + write
        read = (state * C[:, None, :, t]).sum(-1) + D[None, :] * u[:, :, t]
        output.append(read * F.silu(z[:, :, t]))
    return torch.stack(output, dim=-1)


class EvidenceMamba(nn.Module):
    def __init__(self, dim, d_state=24, d_conv=4, expand=1, backend="cuda"):
        super().__init__()
        if backend not in ("cuda", "reference") or min(dim, d_state, d_conv, expand) < 1:
            raise ValueError("无效的 Mamba 参数或 backend。")
        self.dim, self.d_state = dim, d_state
        self.inner = dim * expand
        self.rank = math.ceil(dim / 16)
        self.backend = backend
        self.in_proj = nn.Linear(dim, self.inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.inner, self.inner, d_conv, groups=self.inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.inner, self.rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.rank, self.inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -self.rank ** -0.5, self.rank ** -0.5)
        dt = torch.exp(torch.rand(self.inner) * math.log(0.1 / 0.001) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log()
                                  .expand(self.inner, -1).clone())
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, dim, bias=False)

    def forward(self, tokens, write_weight=None, forget_strength=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("tokens 应为 [batch, length, dim]。")
        batch, length, _ = tokens.shape
        x, z = self.in_proj(tokens).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])
        dt, B, C = torch.split(self.x_proj(x.transpose(1, 2)),
                               [self.rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).transpose(1, 2).contiguous().float()
        B = B.transpose(1, 2).contiguous().float()
        C = C.transpose(1, 2).contiguous().float()
        if write_weight is not None:
            if write_weight.shape != (batch, length):
                raise ValueError("write_weight 应为 [batch, length]。")
            # Changes B in the recurrence. A/delta and output/read gates remain
            # unchanged. The D skip and outer residual are NOT controlled by it.
            B = (B * write_weight[:, None, :]).contiguous()
        if forget_strength is not None:
            if forget_strength.shape != (batch, length):
                raise ValueError("forget_strength 应为 [batch, length]。")
            # delta'=(1+s)*delta, B'=B/(1+s): change state decay while
            # preserving the original write delta*B*u exactly. Both scan
            # backends implement this; D skip and outer residual stay local.
            ratio = 1.0 + forget_strength[:, None, :].float()
            delta = (delta * ratio).contiguous()
            B = (B / ratio).contiguous()
        A = -torch.exp(self.A_log.float())
        if self.backend == "reference":
            y = scan_reference(x.float(), delta, A, B, C, self.D.float(), z.float())
        else:
            if not tokens.is_cuda:
                raise RuntimeError("CUDA backend 需要 CUDA tensor；不会自动切换到替代模型。")
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            except ImportError as error:
                raise RuntimeError("未能导入 mamba_ssm 的选择性扫描；请使用原 convmamba 环境。") from error
            y = selective_scan_fn(x.float().contiguous(), delta, A, B, C,
                                  self.D.float(), z=z.float().contiguous(),
                                  delta_softplus=False)
        return self.out_proj(y.transpose(1, 2))


class EvidenceBlock(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand, backend, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = EvidenceMamba(dim, d_state, d_conv, expand, backend)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, write_weight, forget_strength=None):
        return tokens + self.dropout(self.mamba(self.norm(tokens), write_weight, forget_strength))
