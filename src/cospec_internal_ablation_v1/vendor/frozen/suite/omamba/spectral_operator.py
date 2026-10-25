"""Ridge spectral coefficients + trainable refinement + measurement support.

The fixed hat basis and ridge inverse are established numerical methods.
The resolution diagonal is an empirical support score, not a calibrated
posterior variance or a guarantee that a missing absorption feature is known.
"""
import torch
from torch import nn


class SpectralOperator(nn.Module):
    def __init__(self, wavelengths, dim=64, basis_size=64, ridge=0.01):
        super().__init__()
        wave = torch.as_tensor(wavelengths, dtype=torch.float32)
        if (wave.ndim != 1 or len(wave) < 2 or not torch.isfinite(wave).all()
                or not torch.all(wave[1:] > wave[:-1])):
            raise ValueError("需要有限且严格递增的光谱坐标；坐标模式和单位由实验元数据明确记录。")
        if basis_size < 2 or ridge <= 0:
            raise ValueError("basis_size >= 2，ridge > 0。")
        centers = torch.linspace(wave[0], wave[-1], basis_size)
        spacing = (centers[-1] - centers[0]) / (basis_size - 1)
        phi = (1 - (wave[:, None] - centers[None, :]).abs() / spacing).clamp(min=0)
        phi = phi / phi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # Retain the state key for the existing architecture. In this experiment
        # its values are dimensionless relative coordinates, not physical nm.
        self.register_buffer("wavelengths", wave)
        self.register_buffer("phi", phi)
        self.register_buffer("identity", torch.eye(basis_size))
        self.ridge = ridge
        full_fisher = phi.T @ phi
        full_resolution = torch.linalg.solve(full_fisher + ridge * torch.eye(basis_size), full_fisher)
        self.register_buffer("full_geometry_support", full_resolution.diagonal().mean())
        self.refiner = nn.Linear(basis_size, basis_size)
        nn.init.zeros_(self.refiner.weight)
        nn.init.zeros_(self.refiner.bias)
        self.projection = nn.Sequential(nn.Linear(basis_size, dim), nn.SiLU(), nn.LayerNorm(dim))

    def forward(self, spectra, observed):
        # spectra [B,S,C], observed [B,C]; all matrix solves use float32.
        if spectra.ndim != 3 or observed.shape != (spectra.shape[0], spectra.shape[2]):
            raise ValueError("spectra 应是 [B,S,C]，observed 应是 [B,C]。")
        if observed.dtype != torch.bool:
            raise ValueError("v1 observed 必须是布尔掩码；不接受未定义的带宽响应权重。")
        if spectra.shape[-1] != self.phi.shape[0] or torch.any(observed.sum(-1) == 0):
            raise ValueError("波段数不匹配或存在没有任何观测的样本。")
        mask = observed.to(dtype=torch.float32)
        phi = self.phi.float()
        fisher = torch.einsum("ck,bc,cl->bkl", phi, mask, phi)
        chol = torch.linalg.cholesky(fisher + self.ridge * self.identity)
        transfer = torch.cholesky_solve(phi.T.unsqueeze(0) * mask[:, None, :], chol)
        coeff = torch.einsum("bkc,bsc->bsk", transfer, spectra.float())
        resolution = torch.cholesky_solve(fisher, chol)
        support = resolution.diagonal(dim1=-2, dim2=-1).clamp(0, 1)
        # A bounded learned correction restricted by the soft resolution operator.
        correction = 0.1 * torch.tanh(self.refiner(coeff))
        refined = coeff + torch.einsum("bkl,bsl->bsk", resolution, correction)
        reconstruction = torch.einsum("ck,bsk->bsc", phi, refined)
        features = self.projection(refined)
        return {
            "features": features, "coefficients": refined, "support": support,
            "reconstruction": reconstruction,
            "geometry_support": support.mean(dim=-1),
            "relative_geometry_support": (support.mean(-1) / self.full_geometry_support).clamp(0, 1),
        }
