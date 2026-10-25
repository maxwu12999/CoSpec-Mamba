"""Same parameters for plain / controlled / multiview ablations."""
import torch
from torch import nn

from .ambiguity import AmbiguityEstimator
from .evidence_mamba import EvidenceBlock
from .spectral_operator import SpectralOperator


class ObservationMamba(nn.Module):
    def __init__(self, wavelengths, class_means, class_variances, patch_size=15,
                 dim=64, depth=4, basis_size=64, ridge=0.01, d_state=24,
                 d_conv=4, expand=1, dropout=0.1, backend="cuda", write_floor=0.05):
        super().__init__()
        if patch_size % 2 != 1 or patch_size < 1 or depth < 1 or not 0 <= write_floor < 1:
            raise ValueError("无效的模型设置。")
        self.patch_size, self.write_floor = patch_size, write_floor
        self.operator = SpectralOperator(wavelengths, dim, basis_size, ridge)
        self.ambiguity = AmbiguityEstimator(class_means, class_variances)
        axis = torch.linspace(-1, 1, patch_size)
        row, col = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("positions", torch.stack((row, col), -1).reshape(-1, 2))
        self.position_encoder = nn.Linear(2, dim, bias=False)
        self.layers = nn.ModuleList([EvidenceBlock(dim, d_state, d_conv, expand, backend, dropout)
                                     for _ in range(depth)])
        self.pool_logits = nn.Parameter(torch.zeros(patch_size * patch_size))
        self.readout = nn.Linear(2 * dim, dim)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(),
                                        nn.Dropout(dropout), nn.Linear(2 * dim, len(class_means)))

    def forward(self, patch, observed, controlled=True):
        if patch.ndim != 4 or patch.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError("patch 应为 [B,C,patch_size,patch_size]。")
        spectra = patch.flatten(2).transpose(1, 2)
        output = self.operator(spectra, observed)
        evidence = self.ambiguity(spectra, observed)
        reliability = (output["relative_geometry_support"][:, None] *
                       evidence["retained_information"]).clamp(0, 1)
        proposed_weight = self.write_floor + (1 - self.write_floor) * reliability
        actual_weight = proposed_weight if controlled else torch.ones_like(proposed_weight)
        tokens = output["features"] + self.position_encoder(self.positions)[None, :, :]
        for block in self.layers:
            tokens = block(tokens, actual_weight)
        center = tokens[:, tokens.shape[1] // 2]
        pooled = torch.einsum("s,bsd->bd", self.pool_logits.softmax(0), tokens)
        logits = self.classifier(self.readout(torch.cat((center, pooled), dim=-1)))
        output.update(evidence)
        output.update({
            "logits": logits, "write_weight": actual_weight,
            "proposed_write_weight": proposed_weight, "input_spectra": spectra,
            "observed": observed, "retained_evidence": reliability,
        })
        return output
