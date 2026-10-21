"""Coordinate-aligned scans, observed structure cues and regional refinement.

This is an ablation prototype, not a claim of established novelty. Boundary
scores and measurement support are empirical cues, not calibrated semantic
boundaries or posterior uncertainty. No test labels enter these operators.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoder_comparison import EncoderComparison


VARIANTS = {
    "baseline": {"scan": False, "structure": False, "boundary": False},
    "scan": {"scan": True, "structure": False, "boundary": False},
    "scan_structure": {"scan": True, "structure": True, "boundary": False},
    "scan_boundary": {"scan": True, "structure": False, "boundary": True},
    "full": {"scan": True, "structure": True, "boundary": True},
}
SCAN_NAMES = ("horizontal_snake", "reverse_horizontal_snake",
              "vertical_snake", "reverse_vertical_snake")


def scan_permutations(height, width):
    horizontal = [r * width + c for r in range(height)
                  for c in (range(width) if r % 2 == 0 else range(width - 1, -1, -1))]
    vertical = [r * width + c for c in range(width)
                for r in (range(height) if c % 2 == 0 else range(height - 1, -1, -1))]
    orders = torch.tensor([horizontal, horizontal[::-1], vertical, vertical[::-1]],
                          dtype=torch.long)
    return orders, orders.argsort(dim=-1)


def neighborhood_indices(size):
    indices, valid = [], []
    for row in range(size):
        for col in range(size):
            neighbors, flags = [], []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                ok = 0 <= rr < size and 0 <= cc < size
                neighbors.append(rr * size + cc if ok else row * size + col)
                flags.append(ok)
            indices.append(neighbors)
            valid.append(flags)
    return torch.tensor(indices), torch.tensor(valid)


class StructureCue(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        neighbors, valid = neighborhood_indices(patch_size)
        self.register_buffer("neighbors", neighbors)
        self.register_buffer("valid", valid)
        axis = torch.linspace(-1, 1, patch_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("radius", (xx.square() + yy.square()).sqrt().flatten() / math.sqrt(2))
        self.raw_spectral_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.importance_head = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1))

    def forward(self, clean_spectra, features, observed, reliability):
        # The SAME observation mask applies to all pixels in each patch.
        # Missing values have already been removed with torch.where.
        difference = clean_spectra[:, self.neighbors] - clean_spectra[:, :, None, :]
        denominator = observed.sum(-1).to(difference.dtype)[:, None, None]
        measured_gap = ((difference.square().sum(-1) / denominator + 1e-8).sqrt()
                        - 1e-4).clamp_min(0)
        measured_gap = torch.tanh(F.softplus(self.raw_spectral_scale) * measured_gap)
        normalized = F.normalize(features, dim=-1, eps=1e-6)
        feature_gap = (1 - (normalized[:, self.neighbors] * normalized[:, :, None, :]).sum(-1))
        feature_gap = feature_gap.clamp(0, 2) / 2
        gaps = 0.5 * (measured_gap + feature_gap)
        valid = self.valid.to(features.dtype)[None, :, :]
        local_change = (gaps * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        center = normalized[:, features.shape[1] // 2:features.shape[1] // 2 + 1]
        center_difference = ((1 - (normalized * center).sum(-1)).clamp(0, 2) / 2)
        support = reliability[:, None].expand_as(local_change)
        cues = torch.stack((local_change, center_difference, support,
                            self.radius[None, :].expand_as(local_change)), dim=-1)
        importance = 0.5 + self.importance_head(cues).squeeze(-1).sigmoid()
        return {"boundary_score": local_change * support, "importance": importance,
                "neighbor_gaps": gaps, "reliability": reliability}

    def transition_gap(self, clean_spectra, features, observed, order):
        spectra = clean_spectra.index_select(1, order)
        difference = spectra[:, 1:] - spectra[:, :-1]
        denominator = observed.sum(-1).to(spectra.dtype)[:, None]
        measured = ((difference.square().sum(-1) / denominator + 1e-8).sqrt()
                    - 1e-4).clamp_min(0)
        measured = torch.tanh(F.softplus(self.raw_spectral_scale) * measured)
        normalized = F.normalize(features.index_select(1, order), dim=-1, eps=1e-6)
        feature_gap = ((1 - (normalized[:, 1:] * normalized[:, :-1]).sum(-1))
                       .clamp(0, 2) / 2)
        return F.pad(0.5 * (measured + feature_gap), (1, 0))


class RegionalRefinement(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        neighbors, valid = neighborhood_indices(patch_size)
        self.register_buffer("neighbors", neighbors)
        self.register_buffer("valid", valid)
        self.norm = nn.LayerNorm(dim)
        self.interior = nn.Linear(dim, dim, bias=False)
        self.boundary = nn.Linear(2 * dim, dim, bias=False)
        self.raw_interior_gain = nn.Parameter(torch.tensor(-1.3862944))
        self.raw_boundary_gain = nn.Parameter(torch.tensor(-1.3862944))

    def forward(self, tokens, cue):
        local = self.norm(tokens)
        scores = -4 * cue["neighbor_gaps"]
        valid = self.valid[None, :, :].expand_as(scores)
        # patch_size=1: self-neighbors are a finite, neutral fallback.
        valid = valid | (~valid.any(-1, keepdim=True))
        weights = scores.masked_fill(~valid, -1e9).softmax(-1)
        mean = (local[:, self.neighbors] * weights[..., None]).sum(2)
        interior_update = self.interior(mean - local)
        boundary_update = self.boundary(torch.cat((local, local - mean), dim=-1))
        edge = cue["boundary_score"][..., None]
        importance = cue["importance"][..., None] / 1.5
        reliability = cue["reliability"][:, None, None]
        update = reliability * importance * (
            (1 - edge) * 0.5 * self.raw_interior_gain.sigmoid() * interior_update +
            edge * 0.5 * self.raw_boundary_gain.sigmoid() * boundary_update)
        # Keep the original token as a residual; no hard boundary suppression.
        return tokens + update, update


class StructureMambaV2(EncoderComparison):
    def __init__(self, wavelengths, class_means, class_variances, variant="full",
                 initialization_seed=1, reconstruction=False, **options):
        if variant not in VARIANTS:
            raise ValueError(f"未知版本 {variant}，可用版本：{list(VARIANTS)}")
        super().__init__(wavelengths, class_means, class_variances, encoder="operator",
                         initialization_seed=initialization_seed,
                         reconstruction=reconstruction, **options)
        self.variant = variant
        self.switches = dict(VARIANTS[variant])
        orders, inverse = scan_permutations(self.patch_size, self.patch_size)
        self.register_buffer("scan_orders", orders)
        self.register_buffer("scan_inverse", inverse)
        self.structure_cue = None
        self.region_refinement = None
        self.register_parameter("raw_forget_gain", None)
        # Extra modules must not shift the common backbone/dropout RNG stream.
        # They initialize on CPU, so only seed the CPU default generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed + 1103)
            if self.switches["structure"] or self.switches["boundary"]:
                self.structure_cue = StructureCue(self.patch_size)
            if self.switches["structure"]:
                self.raw_forget_gain = nn.Parameter(torch.tensor(-1.0986123))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed + 1301)
            if self.switches["boundary"]:
                self.region_refinement = RegionalRefinement(self.dim, self.patch_size)

    @staticmethod
    def _extra(name):
        return name.startswith(("structure_cue.", "region_refinement.", "raw_forget_gain",
                                "scan_orders", "scan_inverse"))

    def backbone_state(self):
        return {k: v for k, v in super().backbone_state().items() if not self._extra(k)}

    def component_counts(self):
        counts = {"encoder": 0, "backbone_and_classifier": 0, "common_decoder": 0,
                  "structure_guidance": 0, "regional_refinement": 0, "total": 0}
        for name, parameter in self.named_parameters():
            if name.startswith("operator."):
                key = "encoder"
            elif name.startswith("decoder."):
                key = "common_decoder"
            elif name.startswith("region_refinement."):
                key = "regional_refinement"
            elif self._extra(name):
                key = "structure_guidance"
            else:
                key = "backbone_and_classifier"
            counts[key] += parameter.numel()
            counts["total"] += parameter.numel()
        return counts

    def scan_order_names(self):
        return [SCAN_NAMES[i % 4] if self.switches["scan"] else "row_major"
                for i in range(len(self.layers))]

    def forward(self, patch, observed):
        if (patch.ndim != 4 or patch.shape[1] != self.channels or
                patch.shape[-2:] != (self.patch_size, self.patch_size)):
            raise ValueError("patch 必须匹配 [B,C,patch_size,patch_size]。")
        if observed.dtype != torch.bool or observed.shape != (len(patch), self.channels):
            raise ValueError("observed 必须是形状 [B,C] 的布尔掩码。")
        if torch.any(observed.sum(-1) == 0):
            raise ValueError("每个样本至少需要一个已观测波段。")
        spectra = patch.flatten(2).transpose(1, 2)
        clean = torch.where(observed[:, None, :], spectra, torch.zeros_like(spectra))
        encoded = self.operator(clean, observed)
        features = encoded["features"]
        reliability = encoded["relative_geometry_support"]
        tokens = features + self.position_encoder(self.positions)[None, :, :]
        cue = (self.structure_cue(clean, features, observed, reliability)
               if self.structure_cue is not None else None)
        forgetting = []
        for index, block in enumerate(self.layers):
            order = (self.scan_orders[index % 4] if self.switches["scan"] else
                     torch.arange(tokens.shape[1], device=tokens.device))
            inverse = (self.scan_inverse[index % 4] if self.switches["scan"] else order)
            sequence = tokens.index_select(1, order)
            strength = None
            if self.switches["structure"]:
                gap = self.structure_cue.transition_gap(clean, features, observed, order)
                importance = cue["importance"].index_select(1, order) / 1.5
                strength = 3 * self.raw_forget_gain.sigmoid() * reliability[:, None] * importance * gap
                forgetting.append(strength.index_select(1, inverse))
            sequence = block(sequence, torch.ones_like(sequence[:, :, 0]), strength)
            # Restore TRUE pixel coordinates before the next layer/readout.
            tokens = sequence.index_select(1, inverse)
        update = torch.zeros_like(tokens)
        if self.region_refinement is not None:
            tokens, update = self.region_refinement(tokens, cue)
        center = tokens[:, tokens.shape[1] // 2]
        pooled = torch.einsum("s,bsd->bd", self.pool_logits.softmax(0), tokens)
        logits = self.classifier(self.readout(torch.cat((center, pooled), dim=-1)))
        boundary = cue["boundary_score"] if cue else features.new_zeros(features.shape[:2])
        importance = cue["importance"] if cue else features.new_ones(features.shape[:2])
        forget_map = (torch.stack(forgetting).mean(0) if forgetting else
                      features.new_zeros(features.shape[:2]))
        result = {"logits": logits, "features": features, "input_spectra": spectra,
                  "observed": observed, "boundary_score": boundary, "importance": importance,
                  "forget_strength": forget_map, "regional_update": update,
                  "relative_geometry_support": reliability,
                  "diagnostics": {
                      "boundary_score_mean": boundary.detach().mean(),
                      "importance_mean": importance.detach().mean(),
                      "forget_strength_mean": forget_map.detach().mean(),
                      "regional_update_norm_mean": update.detach().norm(dim=-1).mean(),
                      "geometry_support_mean": reliability.detach().mean()}}
        if self.decoder is not None:
            result["reconstruction"] = self.decoder(features)
        return result
