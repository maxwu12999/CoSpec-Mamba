"""Training-prototype proxy for retained class differences.

Uses diagonal within-class variance and class means from TR only. It is a
small-sample approximation: neither calibrated uncertainty nor proof that
classes are unidentifiable. It intentionally has no classifier-loss target.
"""
import torch
from torch import nn


class AmbiguityEstimator(nn.Module):
    def __init__(self, means, variances, temperature=1.0):
        super().__init__()
        means = torch.as_tensor(means, dtype=torch.float32)
        variances = torch.as_tensor(variances, dtype=torch.float32).clamp_min(1e-4)
        if means.shape != variances.shape or means.ndim != 2 or len(means) < 2:
            raise ValueError("至少两类，means/variances 必须为 [classes, bands]。")
        if temperature <= 0:
            raise ValueError("temperature 必须正数。")
        self.register_buffer("means", means)
        self.register_buffer("variances", variances)
        pairs = torch.triu_indices(len(means), len(means), offset=1)
        self.register_buffer("pair_i", pairs[0])
        self.register_buffer("pair_j", pairs[1])
        signal = ((means[pairs[0]] - means[pairs[1]]) ** 2 /
                  (variances[pairs[0]] + variances[pairs[1]] + 1e-4))
        self.register_buffer("pair_signal", signal)
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, spectra, observed):
        mask = observed.float()
        x = spectra.detach().float()
        # Accumulate one class at a time to avoid [B,S,classes,bands] memory.
        distance = x.new_zeros(x.shape[0], x.shape[1], len(self.means))
        for c in range(len(self.means)):
            difference = (x - self.means[c]) * mask[:, None, :]
            d = difference ** 2 / self.variances[c]
            distance[:, :, c] = d.sum(-1) / mask.sum(-1).clamp_min(1)[:, None]
        probabilities = torch.softmax(-distance / self.temperature, dim=-1)
        # Avoid a near one-hot prototype prediction making every pair weight zero.
        q = probabilities.clamp_min(1e-4)
        q = q / q.sum(-1, keepdim=True)
        pair_weights = q[:, :, self.pair_i] * q[:, :, self.pair_j]
        total_signal = self.pair_signal.sum(-1)
        retained = torch.einsum("pc,bc->bp", self.pair_signal, mask)
        ratio = retained / total_signal.clamp_min(1e-8)[None, :]
        ratio = torch.where(total_signal[None, :] > 1e-8, ratio, torch.zeros_like(ratio))
        ratio[observed.all(-1)] = 1
        information = ((pair_weights * ratio[:, None, :]).sum(-1) /
                       pair_weights.sum(-1).clamp_min(1e-8)).clamp(0, 1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        return {"retained_information": information,
                "prototype_entropy": entropy, "prototype_probabilities": probabilities}
