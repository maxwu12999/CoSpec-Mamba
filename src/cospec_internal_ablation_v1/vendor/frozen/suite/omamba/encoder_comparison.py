"""Compare input encoders while preserving the original spatial Mamba backbone.

The linear baseline knows which measurements are missing. Encoder parameter
counts differ and are reported; a common architecture is not an equal-total-
parameter claim. Optional reconstruction uses the same learned decoder in
both models, never the spectral operator's native reconstruction.
"""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .model import ObservationMamba


class MaskAwareLinearEncoder(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        # Equivalent to one affine layer on [zero-filled values, binary mask],
        # without materializing a doubled spectral tensor for every pixel.
        self.values = nn.Linear(channels, dim)
        self.mask = nn.Linear(channels, dim, bias=False)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, spectra, observed):
        clean = torch.where(observed[:, None, :], spectra, torch.zeros_like(spectra))
        features = self.values(clean) + self.mask(observed.to(spectra.dtype))[:, None, :]
        return {"features": self.norm(self.activation(features))}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class EncoderComparison(ObservationMamba):
    def __init__(self, wavelengths, class_means, class_variances, encoder="operator",
                 initialization_seed=1, reconstruction=False, **options):
        if encoder not in ("linear", "operator"):
            raise ValueError("encoder 必须是 linear 或 operator。")
        # Construct the original architecture first in BOTH runs. Replacing
        # the encoder later must not shift the backbone's initialization RNG.
        super().__init__(wavelengths, class_means, class_variances, **options)
        self.encoder_kind = encoder
        self.channels = len(wavelengths)
        self.dim = self.position_encoder.out_features
        del self.ambiguity  # This experiment has no prototype/state control.
        if encoder == "linear":
            with torch.random.fork_rng(devices=[]):
                # These layers initialize on CPU. manual_seed() also seeds
                # CUDA, which a CPU-only fork would not restore.
                torch.random.default_generator.manual_seed(initialization_seed + 707)
                self.operator = MaskAwareLinearEncoder(self.channels, self.dim)
        self.decoder = None
        if reconstruction:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(initialization_seed + 919)
                self.decoder = nn.Linear(self.dim, self.channels)

    def backbone_state(self):
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith(("operator.", "decoder."))}

    def component_counts(self):
        encoder = sum(p.numel() for p in self.operator.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters()) if self.decoder else 0
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": encoder, "backbone_and_classifier": total - encoder - decoder,
                "common_decoder": decoder, "total": total}

    def forward(self, patch, observed):
        if (patch.ndim != 4 or patch.shape[1] != self.channels or
                patch.shape[-2:] != (self.patch_size, self.patch_size)):
            raise ValueError("patch 应为 [B,C,patch_size,patch_size]，且匹配模型。")
        if observed.dtype != torch.bool or observed.shape != (len(patch), self.channels):
            raise ValueError("observed 必须是形状 [B,C] 的布尔掩码。")
        if torch.any(observed.sum(-1) == 0):
            raise ValueError("每个样本必须至少包含一个已观测波段。")
        spectra = patch.flatten(2).transpose(1, 2)
        clean = torch.where(observed[:, None, :], spectra, torch.zeros_like(spectra))
        output = self.operator(clean, observed)
        features = output["features"]
        tokens = features + self.position_encoder(self.positions)[None, :, :]
        write_weight = tokens.new_ones(tokens.shape[:2])
        for block in self.layers:
            tokens = block(tokens, write_weight)
        center = tokens[:, tokens.shape[1] // 2]
        pooled = torch.einsum("s,bsd->bd", self.pool_logits.softmax(0), tokens)
        logits = self.classifier(self.readout(torch.cat((center, pooled), dim=-1)))
        result = {"logits": logits, "features": features, "input_spectra": spectra,
                  "observed": observed}
        if self.decoder is not None:
            result["reconstruction"] = self.decoder(features)
        # The operator's native reconstruction/support are deliberately not
        # treated as auxiliary targets or fabricated for the linear baseline.
        return result


def observed_reconstruction_mse(output):
    if "reconstruction" not in output:
        raise ValueError("重建损失需要两组共有的可学习解码器。")
    mask = output["observed"][:, None, :]
    difference = torch.where(mask, output["reconstruction"] - output["input_spectra"],
                             torch.zeros_like(output["reconstruction"]))
    denominator = mask.sum() * difference.shape[1]
    return difference.square().sum() / denominator.clamp_min(1)


def encoder_objective(full, partial, labels, reconstruction_weight=0.0):
    classification = (F.cross_entropy(full["logits"], labels) +
                      F.cross_entropy(partial["logits"], labels)) / 2
    reconstruction = classification.new_zeros(())
    if reconstruction_weight:
        reconstruction = (observed_reconstruction_mse(full) +
                          observed_reconstruction_mse(partial)) / 2
    loss = classification + reconstruction_weight * reconstruction
    return loss, {"loss": float(loss.detach()),
                  "classification_loss": float(classification.detach()),
                  "reconstruction_loss": float(reconstruction.detach()),
                  "observed_fraction": float(partial["observed"].float().mean())}
