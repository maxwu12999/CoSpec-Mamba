"""Word v16 section 3.2.2: B / LS / R / R+Delta / D.

Every model is constructed through the archived B or D constructor first.
Replacing an encoder never draws new random numbers or changes the backbone.
"""
import bootstrap
import torch
from torch import nn
from suite_models import build_model as archived_build

VARIANTS = ('B', 'LS', 'R', 'RDelta', 'D')
LABELS = {'B': 'B 掩码编码', 'LS': 'LS 无岭正则', 'R': 'R 仅岭拟合',
          'RDelta': 'R+Δ 无约束修正', 'D': 'D 完整算子'}


class InternalOperator(nn.Module):
    """Keep the exact original basis/projection; alter only the stated component."""
    def __init__(self, original, variant, svd_rtol):
        super().__init__()
        if variant not in ('LS', 'R', 'RDelta') or not 0 < svd_rtol < 1:
            raise ValueError('Invalid component ablation / SVD tolerance')
        self.variant, self.svd_rtol = variant, svd_rtol
        self.ridge = 0.0 if variant == 'LS' else original.ridge
        for name, value in original.named_buffers(recurse=False):
            self.register_buffer(name, value)
        # LS and R really remove these parameters; no dormant trainable refiner.
        if variant == 'RDelta':
            self.refiner = original.refiner
        self.projection = original.projection

    def coefficients(self, spectra, observed):
        if spectra.ndim != 3 or observed.shape != (spectra.shape[0], spectra.shape[2]):
            raise ValueError('Expected spectra [batch,pixels,bands] and mask [batch,bands]')
        if observed.dtype != torch.bool or torch.any(observed.sum(-1) == 0):
            raise ValueError('Every sample requires at least one observed band; mask must be bool')
        if spectra.shape[-1] != self.phi.shape[0]:
            raise ValueError('Band count mismatch')
        clean = torch.where(observed[:, None, :], spectra.float(), 0.)
        phi, mask = self.phi.float(), observed.float()
        if self.variant == 'LS':
            # Zero rows embed Phi_Omega in a rectangular fixed-size batch.
            # They leave its nonzero singular values and minimum-norm solution
            # unchanged. Use explicit rtol, NOT a band-count-dependent default.
            # A full-view batch and each test batch share one mask. Factor
            # each distinct mask once, rather than repeating 128 identical SVDs.
            unique, inverse = torch.unique(observed, dim=0, return_inverse=True)
            design = phi[None] * unique.float()[:, :, None]
            u, s, vh = torch.linalg.svd(design, full_matrices=False)
            kept = s > self.svd_rtol * s[:, :1]
            inv = torch.where(kept, s, torch.ones_like(s)).reciprocal() * kept
            transfer_unique = (vh.transpose(-1, -2) * inv[:, None, :]) @ u.transpose(-1, -2)
            transfer = transfer_unique.index_select(0, inverse)
        else:
            fisher = torch.einsum('ck,bc,cl->bkl', phi, mask, phi)
            chol = torch.linalg.cholesky(fisher + self.ridge * self.identity)
            transfer = torch.cholesky_solve(phi.T.unsqueeze(0) * mask[:, None, :], chol)
        return torch.einsum('bkc,bsc->bsk', transfer, clean)

    def forward(self, spectra, observed):
        with torch.autocast(device_type=spectra.device.type, enabled=False):
            coeff = self.coefficients(spectra, observed)
            refined = coeff
            if self.variant == 'RDelta':
                refined = coeff + 0.1 * torch.tanh(self.refiner(coeff))
            return {'features': self.projection(refined), 'coefficients': refined,
                    'initial_coefficients': coeff}


def build(variant, wave, stats, seed, *, backend='cuda', patch=15,
          dim=64, depth=4, basis=64, svd_rtol=1e-6):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}; use {VARIANTS}')
    model = archived_build('B' if variant == 'B' else 'D', wave, stats, seed,
                          dict(patch=patch, dim=dim, depth=depth, basis=basis, backend=backend))
    if variant in ('LS', 'R', 'RDelta'):
        model.operator = InternalOperator(model.operator, variant, svd_rtol)
    assert model.structure_cue is None and model.region_refinement is None
    return model

