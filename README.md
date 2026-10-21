# CoSpec-Mamba

Research code for hyperspectral image classification with partially observed
spectral bands. The main model combines a coordinate-based spectral operator
with four-direction snake scanning and a Mamba backbone.

This release contains the **constant learning rate 0.001** experiment code used
for the current manuscript, including the full comparison queue and the
7×7 / 11×11 window rerun. The manuscript is in preparation; this repository
does not claim an accepted publication.

## Quick start: use the model

Python 3.10–3.12 is recommended. A CPU environment is sufficient for the example
and public API tests. Formal training uses Linux/WSL and NVIDIA CUDA.

```bash
git clone https://github.com/maxwu12999/CoSpec-Mamba.git
cd CoSpec-Mamba
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```python
import numpy as np
import torch
from cospec import build_model

# Example coordinates and statistics only. For real data, use increasing
# measured wavelengths when available and training-only class statistics.
wave = np.linspace(400, 1000, 32, dtype=np.float32)
means = np.zeros((3, 32), dtype=np.float32)
variances = np.ones_like(means)
model = build_model(wave, means, variances, variant="D", patch_size=15)

x = torch.randn(2, 32, 15, 15)       # batch, channels, height, width
observed = torch.ones(2, 32, dtype=torch.bool)
observed[:, 16:] = False             # False means an unobserved band
output = model(x, observed)
print(output["logits"].shape)          # torch.Size([2, 3])
```

The example uses the slow CPU reference recurrence for verification. For CUDA,
install a compatible `mamba-ssm` and `causal-conv1d` build, construct with
`backend="cuda"`, and move both model and inputs to CUDA. The recorded server
environment used PyTorch 2.5.1 + CUDA 12.4. CUDA extensions must match the
installed PyTorch, CUDA, Python and C++ ABI; GPU families are not interchangeable
without checking that build. Dependencies are not upgraded by experiment launchers.

## Models and experiments

| Identifier | Spectral encoding | Missing-band training augmentation |
|---|---|---|
| A | mask-aware linear | off |
| B | mask-aware linear | on |
| C | spectral operator | off |
| D | spectral operator | on |
| D_structure | spectral operator + structure control | on |
| D_row / D_raster4 / D_random4 | alternative scan orders | on |
| ConvMamba_aug | archived ConvMamba adapter | on |

**D is the main model.** B is retained as an internal encoding control. Structure
control is an ablation rather than part of the main D configuration. External
adapters cover HybridSN, A²S²K, SQSFormer, GSCViT and MiM_v2. Their results are
under this repository's common training protocol, rather than each upstream
paper's separately tuned recipe.

| Queue | Training tasks | Scope |
|---|---:|---|
| Full fixed-LR queue | 210 | internal 99 + external 105 + six HongHu 9×9 tasks |
| Window rerun | 12 | HongHu B/D × 7×7/11×11 × seeds 1/2/3 |

The window queue reads completed 9×9 and 15×15 results without retraining them.
Each task uses 100 epochs and 16 evaluation conditions: full spectrum, then
80%, 60%, 40%, 20% and 10% band retention with three shared nested masks.
These percentages refer to **observed spectral bands**, not training samples.
20% and 10% are evaluation-only conditions and do not add training jobs.

## Restore external model sources

Third-party sources are fetched from pinned upstream commits and checked against
the exact experiment hashes. They are kept outside version control. The public
CoSpec model API above does not require these downloads; complete archived
queues and their frozen-source checks do.

```bash
python tools/fetch_upstreams.py
python tools/fetch_upstreams.py --check
```

See [THIRD_PARTY.md](THIRD_PARTY.md) for source repositories and license scope.
The downloader only retrieves source bytes; it does not run upstream programs.

## Reproduce the archived experiment queues

The archived runners expect the original prepared data caches and fixed TR/TE
split, which are **not distributed in this code repository**. They deliberately
reject a different split or modified cache. The required files, coordinate
conventions and checksums are documented in
[docs/DATA.md](docs/DATA.md). Model weights, raw scenes, personal run logs and
the manuscript are not included.

Activate your CUDA environment, then run from this repository's root:

```bash
# Full queue, with the runtime data-loading acceleration used in the experiments.
python src/cospec_speedup_v1/launcher.py \
  --reference /path/to/COSPEC_SUITE_V1 \
  --output "$PWD/src/diagnostics/COSPEC_FULL_LR001_V1" \
  --max-parallel 10

# A viewer can be reopened without restarting workers.
python src/cospec_full_lr001_v1/control.py watch

# Set a lower concurrency ceiling; running jobs finish normally.
python src/cospec_full_lr001_v1/control.py slots --max-parallel 5

# Request a safe stop and save. Rerun the launch command to resume.
python src/cospec_full_lr001_v1/control.py stop
```

Concurrency is admitted using measured GPU memory and RAM headroom, up to ten
workers. The ceiling does not force ten simultaneous jobs. A task is completed
only after training, all evaluation conditions and its exports finish. The
dashboard shows total/completed tasks, current workers and a throughput estimate.

After the full queue is complete, run the window queue:

```bash
python src/cospec_window_lr001_v1/control.py start \
  --reference /path/to/COSPEC_SUITE_V1 \
  --parent-output "$PWD/src/diagnostics/COSPEC_FULL_LR001_V1" \
  --max-parallel 2
```

For a WSL laptop whose display consumes some GPU memory, the archived local
launcher changes only the initial idle-memory gate:

```bash
python src/cospec_window_local.py --reference /path/to/COSPEC_SUITE_V1 \
  --parent-output "$PWD/src/diagnostics/COSPEC_FULL_LR001_V1"
```

The two archived MiM failures had a zero-width Gaussian mask (`sigma == 0`).
Their audited numerical recovery script and its exact boundary behavior are
included; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Failed jobs are
not counted as successful, and no automatic learning-rate reduction is applied.

## Layout

```text
cospec/                         Public model import
src/cospec_full_lr001_v1/        Frozen scientific core and full queue
src/cospec_speedup_v1/           Archived runtime acceleration
src/cospec_window_lr001_v1/      Fixed-LR 7×7 and 11×11 rerun
src/cospec_mim_recover.py        Audited MiM numerical-boundary recovery
src/cospec_window_local.py       WSL/laptop preflight adaptation
tools/                          Pinned upstream source restoration
tests/                          Public API CPU checks
docs/                           Data contract and provenance
```

The per-package READMEs are historical experiment instructions. Use this root
README for the public repository layout. Scientific source files are preserved
byte for byte so that archived hashes and checkpoint validation remain usable.

## Evaluation scope

All comparisons use fixed within-scene training/test center splits. Neighboring
input patches can overlap; these experiments are not spatially disjoint or
cross-sensor evaluation. Performance depends on dataset and retention level.
The internal B control is included to expose encoding-specific tradeoffs.
Synthetic checks validate implementation behavior and are not classification
accuracy results.

## License

Original code is released under the [MIT License](LICENSE). Third-party files
and installed dependencies retain their original license terms.

