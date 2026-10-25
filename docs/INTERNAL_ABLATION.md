# Internal spectral-encoding ablation

This addition corresponds to **manuscript v18 Section 3.2.3, Table 6**. The
archived Chinese package README refers to the earlier v16 numbering. Its
scientific Python files are preserved byte for byte, including their original
source-identity and checkpoint checks.

## Configurations and training

| CLI ID | Encoding | Learnable correction | Observation constraint |
|---|---|---|---|
| B | Mask-aware linear encoding | none | none |
| LS | 64 fixed basis functions, minimum-norm SVD | none | none |
| R | Fixed-basis ridge fit, alpha 0.01 | none | none |
| RDelta | Same ridge fit | bounded 0.1 tanh correction | none |
| D | Complete original operator | same bounded correction | observation resolution matrix |

LS uses float32 SVD with relative threshold 1e-6 and absolute threshold zero.
RDelta denotes R+Δ in the manuscript. B and D are **independently retrained** in
this batch, rather than copied from the older main-experiment tables.

The formal queue has 30 runs: HongHu/PaviaU × five configurations × three seeds.
All runs use 100 epochs, Adam with constant learning rate 0.001, 15×15 patches,
width 64, four snake-scan Mamba layers, and matched dual-view supervision.
Shared-backbone initialization and fixed-basis projection initialization are
checked by seed. No graph module or additional structure control is enabled.

From the repository root, in the existing compatible CUDA environment:

```bash
# Mathematical, data-identity, initialization and continuation checks on CPU.
python src/cospec_internal_ablation_v1/self_check.py

# Validate the original cache and initialization, without training.
bash src/cospec_internal_ablation_v1/start.sh \
  --reference /path/to/COSPEC_SUITE_V1 \
  --output "$PWD/src/diagnostics/COSPEC_INTERNAL_V1" --check-only

# Train or resume the 30 formal runs.
bash src/cospec_internal_ablation_v1/start.sh \
  --reference /path/to/COSPEC_SUITE_V1 \
  --output "$PWD/src/diagnostics/COSPEC_INTERNAL_V1"
```

The reference directory must contain the original hash-verified caches. See
[DATA.md](DATA.md) for the data contract and remaining split-artifact requirement.
Dependencies are not installed or upgraded by this runner. A CPU implementation
check is separate from the completed formal GPU experiments.

## Recorded results and rebuilding Table 6

`results/internal_v1/` contains the values from the completed result archive:

| File | Content |
|---|---|
| `per_mask_metrics.csv` | 480 original per-case metric records, including OA, AA, kappa and mask/model identities |
| `runs.json` | 30 completed-run identities, epoch counts and scientific source hashes |
| `protocol.json` | Recorded scientific settings and data/mask identities, with runtime cache paths omitted |
| `PROVENANCE.json` | SHA-256 of the supplied result archive and extraction checks |
| `summary/` | Rebuilt Table 6, per-seed metrics, paired differences and aggregation validation |

Rebuild the table without training, checkpoints or original imagery:

```bash
python tools/summarize_internal_results.py
```

Masks are averaged **within each training seed**, followed by the mean and sample
standard deviation (ddof=1) across three seeds. Component differences pair the
same seed before computing their mean and standard deviation. Three masks do
not create three additional independent training runs.

The manuscript table contains only full/80%/60%/40% results. The recorded runner
also evaluated 20% and 10%; those values remain in the per-mask and per-seed
records for provenance and are excluded from this manuscript table export.
Table 6's independent B/D runs must not replace the older main-experiment B/D
values. Higher means are descriptive and do not imply statistical significance.

The archived package `PROVENANCE.json` describes its original code-only delivery;
the completed results have their own provenance file under `results/internal_v1`.
