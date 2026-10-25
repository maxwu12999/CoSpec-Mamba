# Reproducibility and provenance

## Scientific settings

The full 210-task run and the subsequent 12-task window rerun use Adam,
learning rate **0.001 for all 100 epochs**, weight decay 0, training batch size
32, evaluation batch size 128 and gradient norm clipping at 15. There is no
epoch-73 learning-rate change in this release's training worker.

The standard CoSpec backbone uses dimension 64, four layers and 64 spectral
basis functions. D uses a spectral operator, missing-band augmentation and
four-direction snake scanning. B uses mask-aware linear encoding with the same
augmentation. Augmented runs average full-view and missing-view cross-entropy;
A/C are the corresponding no-missing-augmentation controls.

Evaluation includes the complete spectrum and 80/60/40/20/10% retention with
three shared nested masks (16 conditions). Aggregate statistics first average
the three masks within each training seed, then summarize the three training
seeds. Test pixels are the full supplied TE centers.

## Archived source identities

The scientific core directories are copied without editing their Python files
or identity manifests. Third-party bytes are restored separately, because they
retain upstream ownership and license conditions.

| Archive | Scientific source SHA-256 |
|---|---|
| full fixed-LR core | `40a0f0ccaa744c3d0c748caacda4a53f0718d6c3a4a0429a26d70ff9e1c2f841` |
| window rerun core | `08720cb5be7a19b6a315f6761e73537107d46012d4906b7b42c8e9f5805866ef` |

The archived full-run protocol ID is
`3812432c7c2e0e1263921712ca2606756c1fe826a2e0a2c7d59271fd815bf9b4`.
`FROZEN_SHA256.json` records individual model/adaptor source hashes.
`ORIGINAL_PLANS.json` records the original 99 and 105 task lists.
`UPSTREAM_SCIENCE.json` records the window queue's parent scientific settings.

The runtime speedup in `src/cospec_speedup_v1` is archived revision
`9bc02f04314ea1a725b12a6ab1582c343e470106e9bc4dc459dbe5cc39af06a5`.
It batches patch reads and reuses test inputs across conditions. It does not
change optimizer, training batch size, precision, masks or training samples.
Its identity checks reject a modified scientific core.

## MiM zero-sigma numerical recovery

Two archived MiM_v2 tasks failed on nonfinite loss: LongKou seed 2 and Salinas
seed 3. Diagnostic replay localized the LongKou failure to
`T_Mamba.gaussian_mask_vector`: equal token vectors produce zero average
distance, so the original formula divides by zero. The saved model and Adam
state before failure were finite.

`src/cospec_mim_recover.py` preserves the positive-sigma formula and gradients
exactly and defines **uniform weights for sigma == 0**, with a safe denominator
selected before division. It does not sanitize arbitrary nonfinite inputs,
skip samples or reduce the learning rate. It patches the dynamically loaded
class in memory and records the extension and counters separately in results.
It only resumes the two identified original jobs under the expected protocol.

```bash
python src/cospec_mim_recover.py check
# Only for an existing matching archived run with these failed jobs:
python src/cospec_mim_recover.py start --workers 2
```

The recovery is an explicit numerical-boundary extension. It is not represented
as an unmodified upstream MiM result or a general remedy for any training NaN.

## Verification

```bash
# No downloaded external model sources needed:
python -m unittest discover -s tests -v

# Source restoration required for archived checks:
python tools/fetch_upstreams.py --check
python src/cospec_window_lr001_v1/self_check.py --device cpu
python src/cospec_mim_recover.py check
```

The public API checks test predictions when values in unobserved bands change,
finite gradients and invalid coordinate rejection. Archived window checks cover
task/mask identities, data-loading equivalence, checkpoint continuation and
test cursor continuation using synthetic fixtures. They are not formal OA
measurements or guarantees of bitwise replay across different CUDA hardware.

Raw scenes, checkpoints, manuscript files, private runtime logs and credentials
are excluded from version control. `RELEASE_MANIFEST.json` provides SHA-256
for the committed release files, excluding itself.


## Added independent internal-ablation batch

`src/cospec_internal_ablation_v1/` contains the unchanged original scientific
Python files used for the 30-run internal experiment. Its own archived-source
checks and package manifest are preserved. Current manuscript numbering is
Section 3.2.3 and Table 6; the earlier Chinese package documentation records
its original v16 numbering.

`results/internal_v1/PROVENANCE.json` identifies the completed result archive.
The export contains 480 per-mask metric records and 30 completed-run identities.
OA values were checked against the original confusion counts before export;
all recorded test-mask hashes matched the experiment protocol. Results include
all six retention fractions, while the manuscript-table exporter selects four.

```bash
python tools/summarize_internal_results.py
python src/cospec_internal_ablation_v1/self_check.py
python -m unittest discover -s tests -v
```

The aggregation command checks run/condition completeness, shared masks and
within-seed aggregation. The internal CPU checks cover SVD and ridge equations,
constrained versus unconstrained correction, initialization matching and
train/test continuation. These implementation checks do not rerun the formal
100-epoch CUDA experiments. Original split artifacts for all seven datasets are
now included in `data/reference/paper_reference_artifacts.zip`, with checksums
and provenance. Their hashes match the archived paper reference identities;
see `DATA.md` for reconstruction of the image caches from the original scenes.

## Recorded main and window result release

The original per-case, per-seed and aggregate CSV files from the completed
main and window archives are now included. Compact protocol, case and run
identities retain source/model/mask checks and numerical-recovery metadata.
`tools/summarize_paper_results.py` rebuilds ten v18 manuscript tables, all 388
numeric cells of which were checked against the document. The 12 reused
window parent snapshots are verified against the main archive and are not
counted again. See `RESULTS.md` for provenance and table mappings.
