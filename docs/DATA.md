# Data contract

The full paper queue consumes an existing prepared cache, not a new random
split. Raw scenes, labels and trained weights are not redistributed here.
Obtain source datasets under their providers' terms and preserve the specified
band order and original TR/TE split. Different splits are separate experiments.

The archived runner requires this layout under its `--reference` directory:

```text
COSPEC_SUITE_V1/
  cache/
    HongHu/
      cube.npy
      tr.npy
      te.npy
      wave.npy
      stats.npz
      data.json
    PaviaU/
    HanChuan/
    LongKou/
    Houston/
    IndianPines/
    Salinas/
```

| File | Meaning |
|---|---|
| cube.npy | normalized H×W×C scene; memory-mapped on demand |
| tr.npy / te.npy | H×W labels, 0 background and 1..K classes; disjoint center pixels |
| wave.npy | length-C finite strictly increasing spectral coordinates |
| stats.npz | training-only class spectral `means` and `variances` |
| data.json | data identity, normalization, split and file hashes, original evaluation masks |

The exact reference identities and coordinate vectors are retained in
`src/cospec_full_lr001_v1/REFERENCE_DATA.json`; validators use `data_id`, shapes,
class supports and SHA-256 of cache files. Historical acquisition paths inside
that manifest are provenance strings, not paths opened by the runner. Actual
paths come from `--reference`. Do not edit identity hashes to force acceptance.

| Dataset | Training centers | Test centers | Bands | Classes | Coordinates |
|---|---:|---:|---:|---:|---|
| HongHu | 2200 | 384493 | 270 | 22 | measured wavelengths, nm |
| PaviaU | 900 | 41876 | 103 | 9 | relative band coordinates |
| HanChuan | 1600 | 255930 | 274 | 16 | measured wavelengths, nm |
| LongKou | 900 | 203642 | 270 | 9 | measured wavelengths, nm |
| Houston | 1500 | 13529 | 144 | 15 | relative band coordinates |
| IndianPines | 1293 | 8956 | 200 | 16 | relative band coordinates |
| Salinas | 1600 | 52529 | 204 | 16 | relative band coordinates |

Relative indices are not measured wavelengths. Existing physical coordinates
must not be replaced by fabricated evenly spaced wavelengths when claiming
to reproduce the paper.

For a new dataset, use the public `cospec.build_model` API. The archived
`frozen/suite/omamba/data.py::load_split` reads a MAT file containing `input`
(H×W×C), `TR` and `TE`, and returns normalized data, labels, class statistics and
a split report. Its normalization is fitted on training center spectra only.
It is a reusable loader, not a guarantee that a newly assembled dataset is the
original hash-verified paper cache. The exact archived full-queue reproduction
requires that cache; this release does not include an automatic raw-data to
paper-split downloader.

Test masks use a fixed NumPy permutation stream (seed 20261014), with three
repetitions. Within each repetition, lower retention masks are prefixes of the
same band ordering. Every model and training seed sees the same test masks.
Training uses a separate mask generator and excludes the test masks. Deleting
bands never changes the number of training centers.


## Original split artifacts and cache preparation

**Remaining requirement:** the exact original `tr.npy`/`te.npy` files and the
associated coordinate/statistic artifacts are not included in this repository.
Metadata hashes alone cannot recover their pixel locations. The included
result archive also contains no split arrays. The export/restore utility below
does not remove this requirement and never substitutes a new random split.

On the machine that has the original `COSPEC_SUITE_V1` cache, export its exact
small artifacts without the normalized scene cube:

```bash
python tools/reference_artifacts.py export \
  --reference /path/to/COSPEC_SUITE_V1 \
  --output /path/to/paper_reference_artifacts
```

Export validates all five cache hashes against the published reference
identities before writing. It copies `tr.npy`, `te.npy`, `wave.npy`, `stats.npz`
and `data.json` for each dataset, plus `ARTIFACTS.json`. The exported original
artifacts must subsequently be distributed with the code or a linked archive.
Do not replace them with the synthetic fixtures used in tests.

Once those artifacts and the original raw public scene are available, restore
a cache one dataset at a time:

```bash
python tools/reference_artifacts.py restore \
  --artifacts /path/to/paper_reference_artifacts \
  --dataset PaviaU --raw /path/to/PaviaU.mat --mat-key paviaU \
  --output /path/to/COSPEC_SUITE_V1
```

An H×W×C raw `.npy` is also supported, without `--mat-key`. MAT loading uses
SciPy's non-HDF5 MAT reader. Preserve the exact original spectral band order
and band filtering. Restoration uses the recorded training-only global mean
and scale and the original class statistics; it does not refit using the scene
or test labels. All recreated cache files must match the published SHA-256
values before the destination becomes usable. A different raw-data variant or
numerical result is rejected rather than accepted as an exact paper replay.
Existing export/cache directories are never overwritten.
