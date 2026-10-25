# Original paper reference artifacts

`paper_reference_artifacts.zip` contains the original files used for all seven
public datasets: HongHu, HanChuan, PaviaU, LongKou, Houston, IndianPines and
Salinas. Each dataset has `tr.npy`, `te.npy`, `wave.npy`, `stats.npz` and
`data.json`. The original 35 files are preserved byte for byte. The archive
also includes `ARTIFACTS.json`, matching the external checksum list here.

All split-array, coordinate and statistic hashes match the archived
`src/cospec_full_lr001_v1/REFERENCE_DATA.json`. Training/test center counts,
nonoverlap of center labels, coordinate validity and statistic shapes were
verified. `PROVENANCE.json` records hashes of the supplied and published ZIPs.

Extract from the repository root:

```bash
python -m zipfile -e data/reference/paper_reference_artifacts.zip /path/to/paper_reference_artifacts
```

Then follow `docs/DATA.md` to rebuild each hash-verified cache from the original
public raw scene. The artifact archive contains no image cube or checkpoints.
The original raw data version, band filtering and order must match the paper.
Matching data identities does not guarantee identical floating-point training
results across different hardware or software environments.
