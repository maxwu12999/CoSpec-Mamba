# Recorded main and window results

The original result exports and compact run identities are included alongside
the separately published internal spectral-encoding ablation.

| Directory | Training runs | Per-case CSV records | Scope |
|---|---:|---:|---|
| `results/full_lr001_v1/` | 210 | 3360 | Seven-dataset main comparisons, encoding/augmentation controls, scans/structure, and six HongHu 9×9 jobs |
| `results/window_lr001_v1/` | 12 additional | 192 | HongHu B/D at 7×7 and 11×11, three seeds |
| `results/internal_v1/` | 30 independent | 480 | Five spectral-encoding configurations on HongHu and PaviaU |

The window archive also contains 12 **reused** main-batch results at 9×9 and
15×15. Their recipe, model, initialization and complete metric records match
the main archive exactly. They are identified in `run_identities.json` and
are not counted as additional training jobs or duplicated in `metrics_raw.csv`.
The main and window batches together contain 222 unique completed 100-epoch
training runs and 3552 evaluated conditions.

## Archive exports and provenance

The original top-level CSV files are copied byte for byte. Each result directory
also has:

- `PROVENANCE.json`: source archive SHA-256, CSV checksums, counts and validation scope.
- `run_identities.json`: epoch count, job, recipe/model/source identities and recorded runtime details.
- `case_identities.json`: the shared case names and mask hashes for each dataset.
- `protocol.json`: recorded scientific settings, job list and data identities, without runtime data paths.

Every reported OA was checked against its original confusion counts. CSV OA,
AA and kappa values were checked against the per-run result JSON; per-seed
averages and test-mask, model and recipe identities were also verified.
The two MiM_v2 numerical-boundary recoveries (LongKou seed 2 and Salinas seed 3)
remain explicit in the run identities. See `REPRODUCIBILITY.md` for their scope.

## Rebuild manuscript tables without training

From the repository root:

```bash
python tools/summarize_paper_results.py
python tools/summarize_internal_results.py
```

The first command writes `results/paper_tables/`:

| File | Manuscript v18 table |
|---|---|
| `table3_window.csv` | 3: HongHu spatial windows |
| `table4_encoding_augmentation.csv` | 4: encoding and training augmentation |
| `table5_paired_d_minus_b.csv` | 5: paired D−B differences |
| `table7_scan.csv` | 7: scan and structure controls |
| `table8a_full.csv` to `table8d_keep40.csv` | 8a–8d: seven-dataset external comparisons |
| `table8e_low_retention.csv` | 8e: HongHu/PaviaU low-retention comparison |
| `table12_bd_overall.csv` | 12: seven-dataset overall B/D comparison |
| `external_trends.csv` | Numeric means for the six-condition external-comparison trends |

All 388 numeric cells in these ten manuscript tables matched the v18 document.
`manuscript_table_checks.json` records the checks; its DOCX table indices are
zero based. `validation.json` records aggregation and distinct-run counts.

The generator first averages masks within each training seed, then computes
the mean and sample standard deviation across three seeds. D−B differences
pair matching seeds before aggregation. Across-dataset summaries weight the
seven datasets equally. Ranking and formatted mean differences are descriptive
and do not establish statistical significance.

The additional 30-run internal batch remains separate. Its B/D results do not
replace the original B/D values in the main, window or scan tables. The second
command rebuilds its own manuscript Table 6 under `results/internal_v1/summary/`.
