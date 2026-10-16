# Application evidence

A claim-and-evidence map for using this project in job applications. It routes each
claim to the report, test or commit that supports it; it is not itself evidence. Follow
the links before using a fact.

**Verification levels used below**

- **Local:** run and checked on the author's Windows machine.
- **Image:** run and checked in the Linux Docker image built from a clean commit.
- **CI:** run by GitHub Actions. *No claim in this file has CI verification yet:* the
  workflow is configured but has not run on GitHub.

## Claims

| # | Claim | Evidence | Verified |
|---:|---|---|---|
| 1 | Built a reproducible batch forecasting pipeline: a clean checkout regenerates the data audit and backtest report byte-identically | [reports/data_audit.md](../reports/data_audit.md), [reports/backtest.md](../reports/backtest.md), [reports/provenance.json](../reports/provenance.json) (commit `02b63d7`, `git_dirty: false`) | Local, Image |
| 2 | Enforced data contracts on raw input: hash pin, schema checks per sheet, an exact-match rule for the overlapping sheets, and audited exclusions | `tests/test_ingest.py`, `tests/test_audit_and_cache.py`; [failures.md](failures.md) #1 | Local |
| 3 | Found and fixed a data-preparation defect: lower-case and whitespace variants of stock codes that split one product into several series (5,305 distinct codes before normalisation, 5,131 after) | [reports/data_audit.md](../reports/data_audit.md) (stock code normalisation), commit `3746cba` | Local |
| 4 | Evaluated with leakage-safe rolling-origin backtesting: training labels need `t + h <= T`, recipes only see data up to each cutoff, and two planted leaks are detected by tests | `tests/test_leakage.py`, `tests/test_features.py`, `tests/test_backtest.py`; [failures.md](failures.md) #2 | Local |
| 5 | Compared a LightGBM model against naive and moving-average baselines with WAPE, bias and MASE, per fold and per horizon | [reports/backtest.md](../reports/backtest.md) | Local, Image |
| 6 | Pre-registered the promotion criteria: the gate config was committed before the first full-data evaluation, and full-data runs refuse a gate that is uncommitted or edited | commits `2bc7216` (freeze) then `d00b5d1` (first evaluation); `tests/test_gate.py`; [failures.md](failures.md) #4 | Local, Image |
| 7 | The gate rejected a LightGBM model that lowered pooled backtest WAPE by 1.9% but missed the 5% bar, won 3 of 6 folds and was worse at two horizons; a baseline stayed champion | [reports/backtest.md](../reports/backtest.md) (Gate section); [failures.md](failures.md) #3 | Local, Image |
| 8 | Built a model registry with a champion alias, lineage to the gate decision, and checksummed model files | `src/forecastgate/registry.py`, `tests/test_registry.py` | Local |
| 9 | Published predictions atomically and idempotently: staged validation, a lock, a pointer swap, and refusal of same-identity reruns with different content | `tests/test_publish.py`; [failures.md](failures.md) #5 | Local |
| 10 | Served forecasts read-only through FastAPI, refusing stale, future-dated and tampered artifacts with a 503 and the reason | `tests/test_serving.py`; [failures.md](failures.md) #6 | Local |
| 11 | Separated training, batch inference and serving into commands that share only checksummed files | [architecture.md](architecture.md); `forecastgate gate`, `predict`, `serve` | Local |
| 12 | Containerised every command; the image verifies the gate freeze itself and carries no git credentials in any layer | `Dockerfile`, `scripts/scan_image_layers.py`; [failures.md](failures.md) #7 | Image |
| 13 | Test suite of 133 tests; each core guard was also checked by planting a bug and confirming a test fails | `tests/`; the planted-bug checks are described in the commit history, not in a committed script | Local |

## Numbers safe to quote

All from [reports/backtest.md](../reports/backtest.md) and
[reports/data_audit.md](../reports/data_audit.md):

- Pooled WAPE: `naive` 0.7930, `ma4` 0.6212, `lgbm` 0.6096 (1.9% lower than `ma4`).
- `lgbm` won 3 of 6 folds; pooled bias +0.025; MASE 0.952 against `ma4`'s 0.944.
- 1,067,371 rows across the two sheets (1,044,848 after the overlap rule); 1,003,214
  sales rows after exclusions; 104 complete weeks; 200 products per fold.

## Unsafe framings

Do not claim any of these; the evidence does not support them.

- That LightGBM improved forecasts, or that the project improved forecast accuracy. It
  failed the promotion criteria.
- Where LightGBM's gains or losses occur (for example, by product volume). No
  per-product analysis was done.
- Demand forecasting accuracy in business terms, inventory or revenue impact, or live
  users. The target is recorded sales on historical data.
- Production scale, scheduled retraining, monitoring or drift detection. None exist.
- MLflow. The registry is a JSON registry by design.
- Green CI or CI-verified results. CI has not run.
- Hyperparameter tuning. Parameters were fixed and not tuned.

## Suggested résumé bullets

Each bullet uses only claims above; numbers come from the committed reports.

- Built a reproducible weekly sales forecasting pipeline (Python, pandas, Pandera,
  LightGBM, FastAPI, Docker) with data contracts, leakage-safe rolling-origin
  backtesting against baselines, and byte-identical report regeneration.
- Pre-registered model promotion criteria before the first evaluation; the gate
  rejected a LightGBM model that lowered pooled WAPE by 1.9% but missed the 5% threshold
  and won only 3 of 6 folds, keeping a moving-average baseline as champion.
- Implemented a model registry, atomic versioned batch publication and read-only serving
  that refuses stale or tampered prediction artifacts, with each failure path
  demonstrated and tested.
