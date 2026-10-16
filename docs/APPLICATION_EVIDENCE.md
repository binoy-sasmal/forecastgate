# Application Evidence: forecastgate

A claim-and-evidence map for using this project in job applications. It routes each
claim to the report, test or commit that supports it; it is not itself evidence. Follow
the links before using a fact.

## Status

Complete. The authoritative status is the "What is verified" table and the limitations in
the [README](../README.md). The frozen thresholds in [gate.yaml](../gate.yaml) and the
rejected-model result are final for this project.

## What it demonstrates

- Leakage-safe rolling-origin backtesting of a learned model against naive and
  moving-average baselines.
- Promotion criteria pre-registered in git before the first evaluation, enforced in code,
  and honoured when they rejected the candidate.
- Data contracts and a generated data audit on a real, messy public dataset.
- A model registry, atomic versioned batch publication and read-only serving, each with
  its failure paths demonstrated.
- Reproducibility: byte-identical reports on Windows, in a Linux image and in CI.

**Verification levels used below**

- **Local:** run and checked on the author's Windows machine.
- **Image:** run and checked in the Linux Docker image built from a clean commit.
- **CI:** run by GitHub Actions at commit `9b4aa84`: the [push run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38970499923) (Ruff,
  mypy, 133 tests, fixture `reproduce` and `backtest` on the runner and in the image) and
  the manually triggered [full-data run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38971080356) (full `reproduce` in the image, freeze
  verified, `git diff --exit-code` on both committed reports).

## Evidence map

| # | Claim | Evidence | Verified |
|---:|---|---|---|
| 1 | Built a reproducible batch forecasting pipeline: a clean checkout regenerates the data audit and backtest report byte-identically | [reports/data_audit.md](../reports/data_audit.md), [reports/backtest.md](../reports/backtest.md), [reports/provenance.json](../reports/provenance.json) (commit `02b63d7`, `git_dirty: false`); [CI full-data run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38971080356) | Local, Image, CI |
| 2 | Enforced data contracts on raw input: hash pin, schema checks per sheet, an exact-match rule for the overlapping sheets, and audited exclusions | `tests/test_ingest.py`, `tests/test_audit_and_cache.py`; [failures.md](failures.md) #1 | Local, CI |
| 3 | Found and fixed a data-preparation defect: lower-case and whitespace variants of stock codes that split one product into several series (5,305 distinct codes before normalisation, 5,131 after) | [reports/data_audit.md](../reports/data_audit.md) (stock code normalisation), commit `3746cba`; regenerated identically in the [CI full-data run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38971080356) | Local, CI |
| 4 | Evaluated with leakage-safe rolling-origin backtesting: training labels need `t + h <= T`, recipes only see data up to each cutoff, and two planted leaks are detected by tests | `tests/test_leakage.py`, `tests/test_features.py`, `tests/test_backtest.py`; [failures.md](failures.md) #2 | Local, CI |
| 5 | Compared a LightGBM model against naive and moving-average baselines with WAPE, bias and MASE, per fold and per horizon | [reports/backtest.md](../reports/backtest.md); regenerated identically in the [CI full-data run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38971080356) | Local, Image, CI |
| 6 | Pre-registered the promotion criteria: the gate config was committed before the first full-data evaluation, and full-data runs refuse a gate that is uncommitted or edited | commits `2bc7216` (freeze) then `d00b5d1` (first evaluation); `tests/test_gate.py`; [failures.md](failures.md) #4; freeze verified in the [CI full-data run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38971080356) | Local, Image, CI |
| 7 | The gate rejected a LightGBM model that lowered pooled backtest WAPE by 1.9% but missed the 5% bar, won 3 of 6 folds and was worse at two horizons; a baseline stayed champion | [reports/backtest.md](../reports/backtest.md) (Gate section); [failures.md](failures.md) #3; executed in the [CI full-data run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38971080356) | Local, Image, CI |
| 8 | Built a model registry with a champion alias, lineage to the gate decision, and checksummed model files | `src/forecastgate/registry.py`, `tests/test_registry.py` | Local, CI |
| 9 | Published predictions atomically and idempotently: staged validation, a lock, a pointer swap, and refusal of same-identity reruns with different content | `tests/test_publish.py`; [failures.md](failures.md) #5 | Local, CI |
| 10 | Served forecasts read-only through FastAPI, refusing stale, future-dated and tampered artifacts with a 503 and the reason | `tests/test_serving.py`; [failures.md](failures.md) #6 | Local, CI |
| 11 | Separated training, batch inference and serving into commands that share only checksummed files | [architecture.md](architecture.md); `forecastgate gate`, `predict`, `serve` | Local; `gate` and `predict` also run inside `reproduce` in CI |
| 12 | Containerised every command; the image verifies the gate freeze itself and carries no git credentials in any layer | `Dockerfile`, `scripts/scan_image_layers.py`; [failures.md](failures.md) #7 | Image (the credential scan is not part of CI; the image build and runs are) |
| 13 | Test suite of 133 tests; each core guard was also checked by planting a bug and confirming a test fails | `tests/`, 133 passed in the [CI push run](https://github.com/binoy-sasmal/forecastgate/actions/runs/38970499923); the planted-bug checks are described in the commit history, not in a committed script | Local, CI (tests); local only (planted bugs) |

## Numbers safe to quote

All from [reports/backtest.md](../reports/backtest.md) and
[reports/data_audit.md](../reports/data_audit.md):

- Pooled WAPE: `naive` 0.7930, `ma4` 0.6212, `lgbm` 0.6096 (1.9% lower than `ma4`).
- `lgbm` won 3 of 6 folds; pooled bias +0.025; MASE 0.952 against `ma4`'s 0.944.
- 1,067,371 rows across the two sheets (1,044,848 after the overlap rule); 1,003,214
  sales rows after exclusions; 104 complete weeks; 200 products per fold.

## Unsupported claims

Do not claim any of these; the evidence does not support them.

- That LightGBM improved forecasts, or that the project improved forecast accuracy. It
  failed the promotion criteria.
- Where LightGBM's gains or losses occur (for example, by product volume). No
  per-product analysis was done.
- Demand forecasting accuracy in business terms, inventory or revenue impact, or live
  users. The target is recorded sales on historical data.
- Production scale, scheduled retraining, monitoring or drift detection. None exist.
- MLflow. The registry is a JSON registry by design.
- CI verification beyond what the two linked runs at `9b4aa84` show (for example, the
  credential scan, which runs locally only).
- Hyperparameter tuning. Parameters were fixed and not tuned.

## Safe application framing

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

## Verification rule

Before using a number, open the report it comes from at the commit you are citing and
confirm it there. Before claiming CI verification, confirm the linked run's commit SHA and
that it passed. Record the repository commit used in the application analysis rather than
relying on any commit named in this file as current.
