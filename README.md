# forecastgate

Weekly sales forecasting for an online retailer, built to show production-oriented ML
engineering: data contracts, leakage-safe backtesting against baselines, a promotion gate,
and versioned, atomically published batch predictions.

**Status: under construction.** The design is in [docs/design.md](docs/design.md). Only
what the table below marks as implemented exists.

## Run it

Requires [uv](https://docs.astral.sh/uv/). The CLI is the only interface; there is no
Makefile, so the commands are the same on Windows, Linux, CI and Docker.

```sh
uv sync --locked
uv run pytest                                # offline
uv run forecastgate reproduce --fixture      # offline, on the committed data slice
uv run forecastgate backtest --fixture       # offline backtest of every recipe
uv run forecastgate reproduce                # downloads the full dataset (hash-pinned)
```

Docker:

```sh
docker build -t forecastgate .
docker run --rm --network none forecastgate uv run forecastgate reproduce --fixture
```

## Status

| Capability | State | Evidence |
|---|---|---|
| Hash-pinned download, raw data contract, sheet-overlap rule, code normalisation, exclusions | Implemented, tested | `tests/test_ingest.py`; modified real archives rejected in `tests/test_audit_and_cache.py`; corrupted archive in `tests/test_reproduce_fixture.py` |
| Deterministic data audit, provenance kept separate | Implemented, tested | [reports/data_audit.md](reports/data_audit.md), `tests/test_audit_and_cache.py` |
| Parse cache keyed by archive hash, re-validated on read | Implemented, tested | `tests/test_audit_and_cache.py` |
| Weekly panel: complete weeks, zero-transaction weeks as missing | Implemented, tested | `tests/test_panel.py` |
| Product universe as of the cutoff | Implemented, tested | `tests/test_universe_and_recipes.py` |
| `naive` and `ma4` baseline recipes | Implemented, tested | `tests/test_universe_and_recipes.py` |
| Prediction contract, locked atomic publication, idempotent reruns, identity covering the model input | Implemented, tested | `tests/test_publish.py`, `tests/test_reproduce_fixture.py` |
| Features at a forecast origin, training rows with `t + h <= T`, inference rows | Implemented, tested | `tests/test_features.py` (hand-computed alignment, missing weeks) |
| Leakage tests: truncation equivalence at fold cutoffs and training origins, label eligibility, two planted leaks | Implemented, tested on the fixture | `tests/test_leakage.py` |
| Backtest cutoffs | Implemented, tested | `tests/test_features.py` |
| Rolling-origin backtest of `naive`, `ma4` and `lgbm` on truncated panels; WAPE, bias, MASE with edge cases; per-fold oracle | Implemented, tested on the fixture | `tests/test_backtest.py`; `uv run forecastgate backtest --fixture` |
| Full-data backtest refused until `gate.yaml` is committed | Implemented, tested | `tests/test_backtest.py` |
| Full-data backtest report | Planned (after the gate freeze) | |
| Promotion gate, registry, baseline champion | Planned | |
| Read-only serving with freshness checks | Planned | |

At this stage `reproduce` publishes the fixed `ma4` baseline; the gate decides the
champion once it exists.

## Data

UCI *Online Retail II* by D. Chen (2012), CC BY 4.0,
https://doi.org/10.24432/C5CG6D. The committed fixture is a slice of it; see
[data/fixture/README.md](data/fixture/README.md).

## Licence

Code: MIT. Data: CC BY 4.0, as above.
