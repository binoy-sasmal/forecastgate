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
uv run forecastgate gate --fixture           # training: backtest, gate, register (exit 10 = rejected)
uv run forecastgate predict --fixture        # batch inference: publish the champion
uv run forecastgate serve-check --fixture --as-of 2011-12-05   # what serving would serve
uv run forecastgate serve --fixture --as-of 2011-12-05         # read-only API on :8000
uv run forecastgate reproduce                # downloads the full dataset (hash-pinned)
```

Docker:

```sh
docker build -t forecastgate .
docker run --rm --network none forecastgate uv run forecastgate reproduce --fixture
docker run --rm forecastgate uv run forecastgate reproduce      # full data; downloads the archive
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
| Frozen `gate.yaml`: full-data runs require it committed and unchanged; effective config must match | Implemented, tested | `tests/test_gate.py` |
| Gate criteria evaluated in code, decision in the backtest report | Implemented, tested | `tests/test_gate.py` |
| JSON registry with `champion` alias and checksummed models; baseline as initial champion; refit at the publication cutoff | Implemented, tested | `tests/test_registry.py` |
| `gate` exits 10 on rejection with the alias unchanged; `reproduce` still publishes the champion | Implemented, tested | `tests/test_registry.py` |
| Full-data `reproduce` inside Docker, freeze verified from packaged git metadata; reports byte-identical to the committed ones | Implemented, verified locally in the image | manual CI job `full-data-docker` (configured, not yet verified) |
| GitHub Actions CI: lint, types, tests, fixture runs locally and in Docker | Configured, not yet verified: it has not run on GitHub | `.github/workflows/ci.yml` |
| Read-only serving: stale, future-dated, tampered or inconsistent artifacts refused with 503 | Implemented, tested | `tests/test_serving.py`; `forecastgate serve`, `forecastgate serve-check` |

## First full-data evaluation

Against the gate frozen at commit `2bc7216`, `lgbm` reduced pooled backtest WAPE by 1.9%
relative to the 4-week moving average (0.6096 vs 0.6212) but did not satisfy the
promotion criteria: below the 5% pooled bar, 3 of 6 folds won, and worse at horizons 1
and 3. `ma4` is the registered champion and `reproduce` publishes its forecasts. See
[reports/backtest.md](reports/backtest.md).

## Data

UCI *Online Retail II* by D. Chen (2012), CC BY 4.0,
https://doi.org/10.24432/C5CG6D. The committed fixture is a slice of it; see
[data/fixture/README.md](data/fixture/README.md).

## Licence

Code: MIT. Data: CC BY 4.0, as above.
