# forecastgate: design

Status of this document: design, written before implementation. Every capability below is
**planned** until the README status table marks it implemented and names its test.
Revision 2 resolves a design review (training-label boundary, initial champion, gate
arithmetic, leakage tests, target wording, publication protocol, metric edge cases).
Revision 3 resolves a second review (a valid feature mislabelled as a leak, reproduction
under gate rejection, gate scope of undefined metrics, universe, zero-transaction weeks,
publisher lock, freshness bounds, report determinism).
Revision 4 records step 2 findings: stock code normalisation and the input hash in the
artifact identity; counts now live only in the generated data audit.

## 1. Problem

Forecast weekly recorded sales quantity per product for horizons 1 to 4 weeks ahead,
produced at a fixed cutoff and published as a versioned batch artifact.

- **Unit of prediction:** (StockCode, cutoff week T, horizon h), target week T + h.
- **Target:** weekly recorded sales quantity after the documented exclusions in section 2.
  The data records invoice lines and invoice timestamps, not shipments, so the target is
  not called demand or shipped quantity.
- **Zero weeks:** a week with no recorded sales for an in-universe product is recorded as 0
  observed sales. This is not evidence of zero underlying demand: stockouts and
  unavailability are unobserved. The limitation is stated in the README.
- **Week:** ISO week, Monday 00:00 to Sunday 23:59. A week is **complete** only when the
  data's coverage interval spans the whole week. This drops the first week (data starts
  Tuesday 2009-12-01 07:45) and the last (data ends Friday 2011-12-09 12:50).
- **Closure weeks:** a week with zero recorded transactions store-wide is a closure week
  (observed: weeks ending 2010-01-03 and 2011-01-02). Closure weeks are not targets, are
  not evaluated, and enter lag features as missing, not as 0.
- **T** always means the last fully observed (complete) week at a cutoff.

### Product universe (per cutoff)

The universe is recomputed at every cutoff T from data up to T only, by a fixed rule:

1. Eligible: first recorded sale at least 26 weeks before T.
2. Rank eligible products by recorded quantity in the 26 weeks ending at T; ties broken by
   StockCode.
3. Keep the top 200.

Weeks before a product's first recorded sale produce no rows (not zeros). From the first
sale to T, missing weeks are 0 observed sales, except closure weeks. The rule and N are
frozen in config with the gate (section 5) before the first full-data evaluation.

- The universe at a fold's cutoff is frozen for all four horizons and is identical for
  every recipe. A product is scored even if it has no later transactions (0 observed).
- If fewer than 200 products qualify, the universe is all qualifying products and the
  report records the count. Fewer than 50 fails the panel contract.
- Results therefore describe established, higher-volume products, and the README says so.

### Zero-transaction weeks

"Closure week" is an **assumption**: zero transactions store-wide shows a gap in the data,
not a confirmed closure. Calendar spacing is kept, so lag k always means k calendar weeks
back, and a zero-transaction week is a missing observation.

- `naive` uses the most recent non-missing week `<= t`.
- `ma4` averages the non-missing weeks among the last 4 calendar weeks and falls back to
  `naive` when all four are missing.
- Training labels skip zero-transaction target weeks, a decision made from data `<= T`.
- Evaluation never drops scored rows silently. If a backtest target week is a
  zero-transaction week, the backtest stops with an error instead of excluding it (the
  planned folds, June to November 2011, contain none).

## 2. Data

UCI Machine Learning Repository, *Online Retail II*, CC BY 4.0.
Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D

| Fact | Value |
|---|---|
| Download | `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`, no login |
| Zip SHA-256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| Content | one file `online_retail_II.xlsx`, sheets `Year 2009-2010`, `Year 2010-2011` |
| Columns in the file | `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country` |

The UCI web page lists `InvoiceNo`, `UnitPrice` and `CustomerID`; the file is the authority.

**Evidence status.** All counts now come from the generated
[reports/data_audit.md](../reports/data_audit.md), which is the only source for them. The
exploratory counts in revisions 1 to 3 were superseded; some differed because the
exclusions now apply in a fixed order and each row is counted once.

### Preparation rules

Applied in this order. Each exclusion counts a row under the first rule that removes it.

| Issue | Rule |
|---|---|
| Sheet overlap | Sheet 1 supplies rows strictly before sheet 2's first timestamp; sheet 2 supplies everything from it. The rows both sheets cover must be an exact multiset match, or the file is rejected. The check runs on the original rows |
| Stock code case and whitespace | Codes are stripped and upper-cased. The audit found lower-case codes (e.g. `15056p`) selling alongside upper-case twins, usually with the same description, and a code with a trailing space; left alone, each product's history would be split across two series |
| Cancellations (`C` invoice prefix) | Excluded |
| Quantity <= 0 | Excluded |
| Price <= 0 | Excluded |
| Non-product codes | A product code is five digits with an optional letter suffix; everything else is excluded and listed in the audit with row counts |
| Identical rows | Dropped. **Assumption:** they may be legitimate repeated invoice lines; the audit reports count and quantity share |
| Missing Customer ID | Allowed, not used |
| `StockCode` and `Description` mix types | Both read as string at ingest |

CI never downloads. A small real slice (permitted by CC BY 4.0, with attribution) is
committed as the fixture, plus tiny synthetic fixtures with hand-computed answers.

## 3. Pipeline

```
ingest  -> hash check -> raw contract -> overlap rule, exclusions -> weekly panel -> panel contract -> data audit
train   -> per-cutoff universe and features -> rolling-origin backtest of recipes -> report -> gate -> final refit -> registry
predict -> load champion -> features at T -> prediction contract -> staged publish
serve   -> read-only API over the committed artifact, with integrity and freshness checks
```

Train, predict and serve are separate entrypoints sharing only `features/`, the recipe
interface and artifacts on disk.

**Recipe** = a forecasting method plus its config: `naive`, `ma4` (4-week moving average)
or `lgbm`. Every recipe implements the same `fit` / `predict` interface on a panel
truncated at its cutoff (`Panel.truncated(T)`), so no recipe can reach a week after T, and
produces artifacts with the same metadata, so baselines are first-class models.
`lgbm` uses LightGBM's native training API with fixed parameters, single-threaded and
deterministic; the backtest report is byte-identical on Windows and in the Linux image.

## 4. Evaluation

### Training-label boundary

At cutoff T, a training example (product, origin t, horizon h) is eligible only when
`t + h <= T`, the target week is not a closure week, and the product has a first sale
before t. Its features use observations from weeks `<= t` only. Filtering on `t <= T`
alone would leak labels from after T; a test asserts the condition directly.

### Model

One global LightGBM recipe, direct multi-horizon (h is a feature), Tweedie objective so
predictions are non-negative by construction for count-like data. A shared postprocess
step clips to `>= 0` and is used identically in backtest and inference.

Features at origin t (implemented in `features.py`; every window ends at t and counts
calendar weeks, so week t itself is known and week t + 1 is not):

- `lag_0` to `lag_7`: y[t] to y[t - 7]. A zero-transaction week is missing, not 0.
- `mean_4`, `std_4`, `mean_8`, `std_8`: over observed weeks in the window; a mean needs one
  observed week, a population standard deviation two.
- `zero_share_8`: share of observed weeks in the last 8 with 0 recorded sales.
- `price_median_8`: median over the last 8 weeks of each week's median unit price; weeks
  without sales have no price.
- `weeks_since_first_sale`, `horizon`, and `target_week_of_year` (calendar of week t + h,
  known in advance).
Hyperparameters are fixed and documented. No tuning in v1: tuning inside the backtest
would need nested folds.

### Backtest

Six cutoffs 4 weeks apart. The last cutoff is the last complete week minus 4, so every
fold has all four horizons observed. The committed fixture spans October 2010 to the end
of the file, so it runs under exactly the full-run settings and the same fold dates. Each recipe is refitted per fold using only that
fold's eligible examples. Candidate, baselines and champion are always evaluated as
recipes on the **same folds**, never as a model fitted on later data.

### Metrics

With `e = |y - yhat|` over a set S of (product, cutoff, horizon) rows:

- **WAPE(S)** = `sum(e) / sum(y)`. Undefined when `sum(y) = 0`.
- **Bias(S)** = `sum(yhat - y) / sum(y)`. Undefined when `sum(y) = 0`.
- **MASE** per series: mean absolute error divided by the series' in-sample mean absolute
  one-step change over its history up to the fold cutoff (closure weeks skipped). Series
  with a zero denominator are excluded and counted. Reported, not gated.

Reported per fold, per horizon (pooled across folds), and pooled over everything, for the
candidate, both baselines, the champion, and the **per-fold oracle** (the better baseline
chosen separately per fold). The oracle is a demanding reference, not a deployable model,
and is never the gate bar.

### Leakage tests

1. **Truncation equivalence.** At each evaluation cutoff T **and** at sampled individual
   training origins t (`tests/test_leakage.py`, on the real fixture), build universe and features from raw data truncated at that point,
   and from full raw data whose later records are altered (quantities, prices,
   descriptions, products added and removed). The two must be identical. Week t itself is
   available at origin t, so features may include it.
2. **Label eligibility.** Every training example satisfies `t + h <= T`.
3. **Detectors catch planted leaks.** `--inject-leak future_window` (a rolling feature whose
   window includes `y[t+1]`) and `--inject-leak origin_filter` (eligibility on `t <= T`).
   The tests must detect each one. The report records any metric change a leak produces,
   but detection, not metric improvement, is the pass condition.

## 5. Promotion gate

`gate.yaml` (thresholds, universe rule, folds, recipe configs) is committed **before the
first full-data evaluation**; the report records that commit.

Enforcement (`gate.py`): a full-data backtest first verifies, before any download, that
`gate.yaml` is tracked by git and identical to its committed version; the freeze commit is
the last commit that changed it, not HEAD. Every run with a `gate.yaml` then checks that the
effective universe rule, folds, horizons and every recipe's parameters equal the file. The
backtest report and `provenance.json` record the gate's SHA-256 and the freeze commit.
The criteria below are evaluated in code from the backtest summary; that code is committed
in the freeze commit too, so the decision rule cannot change after the results are seen
without the change appearing in history. Planted leaks are refused outside the fixture.

- **Best single baseline:** whichever of `naive` and `ma4` has the lower pooled WAPE on the
  backtest. Both are parameter-free, and choosing the stronger one only raises the bar.
- **Initial champion:** on a fresh installation the champion is the best single baseline
  recipe, refitted and registered like any model. Prediction and serving therefore work
  even if no ML model is ever promoted.

The candidate is promoted only if all hold (written as inequalities, so zero errors need
no division):

1. `WAPE_pooled(cand) <= 0.95 * WAPE_pooled(base)`
2. `WAPE_fold(cand) < WAPE_fold(base)` in at least 4 of 6 folds
3. `WAPE_fold(cand) <= 1.10 * WAPE_fold(base)` in every fold
4. `WAPE_h(cand) <= WAPE_h(base)` for every horizon h
5. `|Bias_pooled(cand)| <= 0.05`
6. Against the champion recipe on the same folds:
   `WAPE_pooled(cand) <= 0.99 * WAPE_pooled(champion)` (at least 1% improvement; otherwise
   the champion stays, reason "no material improvement"). On first run the champion is the
   best baseline, so this is implied by criterion 1.

Any undefined metric **used by a gate criterion** fails the gate (fail closed). Report-only
diagnostics such as MASE never block promotion. Pass: final refit of the candidate recipe
at the publication cutoff on all eligible examples, registered and aliased `champion`.
A failing LightGBM is reported, not tuned until it passes.

**Exit codes.** `gate` exits 0 on promotion, 10 on a quality rejection (structured reasons
written, alias unchanged), and any other non-zero code on an unexpected error.
`forecastgate reproduce` is a Python orchestrator, not a chain of shell targets: it records
exit 10 as an expected outcome and continues to publish with the current champion (the
baseline on a fresh installation). Any other failure in training, validation or
publication stops the run.

Registry (implemented in `registry.py`): the planned fallback, a file-based JSON registry,
chosen over MLflow. Two of the three recipes are parameter-free baselines, and what the
registry must do here (versions, a `champion` alias, lineage to the gate decision and
checksummed model files) needs no tracking server or model wrappers. MLflow was not
added. Each version records recipe, parameters, data and input hashes, gate SHA-256 and
freeze commit; loading verifies every model file against its registered checksum.

Commands and exit codes:

- `forecastgate gate` (training): backtest, decide, register. On an empty registry the
  best single baseline is refitted at the publication cutoff and registered as the
  initial champion. A promoted candidate is refitted and becomes champion. A champion
  fitted on different input is refitted under its own recipe. Exits 0 on promotion and
  10 on a quality rejection; every run appends to `decisions.jsonl`.
- `forecastgate predict` (batch inference): loads the champion, refuses one fitted on
  different input, and publishes.
- `forecastgate reproduce`: data audit, gate, predict. A rejection is recorded and
  publication continues with the champion; any other failure stops the run.

## 6. Artifacts, publication and serving

**Identity** = hash of (cutoff, model version, recipe config hash, source archive hash,
input hash). The input hash covers the exact model input (prepared panel, universe and
horizons), so a change in data preparation yields a new artifact instead of a refusal, and
a refusal still means non-determinism, tampering, or a recipe whose behaviour changed
without a config change.
Predictions are a CSV with a fixed byte format so the content hash is stable: UTF-8 without
BOM, LF newlines, header row, fixed column order (`stock_code, cutoff_week, horizon,
target_week, prediction`), ISO `YYYY-MM-DD` dates, `%.6f` floats, rows sorted by
(`stock_code`, `horizon`).

Publication protocol (one publisher at a time):

0. Acquire `predictions/.lock` by exclusive create. If it exists, refuse with a message
   naming the holder; a stale lock is removed by hand, as documented. Every later step
   runs under the lock.
1. Remove leftover `predictions/.staging/*` from interrupted runs.
2. If `predictions/<identity>/` exists: same content hash means no-op; different content
   means refuse with non-zero exit (non-determinism or tampering).
3. Otherwise write `predictions.csv` and `manifest.json` (identity fields, git SHA, row
   count, file checksums) into `predictions/.staging/<identity>-<nonce>/`, run the
   prediction contract and a manifest self-check there, then rename the directory to
   `predictions/<identity>/` on the same filesystem.
4. Write `CURRENT.tmp` naming the identity, then `os.replace` it to `CURRENT`.

A crash before step 3's rename leaves only staging, which serving never reads. A crash
between steps 3 and 4 is repaired by rerunning: step 2 finds identical content and step 4
runs. A new model or config version has a new identity and is a normal new artifact.

**Prediction contract:** exactly universe(T) x {1, 2, 3, 4} rows, unique keys, target week
equals T + h, values finite and `>= 0`, cutoff equal to the manifest's.

**Serving** (implemented: `serving.py`, `api.py`; `forecastgate serve` and
`forecastgate serve-check`) is read-only and re-validates on every request: it resolves
`CURRENT`, requires a committed artifact whose manifest names it, verifies the
predictions file against the manifest checksum, re-checks the content (rows, keys,
cutoff, target weeks, finite non-negative values), and checks freshness against a
simulated `as_of` date. An artifact for cutoff week T covers data through T + 6 days (the
Sunday); it is served only if `0 <= as_of - (T + 6 days) <= max_age_days` (default 7), so
stale and future-dated artifacts are both refused. A refusal is a 503 with the reason
(`serve-check` exits 11); serving never falls back to an older artifact. `/health`
reports the identity, model version and age served.

## 7. Failure demonstrations

| Demo | Mechanism | Evidence |
|---|---|---|
| Invalid input rejected | Hash pin, raw and panel contracts, overlap check | Corrupted fixtures: non-zero exit, nothing written |
| Leakage detected | Truncation and eligibility tests | Planted leaks are detected; any metric change they cause is recorded |
| Weak model blocked | Gate on a crippled recipe | Exit 10, alias unchanged, reproduction still publishes the champion |
| Failed validation blocks publication | Prediction contract in staging | Staging discarded, `CURRENT` unchanged |
| Interrupted publication | Crash injected between steps 3 and 4 | Rerun completes; serving never saw a partial artifact |
| Stale or tampered artifact refused | Freshness and checksum checks | 503 with the reason |
| Safe rerun | Identity and content hash | Same identity: no-op; altered content: refused; new version: published |

Each demo is a pytest test and a section in `docs/failures.md` with real command output.

## 8. Stack

Python 3.12, uv, pandas, Pandera, LightGBM, MLflow (local), FastAPI (optional), Typer,
pytest, Ruff, mypy, one Docker image, GitHub Actions.

Excluded on purpose: Airflow, DVC, Compose, cloud services, Terraform, dashboards, a second
model family, drift monitoring (planned only). Compute is a laptop: about 1M rows trains in
minutes, and reproduction must not need cluster access.

## 9. Build order

The walking skeleton comes first, so integration problems surface on day 1, not day 2.

Day 1
1. Skeleton: uv project, Ruff, mypy, pytest, committed fixture slice, `uv run forecastgate reproduce --fixture`
   running ingest, `ma4` fit, predict and publish offline on the fixture; minimal CI;
   Dockerfile. CI and Docker use the same commands.
   Verify: CI green, `docker run` completes the fixture run.
2. Ingest: hash pin, raw contract, overlap rule, exclusions, weekly panel, closure weeks,
   `reports/data_audit.md`. Verify: corrupted fixtures rejected; audit generated.
3. Universe and features with leakage tests 1 to 3. Verify: tests pass; planted leaks fail.
4. Backtest harness, metrics with edge cases, baselines and LightGBM, **fixture only**.
   Verify: fold metrics identical across two runs; hand-computed WAPE matches.
5. Commit `gate.yaml` and the universe config. This is the freeze point.

Day 2
6. First full-data backtest and committed report.
7. Gate, initial baseline champion, final refit, registry. Verify: crippled recipe blocked.
8. Publication protocol and prediction contract. Verify: every row of section 7.
9. Serving (optional), docs: `architecture.md`, `failures.md`, limitations, README status
   table, `docs/APPLICATION_EVIDENCE.md`.

Cut order: FastAPI (fall back to a `serve-check` CLI), MLflow (JSON registry), MASE. The
leakage tests, gate and failure tests are never cut.

## 10. Acceptance criteria

Full-data runs inside the Docker image: the image carries the repository's `.git` and
the full working tree, so the freeze is verified there by the same check as in a
checkout and provenance records the real commit and `git_dirty`; nothing is bypassed,
and editing `gate.yaml` inside a container is refused. The manual CI job
`full-data-docker` runs the full `reproduce` in the image and requires the regenerated
deterministic reports to equal the committed ones (configured; it has not run on GitHub).

Tradeoff, accepted for a local portfolio project: shipping `.git` makes the image larger
and exposes the repository history inside it, in exchange for a freeze check that is the
same code path as in a checkout rather than a separate, weaker manifest. The build context
excludes downloaded data, caches, local registries, artifacts and credential files. The
context's `.git` is read only by a discarded build stage, which passes on just `HEAD`,
objects, refs and the index with a freshly written config; the original config (remote
URLs, credential helpers, extra headers), reflogs and hooks reach no layer of the final
image. Deleting them in a later layer would not have been enough, since earlier layers
keep them. CI checkouts use `persist-credentials: false`. A production image would
instead ship a signed freeze attestation and no history.

- Public GitHub repository with green CI on `main`.
- From a clean checkout: `uv sync --locked`, then `uv run forecastgate reproduce` downloads with the hash
  check, writes the data audit, backtests, gates, refits and publishes, completing even
  when the gate rejects the candidate (section 5).
- Reports are split: `reports/data_audit.md` and `reports/backtest.md` hold only
  deterministic content (metrics, counts, data hash, `gate.yaml` content hash), and
  `reports/provenance.json` holds run provenance (git SHA, timestamps, versions). Inside the
  Docker image, regenerated deterministic reports match the committed ones byte for byte;
  across platforms, differences are documented.
- `uv run pytest` passes offline, and `uv run forecastgate reproduce --fixture` runs
  offline end to end.
- The CLI is the only official interface; there is no Makefile, so Windows, Linux and the
  Docker image use identical commands.
- Every README claim links to a test, report or artifact; capabilities are labelled
  implemented and tested, simulated, or planned.

## 11. Limitations known in advance

Two years of history, one retailer, one prior Christmas season for the final folds,
recorded sales rather than demand (stockouts unobserved), intermittent series that may
favour baselines, the identical-row assumption, a single run per cutoff, historical data
with a simulated clock, no live users.
