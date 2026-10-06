# Study source: scope and reproduction

This folder preserves the authored study's package and script topology. It is
source for inspection and synthetic verification, not a self-contained refit of
the licensed financial study. No market observations, security-level results,
credentials, pretrained models, native binaries or original authors' code are
included. Do not run the historical writers against accepted study directories.

## Start here

From the release root, install Python 3.12 and the source dependencies, then run:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r research/requirements-study.txt
.venv/bin/python research/run_portable_tests.py --output /new/path/synthetic-checks
```

The command copies the source to temporary storage before testing. It checks a
specified subset with synthetic arrays, SQLite checkpoints, economic operator
identities, gradients, date cutoffs, E2/E3/E4 behavior and the industry projection.
No licensed data, R, Apple native compiler or historical fitting is invoked.
Linux/macOS are the intended public test platforms; Windows is not validated.
The full retained test collection is supplied for inspection but includes tests
requiring archives, R or macOS. Do not interpret the subset as a full refit.

Aggregate paper-number replication is separate: use the top-level replication/
package and its README. It reuses saved aggregate returns and does not refit.

## Current and historical entry points

| Scope | Source |
|---|---|
| Baseline accounting, finite dynamic operators, risk and reciprocal wealth | baseline/ |
| Earlier E1 confidence/step-size study, excluded from final model | confidence/, learner/, comparisons/, inference/; scripts/step7*, step8*, step9*, step10* |
| E2 characteristic scales and historical independent controls | feature_geometry/, geometry_study/ |
| E3 temporal feature gains and original optimizer/selection | temporal_extension/, temporal_study/ |
| Retained equivalent E3 engineering generations | temporal_fast/, temporal_fast_v2/, temporal_fast_v3/, temporal_fast_v4/, temporal_storage/, temporal_numerics/, temporal_continuity/ |
| E4 state construction, conditional moments and coefficients | extension4/, extension4_execution/ |
| Final yearly E2/E3/E4 fitting, staged selection, IND holdings and inference | annual_ind/, annual_staged/; scripts/annual_ind_staged_20261001.py |
| IND target projection and historical component screens | scripts/industry_projection_kernels_20260928.py and the named screen scripts |
| WRDS acquisition, data transformations and R parity adapters | scripts/step3* through scripts/step6*; baseline/reference_helpers.R |

The final study is the staged annual implementation. Older dated scripts are
retained scientific provenance, not competing current launch instructions.
Executed configuration retains its original overlapping 2020–2025 development
window; the final paper reports only 2021–2025 as the extension. Do not silently
change historical configs or their expected hashes to conceal this distinction.

## Full financial reconstruction prerequisites

A fresh checkout cannot directly execute the historical final runner. It needs
WRDS access to JKP characteristics and CRSP monthly/daily/distribution data,
source vintages and characteristic metadata, prepared monthly/risk matrices,
state histories, fitted parent banks and the seven prior acceptance manifests.
The final runner validates historical code/config/checkpoint hashes and expects
those retained dependencies. Its fixed output paths are not a general CLI.

The sole bundled output/ file is factual feature/settings metadata at
output/step2_20260910/source_settings.json. It permits loading modules that read
settings on import. It contains no security observations. Some historical
modules create empty local data/output directories on import; the public test
runner confines this behavior to its temporary copy.

Acquisition code is provided for inspection. With independently obtained source
licenses, a separate research workspace and compatible metadata, its acquisition
entry is `python scripts/step3_acquire.py {sample,monthly,events,daily}`; these
stages require sequential preparation, and `daily` needs the prior ID inventory.
WRDS credentials belong in the user's own libpq configuration; never in the
repository. Data extraction is not exercised by public tests.

Native v3/v4 paths require macOS, Xcode/Command Line Tools (`xcrun`, Clang and the
matching macOS SDK), Apple Accelerate and libproc. The independent R verification
adapters require R plus data.table, dplyr, lubridate, purrr, expm, jsonlite, Rcpp
and any dependencies requested by the retained adapters. Original author R/C++
sources must be obtained separately from the pinned upstream release:
https://github.com/theisij/ml-and-the-implementable-efficient-frontier
(reference commit 13d836b5187cbfa69730dd256d8eb767e65ca21f).
They are not redistributed; no upstream redistribution license was verified.

## Export changes

Source_inventory.csv gives each original and exported hash and every change.
All approved scientific originals remain unchanged in the canonical study.
Local project paths in configuration become `<research-root>`. Conversation and
operational authorization fields are omitted from public config copies; these
copies do not satisfy old acceptance hashes. A hardcoded fallback list of case
identifiers in the data-report adapter is omitted; reconstruction requires its
original licensed case ledger. The scientific data transformations are retained.

The E4 import uses the independently written public_support/conditional_features.py
instead of distributing the unlicensed reference-package helper. It implements
exactly the documented design `[B, c1*B[:,J], ..., cK*B[:,J]]`, without another
normalization. A local one-time comparison checked valid and invalid synthetic
cases against the retained reference; public tests use an independent formula.
No selected model, coefficient, saved return or accepted source was altered.

The package contains authored code and factual metadata, with copyright retained
unless a separate release license states otherwise. This is not a claim that
all dependencies or licensed financial datasets can be redistributed.
