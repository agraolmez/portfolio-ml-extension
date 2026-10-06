# Portfolio-ML: learned representations and industry constraints

[![Verification](https://github.com/agraolmez/portfolio-ml/actions/workflows/verify.yml/badge.svg)](https://github.com/agraolmez/portfolio-ml/actions/workflows/verify.yml)

Code and aggregate replication materials for **Learned Representations and Industry Constraints in Implementable Portfolio Choice**, by Agra Emir Olmez, Columbia Business School.

The study compares a reconstructed Portfolio-ML baseline with learned characteristic scales, temporal transformations, valuation-dependent coefficients, and an incremental industry constraint. Both procedures use common risk, trading and cost machinery.

| Period | Baseline annual utility | Combined annual utility | Gain (percentage points) | One-sided, unadjusted p-value |
|---|---:|---:|---:|---:|
| 1981–2020: main comparison | 6.092% | 7.184% | 1.093 | 0.001865 |
| 2021–2025: nonoverlapping robustness | 4.832% | 7.012% | 2.180 | 0.044492 |

The improvements are statistically significant at 1% and 5%, respectively, under the reported one-sided normal test. Rounded utility levels and the rounded difference are calculated separately. Utility is the specified return–risk–cost objective, not CAGR. The manuscript and supplement explain development history, component comparisons and retrospective dependence sensitivities.

[Paper](paper/paper.pdf) · [Supplement](paper/supplemental_material.pdf) · [Source map](docs/SOURCE_MAP.md) · [Reconstruction guide](docs/RECONSTRUCTION.md)

## Reproduce the reported aggregate results

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_release.py
python replication/scripts/verify_results.py --output results/aggregate
python tools/run_portable_tests.py
```

The selected synthetic suite targets Linux and macOS. The aggregate verifier is independent of the historical training implementation and uses only the bundled monthly strategy returns. It reproduces six performance rows, two utility comparisons, the reported dependence sensitivities, and the arithmetic of 19 historical component summaries. Successful execution prints `PASS`. Use a new output directory on subsequent runs; existing results are never overwritten.

The portable tests exercise numerical invariants on synthetic inputs. They do not require vendor data or train portfolios. See [validation scope](docs/VALIDATION.md). The [GitHub workflow](https://github.com/agraolmez/portfolio-ml/actions/workflows/verify.yml) runs these checks on a clean checkout; its badge above shows the current status.

## Repository map

| Path | Contents |
|---|---|
| `paper/` | Author manuscript, supplement and standalone LaTeX sources |
| `replication/` | Aggregate monthly returns, expected tables and independent verifier |
| `research/` | Curated authored scientific packages, preparation scripts, execution settings and synthetic tests |
| `docs/` | Source map, reconstruction prerequisites, provenance, rights and validation scope |
| `tools/` | Release-integrity check and portable test entry point |
| `.github/workflows/` | Continuous verification after uploading to GitHub |

The authored study packages retain their relative directory structure under `research/` so existing scientific imports remain recognizable. `docs/source_inventory.csv` records original and exported hashes and any export-only changes. This release contains no original third-party implementation, vendor observations, fitted portfolios, local environments, historical Git history or private correspondence.

## Full study reconstruction

Full re-estimation requires separately licensed source data, the documented historical source vintages, upstream dependencies, earlier prepared artifacts and, for the accelerated execution path, macOS native libraries. Historical runners depend on accepted input manifests and are retained as inspectable implementation; launching a runner from a fresh clone is not a complete rebuild. [RECONSTRUCTION.md](docs/RECONSTRUCTION.md) gives the dependency order and clearly separates tested portable reproduction from this larger workflow.

The paper's reporting windows are 1981–2020 and 2021–2025. Some preserved historical configurations also contain 2020–2025, a development output that is not an additional reported robustness period.

## Citation and rights

Use [CITATION.cff](CITATION.cff) for the manuscript citation; this release does not claim a journal acceptance or DOI. [RIGHTS.md](RIGHTS.md) describes the rights applying to this release. The public repository is [agraolmez/portfolio-ml](https://github.com/agraolmez/portfolio-ml). See [maintainer instructions](docs/GITHUB_UPLOAD.md) for updating this curated release. Contact: aolmez28@gsb.columbia.edu.
