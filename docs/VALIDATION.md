# Validation scope

The release preparation validates an isolated copy with the listed Python dependencies, its file inventory, independent aggregate reproduction, and selected synthetic numerical tests. See the release's source map for the selected suite. The [CI workflow](https://github.com/agraolmez/portfolio-ml/actions/workflows/verify.yml) repeats these portable checks on GitHub. Its run history reports the actual result for each commit.

The aggregate verifier checks both reporting periods, all six model-period metric rows, both final comparisons, all reported dependence diagnostics and the arithmetic of 19 historical component summaries. It verifies package hashes before calculation and refuses to overwrite an existing output directory. Tests of future-outcome invariance, selection chronology and an independent dense projection oracle are included among the synthetic contracts.

Historical native/reconstruction tests remain inspectable in research/tests/ but are not all portable: some require macOS Accelerate, original-source prerequisites, accepted data receipts or licensed observations. Running every historical test from a fresh clone is not a supported verification command. Use tools/run_portable_tests.py for the selected suite. Full data acquisition and refitting were not run for this release.
