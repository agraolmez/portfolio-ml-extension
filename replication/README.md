# Independent aggregate replication

From the repository root, follow the environment setup in ../README.md and run:

```sh
python replication/scripts/verify_results.py --output results/aggregate
```

The verifier refuses an existing output directory or an output inside this replication directory. It checks all bundled hashes and recalculates six performance rows, two final comparisons, HAC p-values at five lag choices and two stationary-bootstrap confidence intervals per reporting period. It also checks the arithmetic of 19 retained component contrasts. The expected tables and definitions are unchanged from the verified anonymous reviewer package.

Annual utility is 12 times mean net excess return minus 60 times sample variance of gross excess return. The one-sided comparison uses monthly utility flows and the reported normal reference. These are retrospective results conditional on the saved model paths, not full-data re-estimation or a correction for the complete specification search. See the paper, supplement and ../docs/RECONSTRUCTION.md for scope and data conventions.

Implementation source is organized once at ../research/. PROVENANCE.json binds these aggregate inputs to the retained study hashes. No security-level observations or vendor credentials are included.
