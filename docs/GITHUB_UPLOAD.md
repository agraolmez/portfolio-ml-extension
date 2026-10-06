# Maintaining the public repository

The repository is [agraolmez/portfolio-ml-extension](https://github.com/agraolmez/portfolio-ml-extension). Work in this curated checkout; the private study checkout retains other research artifacts and has a separate history.

```sh
git clone https://github.com/agraolmez/portfolio-ml-extension.git
cd portfolio-ml-extension
```

Follow the root README to install the dependencies and reproduce the saved results. Preserve the aggregate input and expected-table manifests unless publishing an explicitly documented scientific revision. The root MANIFEST.json records the distributed files. Intentional edits require reviewing and refreshing the affected root file hashes before running tools/check_release.py and the CI workflow.

Keep results/, virtual environments, reconstructed licensed data and fitted model outputs ignored. The exact factual settings JSON under research/output/ is the documented exception. Do not copy private correspondence, upstream source code or the original study's Git history into this repository. Code reuse follows RIGHTS.md.
