# Reproduction and full reconstruction

## Tested route: aggregate results

Run the commands in the root README. `replication/scripts/verify_results.py` checks its input hashes, recomputes the six performance rows and two final comparisons from 540 months of aggregate strategy returns, and reproduces the retrospective HAC and stationary-bootstrap diagnostics. It also checks the 19 component summary contrasts. It does not rerun the historical component bootstrap draws or model selection.

The monthly inputs contain only dates, model labels and aggregate gross excess return, impact cost and net excess return. No identifiers, stock-level returns, holdings or licensed vendor tables are distributed. The expected tables retain the study's definitions and units.

## Research implementation

`research/` contains the authored baseline and extension packages and their dependency closure. The source inventory identifies export adaptations; original scientific files in the retained study are unchanged. The independent public E4 helper, if listed in the inventory, implements the documented feature-interaction operation and is validated against the retained reference; it does not redistribute the supplied reference source.

Historical runners have resource policies, fixed stage destinations and accepted-input guards. They are documentary entry points, not a one-command fresh-clone refit. Read SOURCE_MAP.md before using them.

## Dependency order for full re-estimation

1. Obtain permitted original Portfolio-ML source and feature dictionary from the authors' release. The original article is [Machine Learning and the Implementable Efficient Frontier](https://doi.org/10.1093/rfs/hhag022). The characteristic infrastructure is [Is There a Replication Crisis in Finance?](https://doi.org/10.1111/jofi.13249). Upstream code retains its own rights and is excluded from this package.
2. Acquire licensed JKP monthly characteristics and returns, CRSP CIZ monthly/daily security and event histories, CRSP 30-day bill returns, and the accounting data underlying the characteristics. Obtain the French risk-free and official JKP market reference series separately. The accepted study used September 2026 source captures, not today's vendor snapshot.
3. Rebuild dated raw extracts and the preparation artifacts shown in `research/scripts/step3_author_bundle.py` and `step3_author_complete_inputs.py`: return conventions, ranks, eligibility, reciprocal wealth, reference rates, risk inputs and integrity receipts. The original query receipts and security-level arrays are not bundled. Paths in preserved source describe the historical stage topology.
4. Reconstruct baseline matrices/moments; E2 scale-study artifacts; E3 temporal-study parents; E4 states and conditional estimators; then final annual candidate banks. The final `annual_staged` catalog requires preceding stage manifests. Those dependencies must be recreated and independently validated before running its selection and continuous-holdings evaluation.
5. Match Python and numerical dependencies. The executed native v3/v4 route uses Clang, the macOS SDK and Accelerate. Other platforms require a validated replacement for that accelerated route. The aggregate verifier and selected synthetic tests use the portable dependencies in requirements.txt.
6. Use an isolated research root, separately chosen outputs and an explicit computing budget. Generate new verification receipts rather than rewriting the accepted historical receipts. No full licensed-data reconstruction or new portfolio experiment was performed in preparing this release.

## Scientific conventions

The comparison fixes risk aversion ten and a common wealth path anchored at USD 10 billion in December 2020. It retains the documented availability/universe conventions, missing-return treatment and entry/exit mechanics; these limit point-in-time and trading interpretations. Full model fitting, specification search and historical data availability are not certified by aggregate replication. The manuscript and supplement provide the complete discussion.
