"""Run the declared synthetic subset in an isolated temporary source copy.

No licensed observations, model selection, historical portfolio replay, R, or
macOS native build is required. Full test discovery is intentionally not used:
retained historical suites have documented platform/archive dependencies.
"""
import argparse, hashlib, importlib.metadata, io, json, os
from pathlib import Path
import shutil, subprocess, sys, tempfile

BASE = Path(__file__).resolve().parent
TESTS = ('test_feature_geometry', 'test_geometry_study', 'test_temporal_extension', 'test_temporal_optimizer', 'test_temporal_streaming', 'test_temporal_training', 'test_temporal_validation', 'test_temporal_selection', 'test_temporal_analysis', 'test_annual_ind_selection', 'test_annual_ind_checkpoints', 'test_annual_ind_conditional', 'test_extension4', 'test_adaptive_overlays', 'test_portfolio_ensemble_screen', 'test_industry_projection', 'test_public_feature_interactions')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New results directory outside this source tree')
    parser.add_argument('--inside-isolated-copy', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside_isolated_copy:
        # Fail explicitly if the supposedly portable tests invoke the native backend.
        sys.path[:0] = [str(BASE), str(BASE/'tests'), str(BASE/'scripts')]
        import unittest
        from unittest.mock import patch
        from contextlib import ExitStack
        # Block optional macOS backends even on the original development host.
        blocked = [
            'temporal_fast.backend.accelerate',
            'temporal_fast_v3.native.library',
            'temporal_fast_v4.native.library',
            'temporal_fast_v4.source_checks.library',
            'temporal_fast_v4.memory._function',
        ]
        suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
        log = io.StringIO()
        with ExitStack() as guards:
            for target in blocked:
                guards.enter_context(patch(target, side_effect=AssertionError('Native backend is excluded from portable tests')))
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            skipped=len(result.skipped), modules=TESTS,
            versions={n:importlib.metadata.version(n) for n in ('numpy','pandas','scipy','pyarrow','psutil','threadpoolctl')},
            python=sys.version.split()[0], synthetic_only=True,
            native_backend_calls_blocked=True,
            historical_portfolio_paths_rerun=False, licensed_data_used=False,
            excluded_suites='Platform-specific native, R-reference, operational and archived-data suites are not in this subset.')
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'synthetic_tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (args.output/'synthetic_tests.txt').write_text(log.getvalue())
        print(json.dumps(receipt,indent=2))
        if not result.wasSuccessful(): print(log.getvalue(), file=sys.stderr)
        return 0 if result.wasSuccessful() else 1
    if args.output:
        args.output = args.output.resolve()
        if args.output.is_relative_to(BASE) or args.output.exists():
            parser.error('--output must be a new directory outside research/')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='portfolio-source-test-') as tmp:
        folder = Path(tmp)/'research'
        shutil.copytree(BASE,folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
        output = args.output or Path(tmp)/'results'
        command = [sys.executable,'-B',str(folder/'run_portable_tests.py'),
                   '--inside-isolated-copy','--output',str(output)]
        result = subprocess.run(command,cwd=folder,env=env)
        if args.output: print('Synthetic test results:',args.output)
        return result.returncode

if __name__ == '__main__': raise SystemExit(main())
