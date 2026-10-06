"""Synthetic software tests plus a bounded neutral archive-loader check; no fits."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.full import preflight
from feature_geometry.history import CanonicalHistory, sha
from temporal_extension.history import AcceptedTemporalHistory, load_parent


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT/'output' or not out.is_relative_to(ROOT/'output'):
        raise ValueError('Use a new folder under the canonical output directory')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        log = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        write(out/'tests.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
                                     tests_run=result.testsRun, failures=len(result.failures),
                                     errors=len(result.errors), log=log.getvalue(), historical_fit_run=False))
        if not result.wasSuccessful():
            raise RuntimeError('Software tests failed; see tests.json')
        print(f'{result.testsRun} software tests passed; checking accepted inputs', flush=True)
        receipt, _ = preflight()
        write(out/'financial_preflight.json', receipt)
        # Test-fixture weights exercise the proposed six-lag adapter. They are
        # not adopted historical settings, and all gains remain exactly one.
        history = AcceptedTemporalHistory(ROOT, receipt, weights=np.ones(6)/6.)
        original = CanonicalHistory(ROOT, '1981-11-30', '1981-12-31', receipt)
        checks = []
        with threadpool_limits(limits=4):
            for day, year in [('1981-11-30', 1981), ('1981-12-31', 1982)]:
                parent = load_parent(ROOT, year, 1, expected_cutoff=f'{year-1}-12-31')
                parent.assert_decision(day)
                actual, expected = history.market(day), original.market(day)
                np.testing.assert_array_equal(actual.ids, expected.ids)
                for field in ('characteristics', 'volatility', 'growth'):
                    for a, b in zip(getattr(actual, field), getattr(expected, field)):
                        np.testing.assert_array_equal(a, b)
                for field in ('m', 'covariance', 'impact'):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
                assert actual.wealth == expected.wealth and actual.gamma == expected.gamma
                bound = history.innovations(actual)
                innovations = history.validate_market(actual, bound)
                errors = []
                for z, vol, innovation in zip(actual.characteristics, actual.volatility, innovations):
                    new = parent.feature_map.signals(z, vol, innovation, np.ones(13))
                    old = parent.model.feature_map.signals(z, vol, parent.model.theta)
                    np.testing.assert_array_equal(new, old)
                    errors.append(float(np.max(abs(new-old))))
                np.testing.assert_array_equal(parent.feature_map.signals(actual.characteristics[0],
                    actual.volatility[0], innovations[0], np.ones(13)) @ parent.model.beta,
                    parent.model.aim(expected.characteristics[0], expected.volatility[0], day))
                coverage = {key:sum(x.reasons[key] for x in innovations) for key in innovations[0].reasons}
                checks.append(dict(date=day, parent_year=year, parent_fingerprint=parent.fingerprint,
                                   stocks=len(actual.ids), feature_slice_max_error=max(errors),
                                   cached_months=len(history.cached_months), cache_limit=history.cache_limit,
                                   coverage_all_13_slices=coverage,
                                   source_fingerprint=history.source_fingerprint,
                                   reference_fingerprint=history.reference_fingerprint))
                print(f'Neutral dated history check passed: {day}', flush=True)
        history.files.unchanged()
        # Final cryptographic check supplements per-read file-stat mutation guards.
        for relative, expected_hash in history.files.consumed.items():
            if sha(ROOT/relative) != expected_hash:
                raise ValueError('Accepted input changed during check: '+relative)
        write(out/'neutral_history_adapter.json', dict(status='PASS', checks=checks,
              gains='all ones', historical_parameters_chosen=False, historical_fit_run=False,
              fixture_reference='six equal calendar lags, solely to test the adapter',
              full_period_coverage_measured=False, consumed_input_sha256=dict(history.files.consumed)))
        files = [*sorted((ROOT/'temporal_extension').glob('*.py')),
                 *sorted((ROOT/'tests').glob('test_temporal_*.py')), Path(__file__).resolve()]
        manifest = dict(status='PASS_IMPLEMENTATION_CHECKS', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        seconds=time.monotonic()-start, tests_run=result.testsRun,
                        bounded_neutral_history_months=2, historical_fit_run=False,
                        full_experiment_run=False, paid_resources_used=False,
                        scientific_settings_approved=False,
                        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
                        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in out.iterdir() if p.is_file()})
        write(out/'implementation_checks.json', manifest)
        print(json.dumps({k:manifest[k] for k in ('status','tests_run','seconds')}, indent=2), flush=True)
    except Exception as exc:
        write(out/'failed_attempt.json', dict(status='FAILED_PRESERVED', error_type=type(exc).__name__,
                                             error=str(exc), historical_fit_run=False))
        raise


if __name__ == '__main__':
    main()
