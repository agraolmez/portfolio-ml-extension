#!/usr/bin/env python3
"""Optimized-computation full Extension-3 entry point: requires a separate approved pilot/run receipt."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--approval', required=True, type=Path,
                        help='Explicit full-run authorization binding config and successful pilot hashes')
    parser.add_argument('--budget-extension', action='append', default=[], type=Path,
                        help=('Explicit chained increase to the cumulative wall budget; repeat for additional receipts. '
                              'Does not authorize retrying partial or nonconverged fits.'))
    args = parser.parse_args()
    from temporal_fast_v2.study import run_study
    from temporal_fast.backend import single_thread_blas
    import pyarrow as pa
    # Preparation is serial; independent monthly numerical work has four workers.
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    with single_thread_blas():
        run_study(ROOT, args.output, config_path=args.config, approval_path=args.approval,
                  budget_extensions=args.budget_extension)


if __name__ == '__main__':
    main()
