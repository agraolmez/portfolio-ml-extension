"""Explicit Extension-3 study entry points. Importing or reviewing runs nothing."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pilot'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/extension3_study_v1.json')
    parser.add_argument('--prior-attempt', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.stage == 'pilot':
        from temporal_study.pilot import run_pilot
        run_pilot(ROOT, args.output, config_path=args.config, prior_attempts=args.prior_attempt)


if __name__ == '__main__':
    main()
