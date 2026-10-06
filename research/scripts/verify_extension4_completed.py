"""Read-only verification of the completed Extension-4 acceptance inventory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root/'output/extension4_full_study_20260924/acceptance_manifest.json'
    accepted = json.loads(path.read_text())
    if (accepted.get('status') != 'ACCEPTED_COMPLETED_HISTORICAL_CONDITIONAL_STUDY'
            or accepted.get('study_complete') is not True or accepted.get('accepted') is not True
            or not accepted.get('gates') or not all(g['passed'] is True for g in accepted['gates'])
            or accepted.get('monthly_records') != 11340 or accepted.get('economic_fits') != 1035):
        raise ValueError('Complete accepted study required')
    hashes = accepted['artifact_sha256']
    if not hashes:raise ValueError('Empty acceptance inventory')
    for relative, expected in hashes.items():
        source = (root/relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError('Missing or out-of-project accepted file: '+relative)
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            if sys.platform == 'darwin':
                import fcntl
                fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
            for block in iter(lambda: handle.read(2**20), b''):digest.update(block)
        if digest.hexdigest() != expected:raise ValueError('Changed accepted artifact: '+relative)
    result = dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  accepted_files_verified=len(hashes),study_complete=True,
                  financial_work_launched=False,artifact_files_modified=False)
    if args.receipt:
        with args.receipt.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
