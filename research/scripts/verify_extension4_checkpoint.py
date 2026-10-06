"""Verify the retained engineering checkpoint without starting economic work."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-code', action='store_true', help='Also require live source to equal this checkpoint version')
    args = parser.parse_args()
    path = ROOT/'output/extension4_implementation_20260924/checkpoint_manifest.json'
    receipt = json.loads(path.read_text())
    if receipt['status'] != 'VERIFIED_AUDIT_AND_INITIAL_BRIDGE' or receipt['engineering_milestone_complete'] is not True:
        raise ValueError('Incomplete engineering checkpoint')
    hashes = dict(receipt['artifact_sha256'])
    if args.current_code:
        hashes.update(receipt['source_sha256'])
    hashes.update(receipt['protected_sha256'])
    for relative, expected in hashes.items():
        p = (ROOT/relative).resolve()
        if not p.is_relative_to(ROOT) or digest(p) != expected:
            raise ValueError('Changed checkpoint file: '+relative)
    print(json.dumps(dict(status='PASS', hashes_verified=len(hashes),
        live_code_checked=args.current_code, economic_study_complete=False,
        manifest_sha256=digest(path)), indent=2))


if __name__ == '__main__':
    main()
