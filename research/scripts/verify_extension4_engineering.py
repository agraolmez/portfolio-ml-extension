"""Verify the current E4 engineering evidence without launching research."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extension4.run import require_engineering
from feature_geometry.history import sha


def main():
    digest=require_engineering(ROOT)
    path=ROOT/'output/extension4_engineering_20260925/engineering_manifest.json'
    receipt=json.loads(path.read_text())
    for rel,expected in {**receipt['source_sha256'],**receipt['protected_sha256']}.items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=expected:raise ValueError('Changed checkpoint dependency: '+rel)
    print(json.dumps(dict(status='PASS',manifest_sha256=digest,
        engineering_artifacts=len(receipt['artifact_sha256']),current_sources=len(receipt['source_sha256']),
        protected_files=len(receipt['protected_sha256']),historical_study_complete=False),indent=2))


if __name__=='__main__':main()
