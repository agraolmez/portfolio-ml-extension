"""Use fresh Step 3 helpers with a separate September 11 repair vintage."""
from pathlib import Path
import step3_io as base

ROOT = base.ROOT
PRIOR_RAW = ROOT / "data/raw/20260910"
PRIOR_OUT = ROOT / "output/step3_20260910"
OUT = ROOT / "output/step3_20260911"
RAW = ROOT / "data/raw/20260911"
PREP = ROOT / "data/prepared/step3_v2"
for p in (OUT, RAW, PREP, OUT / "queries"):
    p.mkdir(parents=True, exist_ok=True)
base.OUT, base.RAW, base.PREP = OUT, RAW, PREP
connect, extract, sha, utc, save_json = base.connect, base.extract, base.sha, base.utc, base.save_json


def resource_guard():
    report = base.resource_guard()
    total = sum(p.stat().st_size for parent in (ROOT / "data", ROOT / "output/step3_20260910", OUT, ROOT / "env/step3") for p in parent.rglob("*") if p.is_file()) / 2**30
    if total > 60:
        raise RuntimeError("Combined fresh Step 3 storage exceeds 60 GiB")
    return {**report, "combined_step3_storage_gib": total}
