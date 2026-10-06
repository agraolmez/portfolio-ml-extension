"""Check the release inventory without importing historical research runners."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'MANIFEST.json').read_text())
for name, expected in manifest['files_sha256'].items():
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe inventory path')
    if hashlib.sha256((root / p).read_bytes()).hexdigest() != expected:
        raise ValueError('Release file changed: ' + name)
print('PASS: release inventory verified (' + str(len(manifest['files_sha256'])) + ' files)')
