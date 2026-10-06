"""Run the selected synthetic contracts using this interpreter."""
from pathlib import Path
import subprocess, sys
root = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable, str(root / 'research/run_portable_tests.py')], cwd=root))
