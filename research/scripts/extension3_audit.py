"""Run the bounded Extension-3 parent audit; never launch historical fitting."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal_extension.audit import main

if __name__ == "__main__":
    main()
