"""Entry point; a pending proposal cannot authorize economic execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extension4.run import main

if __name__=='__main__':main()
