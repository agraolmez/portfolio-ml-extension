"""Replay explicitly selected retained WRDS queries into their original vintage.

Existing completed files are hash-checked and reused. Deleting a raw file alone
is not a refresh request: retain the vintage and use a new name for changed data.
"""
import argparse
import json
from step3_io import OUT, connect, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Exact extraction basenames shown in raw_inventory.csv")
    args = parser.parse_args()
    selected = []
    for name in args.names:
        if not name.replace("_", "").isalnum():
            raise ValueError("Expected an extraction basename")
        path = OUT / "queries" / f"{name}.sql"
        query = path.read_text().removesuffix("\n")
        if not query.lstrip().upper().startswith("SELECT "):
            raise ValueError("Only retained read-only SELECT queries can be replayed")
        selected.append((name, query, json.loads(path.with_suffix(".params.json").read_text())))
    with connect() as conn:
        for name, query, params in selected:
            extract(conn, name, query, params)


if __name__ == "__main__":
    main()
