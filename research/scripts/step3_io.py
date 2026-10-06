"""Fresh, read-only Step 3 acquisition helpers; no pre-reset dependencies."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import resource
import shutil
import time
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/step3_20260910"
RAW = ROOT / "data/raw/20260910"
PREP = ROOT / "data/prepared/step3_v1"
_storage_checked_at = 0.0
_storage_gib = 0.0
for path in (OUT, RAW, PREP, OUT / "queries"):
    path.mkdir(parents=True, exist_ok=True)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if x is pd.NA or x is pd.NaT:
            return None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False) + "\n")


def resource_guard():
    global _storage_checked_at, _storage_gib
    free = shutil.disk_usage(ROOT).free / 2**30
    if time.monotonic() - _storage_checked_at > 60:
        roots = [RAW, PREP, OUT, ROOT / "env/step3"]
        _storage_gib = sum(p.stat().st_size for root in roots for p in root.rglob("*") if p.is_file()) / 2**30
        _storage_checked_at = time.monotonic()
    # macOS reports bytes, Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gib = rss / (2**30 if os.uname().sysname == "Darwin" else 2**20)
    if free < 30 or rss_gib > 24 or _storage_gib > 60:
        raise RuntimeError(f"Approved resource boundary reached: free={free:.2f} GiB, peak RSS={rss_gib:.2f} GiB, new storage={_storage_gib:.2f} GiB")
    return {"free_disk_gib": free, "process_peak_rss_gib": rss_gib, "new_storage_gib_last_minute": _storage_gib}


def connect():
    # libpq reads the password itself. Never copy or log credential values.
    username = os.getenv("WRDS_USERNAME")
    if not username:
        for line in (Path.home() / ".pgpass").read_text().splitlines():
            if line.startswith("wrds-pgdata.wharton.upenn.edu:9737:wrds:"):
                username = line.split(":")[3]
                break
    if not username:
        raise RuntimeError("No configured WRDS username; configure local libpq authentication.")
    return psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds",
        user=username, sslmode="require", connect_timeout=20,
        options="-c default_transaction_read_only=on -c statement_timeout=180000",
        application_name="PortfolioML_Step3_readonly",
    )


def extract(conn, name, query, params=None, chunk_rows=25000):
    """Stream one bounded SELECT to immutable Parquet plus an acquisition receipt."""
    resource_guard()
    path = RAW / f"{name}.parquet"
    receipt = RAW / f"{name}.receipt.json"
    query_path = OUT / "queries" / f"{name}.sql"
    fingerprint = hashlib.sha256((query + json.dumps(params, default=str)).encode()).hexdigest()
    if path.exists() != receipt.exists():
        raise RuntimeError(f"Incomplete file/receipt pair: {name}; inspect before resuming.")
    if path.exists() and receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior["query_fingerprint"] != fingerprint or prior["sha256"] != sha(path):
            raise RuntimeError(f"Existing acquisition differs: {name}; use a new vintage/name.")
        return prior
    query_path.write_text(query + "\n")
    save_json(query_path.with_suffix(".params.json"), params)
    temp = path.with_suffix(".partial")
    started = utc()
    timer = time.monotonic()
    count = 0
    writer = None
    try:
        with conn.cursor(name="step3_extract") as cur:
            cur.itersize = chunk_rows
            cur.execute(query, params)
            while rows := cur.fetchmany(chunk_rows):
                columns = [item.name for item in cur.description]
                frame = pd.DataFrame.from_records(rows, columns=columns, coerce_float=True)
                # Known identifier/date strings have stable types, even in all-null chunks.
                for desc in cur.description:
                    col = desc.name
                    if desc.type_code in {18, 19, 25, 1042, 1043}:
                        frame[col] = frame[col].astype("string")
                    elif desc.type_code in {1082, 1114, 1184}:
                        frame[col] = pd.to_datetime(frame[col])
                    elif desc.type_code in {20, 21, 23, 700, 701, 1700}:
                        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(temp, schema, compression="zstd")
                else:
                    table = table.cast(schema)
                writer.write_table(table)
                count += len(frame)
                resource_guard()
        conn.commit()
        if writer is None:
            raise RuntimeError(f"Required extraction returned no rows: {name}")
        writer.close()
        writer = None
        temp.replace(path)
        record = {
            "name": name, "source": "WRDS PostgreSQL", "read_only": True,
            "query_path": str(query_path.relative_to(ROOT)), "query_fingerprint": fingerprint,
            "parameters": params, "started_at_utc": started, "finished_at_utc": utc(),
            "rows": count, "columns": pq.read_schema(path).names,
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha(path), "seconds": time.monotonic() - timer,
            "resources": resource_guard(),
        }
        save_json(receipt, record)
        print(json.dumps({"extract": name, "rows": count, "seconds": round(record["seconds"], 1)}), flush=True)
        return record
    except BaseException:
        conn.rollback()
        if writer:
            writer.close()
        temp.unlink(missing_ok=True)
        raise


def write_prepared(name, frame):
    path = PREP / f"{name}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    return {"path": str(path.relative_to(ROOT)), "rows": len(frame), "sha256": sha(path)}
