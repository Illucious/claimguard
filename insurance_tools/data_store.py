"""CSV-backed data access layer.

This is the "external system" that the MCP server wraps. The agent never reads
these files directly — it can only reach them through tools.

Files are re-read on every call so manual edits to the CSVs show up instantly.
"""
from __future__ import annotations

import csv
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOCK = threading.RLock()

POLICIES_FILE = "policies.csv"
CLAIMS_FILE = "claims.csv"
HISTORY_FILE = "claim_history.csv"


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))


def _read_with_header(filename: str) -> tuple[list[dict], list[str]]:
    path = data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with _LOCK, path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items() if k} for row in reader]
    return rows, fieldnames


def _read(filename: str) -> list[dict]:
    return _read_with_header(filename)[0]


def _write(filename: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Atomic write: write to a temp file, then replace."""
    path = data_dir() / filename
    with _LOCK:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)


# --------------------------------------------------------------------------- reads
def get_policy(policy_id: str) -> Optional[dict]:
    pid = policy_id.strip().upper()
    return next((p for p in _read(POLICIES_FILE) if p["policy_id"].upper() == pid), None)


def find_policies_by_customer(name: str) -> list[dict]:
    needle = name.strip().lower()
    return [p for p in _read(POLICIES_FILE) if needle in p["customer_name"].lower()]


def get_claim(claim_id: str) -> Optional[dict]:
    cid = claim_id.strip().upper()
    return next((c for c in _read(CLAIMS_FILE) if c["claim_id"].upper() == cid), None)


def claims_for_policy(policy_id: str) -> list[dict]:
    pid = policy_id.strip().upper()
    return [c for c in _read(CLAIMS_FILE) if c["policy_id"].upper() == pid]


def history_for_policy(policy_id: str) -> list[dict]:
    pid = policy_id.strip().upper()
    return [h for h in _read(HISTORY_FILE) if h["policy_id"].upper() == pid]


# -------------------------------------------------------------------------- writes
def update_claim(claim_id: str, updates: dict) -> Optional[tuple[dict, dict]]:
    """Update one claim row. Returns (old_row, new_row) or None if not found."""
    cid = claim_id.strip().upper()
    with _LOCK:
        rows, fieldnames = _read_with_header(CLAIMS_FILE)
        for i, row in enumerate(rows):
            if row["claim_id"].upper() == cid:
                old = dict(row)
                row.update({k: str(v) for k, v in updates.items() if k in fieldnames})
                rows[i] = row
                _write(CLAIMS_FILE, rows, fieldnames)
                return old, row
    return None
