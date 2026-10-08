import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def temp_data(tmp_path, monkeypatch):
    """Copy the CSVs to a temp dir so tests never modify the real data."""
    data = tmp_path / "data"
    data.mkdir()
    for f in ("policies.csv", "claims.csv", "claim_history.csv"):
        shutil.copy(ROOT / "data" / f, data / f)
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("CHECKPOINT_DB", str(data / "checkpoints.sqlite"))
    monkeypatch.setenv("TRACE_FILE", str(data / "traces" / "traces.jsonl"))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    return data
