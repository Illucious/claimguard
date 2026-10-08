"""Central configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


class Settings:
    def __init__(self) -> None:
        # ---- LLM -------------------------------------------------------------
        self.llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()        # groq | gemini
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_summary_model = os.getenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.gemini_summary_model = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-2.5-flash")

        # ---- Tools / MCP -----------------------------------------------------
        self.tool_mode = os.getenv("TOOL_MODE", "mcp").lower()               # mcp | local
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "")                # empty → spawn stdio server

        # ---- Persistence ------------------------------------------------------
        self.data_dir = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
        self.checkpoint_db = Path(os.getenv("CHECKPOINT_DB", str(self.data_dir / "checkpoints.sqlite")))
        self.trace_file = Path(os.getenv("TRACE_FILE", str(self.data_dir / "traces" / "traces.jsonl")))

        # ---- Agent loop limits (Task 7) --------------------------------------
        self.compaction_threshold = _int("COMPACTION_THRESHOLD", 10)   # summarize when len(messages) > this
        self.compaction_batch = _int("COMPACTION_BATCH", 8)            # summarize the first N messages
        self.max_steps = _int("MAX_REASONING_STEPS", 15)
        self.recursion_limit = _int("RECURSION_LIMIT", 60)

        # ---- Business thresholds (must match insurance_tools/rules.py) -------
        self.high_value_threshold = float(os.getenv("HIGH_VALUE_THRESHOLD", "500000"))
        self.risk_medium_min = _int("RISK_MEDIUM_MIN", 30)
        self.risk_high_min = _int("RISK_HIGH_MIN", 60)

        # ---- API ---------------------------------------------------------------
        self.api_key = os.getenv("API_KEY", "")                        # optional X-API-Key for /webhook


settings = Settings()
