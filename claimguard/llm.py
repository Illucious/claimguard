"""LLM factory: Groq or Gemini, chosen by LLM_PROVIDER."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Settings, settings as default_settings


def create_chat_model(purpose: str = "agent", cfg: Settings | None = None) -> BaseChatModel:
    """purpose: 'agent' (reasoning), 'summary' (Task 7 compaction) or 'judge' (Task 9 eval)."""
    cfg = cfg or default_settings
    if cfg.llm_provider == "groq":
        from langchain_groq import ChatGroq

        model = cfg.groq_summary_model if purpose == "summary" else cfg.groq_model
        return ChatGroq(model=model, temperature=0, max_retries=2)

    if cfg.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = cfg.gemini_summary_model if purpose == "summary" else cfg.gemini_model
        return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=2)

    raise ValueError(f"Unsupported LLM_PROVIDER '{cfg.llm_provider}'. Use 'groq' or 'gemini'.")


def message_text(message) -> str:
    """Normalise message content (Gemini may return a list of parts) to plain text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)
