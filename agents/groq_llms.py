"""Groq clients: reload key from .env each call (no stale cache) and normalize formatting."""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

_DOTENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _normalize_api_key(raw: str | None) -> str:
    if not raw:
        return ""
    s = raw.strip().strip("\ufeff")
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def _api_key() -> str:
    load_dotenv(_DOTENV, override=True)
    return _normalize_api_key(os.getenv("GROQ_API_KEY"))


def chat_groq_70b() -> ChatGroq:
    key = _api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a `.env` file in the project root "
            "(copy from `.env.example`) and add your Groq API key."
        )
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=key)


def chat_groq_8b() -> ChatGroq:
    key = _api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a `.env` file in the project root "
            "(copy from `.env.example`) and add your Groq API key."
        )
    return ChatGroq(model="llama-3.1-8b-instant", api_key=key)
