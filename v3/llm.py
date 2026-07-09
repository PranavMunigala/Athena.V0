"""Small Instructor/OpenAI helpers for Athena v3."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Type, TypeVar

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def structured_client():
    """Return an Instructor-wrapped OpenAI client."""
    return instructor.from_openai(OpenAI())


def structured_completion(
    *,
    response_model: Type[T],
    system: str,
    user: str,
    temperature: float = 0.2,
) -> T:
    """Create a structured response with Instructor."""
    model = os.getenv("ATHENA_STRUCTURED_MODEL", os.getenv("ATHENA_CHAT_MODEL", "gpt-4o-mini"))
    return structured_client().chat.completions.create(
        model=model,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_retries=2,
    )


def text_completion(
    *,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    """Create a plain chat completion for synthesis steps."""
    model = os.getenv("ATHENA_CHAT_MODEL", "gpt-4o-mini")
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def compact_lines(lines: Iterable[str], limit: int = 12) -> str:
    """Bound long tool outputs before putting them back into prompts."""
    clean = [line.strip() for line in lines if line and line.strip()]
    return "\n".join(clean[:limit])
