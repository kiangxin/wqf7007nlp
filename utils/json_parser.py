"""
Robust LLM JSON parser — handles all edge cases encountered in practice.

Public API:
    parse_llm_json(raw: str) -> dict | None

Cases handled (in order of parsing passes):
  1.  Markdown code fences (```json ... ``` or ``` ... ```)
  2.  <think>...</think> reasoning blocks (Qwen3 / DeepSeek-R1)
  3.  Direct json.loads() on cleaned text
  4.  Trailing commas before } or ]
  5.  Bracket-balanced JSON extraction from preamble / postamble text
  6.  Truncated / incomplete JSON — repair by closing open strings + containers
  7.  Python-style single-quoted dicts (last resort)

Compatible with Python 3.9+.
"""
from __future__ import annotations  # enables X | Y union hints on Python 3.9

import json
import re


# ── Pre-processing ─────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences."""
    # Leading: ```json or ``` (with optional whitespace / newline)
    text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
    # Trailing: ``` (with optional whitespace)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks emitted by Qwen3 / DeepSeek-R1.
    Must be applied BEFORE bracket extraction so the first '{' is the JSON body,
    not an opening brace inside a <think> block.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] — a common LLM formatting error."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_python_literals(text: str) -> str:
    """
    Best-effort conversion of Python-style output to JSON:
      True/False/None → true/false/null
      single-quoted strings → double-quoted (naïve — skips embedded single quotes)
    """
    text = re.sub(r"\bTrue\b",  "true",  text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b",  "null",  text)
    text = re.sub(r"'([^']*)'", r'"\1"', text)
    return text


# ── JSON object extraction ─────────────────────────────────────────────────────

def _extract_json_object(text: str) -> str | None:
    """
    Extract the first complete, bracket-balanced JSON object { ... } from
    arbitrary text.  Handles nested objects / arrays and strings containing
    braces.  Returns the extracted string or None if no balanced object found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # no balanced object found (truncated)


# ── Truncation repair ──────────────────────────────────────────────────────────

def _repair_truncated(text: str) -> str:
    """
    Close a JSON string that was cut off before the final token:

    Algorithm:
      Walk character by character tracking string/escape state and an open-
      container stack ({ or [).  After the last character:
        1. If still inside a string literal, append a closing quote.
        2. Append closing delimiters for each still-open container in
           reverse stack order.

    The result is syntactically valid JSON (parseable) even though some field
    values will be incomplete strings.  Callers should treat results from this
    pass as partial / best-effort data.

    Example
    -------
    Input:  '{"detected_aspects": [{"id": 8, "category": "staff-support'
    Output: '{"detected_aspects": [{"id": 8, "category": "staff-support"}]}'
    """
    stack: list[str] = []   # stack of opening delimiters still open
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    result = text
    if in_string:
        result += '"'           # close the dangling string literal

    close = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        result += close[opener] # close every unclosed container

    return result


# ── Main entry point ───────────────────────────────────────────────────────────

def parse_llm_json(raw: str) -> dict | None:
    """
    Best-effort JSON parser for LLM output.

    Pass 1  Strip fences + <think> blocks → direct json.loads()
    Pass 2  Fix trailing commas → json.loads()
    Pass 3  Bracket-balanced extraction → json.loads()  (handles preamble text)
    Pass 4  Truncation repair → close open strings/brackets → json.loads()
    Pass 5  Python-literal fix (single quotes, True/False) → json.loads()

    Returns the parsed dict, or None if all passes fail.
    """
    if not raw:
        return None

    # ── Pre-process ───────────────────────────────────────────────────────────
    text = raw.strip()
    text = _strip_fences(text)
    text = _strip_think_blocks(text)
    text = text.strip()

    # ── Pass 1: direct parse ──────────────────────────────────────────────────
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # ── Pass 2: fix trailing commas ───────────────────────────────────────────
    text_clean = _fix_trailing_commas(text)
    try:
        result = json.loads(text_clean)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # ── Pass 3: bracket-balanced extraction ───────────────────────────────────
    candidate = _extract_json_object(text_clean)
    if candidate:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        cand_clean = _fix_trailing_commas(candidate)
        try:
            result = json.loads(cand_clean)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    else:
        cand_clean = None

    # ── Pass 4: truncation repair ─────────────────────────────────────────────
    # Try each available base string (most processed → least processed)
    for base in filter(None, [cand_clean, candidate, text_clean, text]):
        repaired = _repair_truncated(base)
        repaired = _fix_trailing_commas(repaired)   # commas may appear at cut-off point
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # ── Pass 5: Python-literal fallback ──────────────────────────────────────
    py_fixed = _fix_python_literals(text)
    try:
        result = json.loads(py_fixed)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None
