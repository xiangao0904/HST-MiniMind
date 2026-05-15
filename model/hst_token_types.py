from __future__ import annotations

import json
import string
import unicodedata
from pathlib import Path

TOKEN_TYPE_NORMAL = 0
TOKEN_TYPE_PUNCT = 1
TOKEN_TYPE_NEWLINE = 2
TOKEN_TYPE_DIGIT = 3
TOKEN_TYPE_LATIN = 4
TOKEN_TYPE_CJK = 5
TOKEN_TYPE_BRACKET = 6
TOKEN_TYPE_CODE_SYMBOL = 7
TOKEN_TYPE_WHITESPACE = 8
TOKEN_TYPE_SPECIAL = 9
TOKEN_TYPE_UNKNOWN = 10
TOKEN_TYPE_NAMES = [
    "normal",
    "punctuation",
    "newline",
    "digit",
    "latin",
    "cjk",
    "bracket",
    "code_symbol",
    "whitespace_like",
    "special_token",
    "unknown",
]

BRACKETS = set("()[]{}<>（）【】《》")
CODE_SYMBOLS = set("=+-*/%_#@$\\|`~^&:;.")
PUNCT = set("，。！？、,.!?\"'“”‘’")


def classify_token_text(text: str) -> int:
    if text == "":
        return TOKEN_TYPE_UNKNOWN
    if text.startswith("<") and text.endswith(">"):
        return TOKEN_TYPE_SPECIAL
    if "\n" in text or "\r" in text:
        return TOKEN_TYPE_NEWLINE
    if text.isspace():
        return TOKEN_TYPE_WHITESPACE
    if all(ch.isdigit() for ch in text):
        return TOKEN_TYPE_DIGIT
    if all(("a" <= ch.lower() <= "z") for ch in text if ch.strip()):
        return TOKEN_TYPE_LATIN
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return TOKEN_TYPE_CJK
    if any(ch in BRACKETS for ch in text):
        return TOKEN_TYPE_BRACKET
    if any(ch in CODE_SYMBOLS for ch in text):
        return TOKEN_TYPE_CODE_SYMBOL
    if any(ch in PUNCT or unicodedata.category(ch).startswith("P") for ch in text):
        return TOKEN_TYPE_PUNCT
    if all(ch in string.printable for ch in text):
        return TOKEN_TYPE_NORMAL
    return TOKEN_TYPE_UNKNOWN


def build_token_type_cache(id_to_text: dict[int, str], cache_path: str | Path | None = None) -> list[int]:
    max_id = max(id_to_text) if id_to_text else -1
    types = [TOKEN_TYPE_UNKNOWN] * (max_id + 1)
    for token_id, text in id_to_text.items():
        types[token_id] = classify_token_text(text)
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(types), encoding="utf-8")
    return types
