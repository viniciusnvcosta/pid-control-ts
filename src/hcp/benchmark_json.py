# ABOUTME: Validates finite runtime JSON and genuine interrupted object prefixes.
# ABOUTME: Implements pure token-state checks without filesystem operations.
"""Strict runtime JSON decoding and incomplete-token recognition."""

from __future__ import annotations

import json
import math


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("runtime JSON number must be finite")
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"invalid runtime JSON constant: {token}")


def decode_runtime_object(text: str) -> dict:
    """Decode a JSON object, rejecting nonfinite numbers at any nesting depth."""
    event = json.loads(text, parse_float=_finite_float, parse_constant=_reject_constant)
    if not isinstance(event, dict):
        raise ValueError("runtime event must be an object")  # noqa: TRY004 - persisted data
    return event


def incomplete_object_prefix(text: str, *, partial_utf8: bool = False) -> bool:
    """Recognize a valid unfinished JSON object, including partial token states.

    Args:
        text: Decoded final line, excluding any partial final UTF-8 code point.
        partial_utf8: Require that the missing code point is within an open string,
            outside an escape, where a non-ASCII character is permitted.
    """
    position = 0

    def peek() -> str:
        return text[position : position + 1]

    def whitespace() -> None:
        nonlocal position
        while peek() and peek() in " \t\r\n":
            position += 1

    def expect(character: str) -> None:
        nonlocal position
        whitespace()
        if not peek():
            raise EOFError("syntax")
        if peek() != character:
            raise ValueError("invalid JSON punctuation")
        position += 1

    def string() -> None:
        nonlocal position
        expect('"')
        while True:
            character = peek()
            if not character:
                raise EOFError("string")
            position += 1
            if character == '"':
                return
            if ord(character) < 32:
                raise ValueError("invalid string control character")
            if character == "\\":
                escape = peek()
                if not escape:
                    raise EOFError("escape")
                position += 1
                if escape == "u":
                    for _ in range(4):
                        digit = peek()
                        if not digit:
                            raise EOFError("unicode")
                        if digit not in "0123456789abcdefABCDEF":
                            raise ValueError("invalid Unicode escape")
                        position += 1
                elif escape not in '"\\/bfnrt':
                    raise ValueError("invalid string escape")

    def digits() -> None:
        nonlocal position
        if not peek():
            raise EOFError("number")
        if peek() not in "0123456789":
            raise ValueError("invalid number digit")
        while peek() and peek() in "0123456789":
            position += 1

    def number() -> None:
        nonlocal position
        start = position
        if peek() == "-":
            position += 1
        if not peek():
            raise EOFError("number")
        if peek() == "0":
            position += 1
        else:
            digits()
        if peek() == ".":
            position += 1
            digits()
        if peek() and peek() in "eE":
            position += 1
            if peek() and peek() in "+-":
                position += 1
            digits()
        token = text[start:position]
        if any(character in token for character in ".eE"):
            _finite_float(token)

    def value() -> None:
        nonlocal position
        whitespace()
        character = peek()
        if not character:
            raise EOFError("value")
        if character == '"':
            string()
        elif character in "{[":
            container(character)
        elif character in "-0123456789":
            number()
        else:
            literal = {"t": "true", "f": "false", "n": "null"}.get(character)
            if literal is None:
                raise ValueError("invalid JSON value")
            for expected in literal:
                if not peek():
                    raise EOFError("literal")
                if peek() != expected:
                    raise ValueError("invalid JSON literal")
                position += 1

    def container(opening: str) -> None:
        nonlocal position
        expect(opening)
        closing = "}" if opening == "{" else "]"
        whitespace()
        if peek() == closing:
            position += 1
            return
        while True:
            if opening == "{":
                string()
                expect(":")
            value()
            whitespace()
            if peek() == closing:
                position += 1
                return
            expect(",")

    try:
        container("{")
        whitespace()
        if peek():
            raise ValueError("trailing JSON data")
    except EOFError as exc:
        return not partial_utf8 or str(exc) == "string"
    except (ValueError, RecursionError):
        return False
    return False
