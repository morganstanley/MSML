"""Shared Alpha Lab constants."""

from __future__ import annotations

from enum import StrEnum, auto


class DefaultType:
    """Sentinel class whose sole instance DEFAULT denotes the default value
    of an argument that can be None."""


DEFAULT = DefaultType()


class MemoryKind(StrEnum):
    """Supported memory information shapes.

    - ``fact``: a definition, statement or detail that is known to be true
    - ``idea``: a hypothesis, prediction, or direction to consider
    - ``unknown``: explicit declaration that something is not known
    - ``pattern``: information about observed data that has been largely confirmed
    - ``constraint``: information about a requirement
    """

    FACT = auto()
    IDEA = auto()
    UNKNOWN = auto()
    PATTERN = auto()
    CONSTRAINT = auto()


class Phase(StrEnum):
    """Supported pipeline phases."""

    PHASE0 = auto()
    PHASE1 = auto()
    PHASE2 = auto()
    PHASE3 = auto()
