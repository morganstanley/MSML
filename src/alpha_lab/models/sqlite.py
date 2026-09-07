"""Model-side declaration of SQLite-indexed columns.

A :class:`SQLiteModel` marks indexed fields with a :class:`SQLite` marker in
their ``Annotated`` metadata; the orthogonal ``fulltext`` (FTS) and ``embedded``
(vector-embedded) index roles are flags set on that one marker. Each marked
field is resolved once, at model-class definition, into a :class:`SQLiteFieldInfo`:

- ``sqlite_type`` — the SQLite storage type (``str`` for multi-value columns,
  else the scalar type normalized to its SQL base).
- ``multi_value`` — whether the field is a membership (list/tuple) column.
- ``fulltext`` / ``embedded`` — the marker's index roles.

Annotation spellings::

    summary: Annotated[str, SQLite(fulltext=True)]                 # column + FTS
    content: Annotated[str, SQLite(fulltext=True, embedded=True)]  # column + FTS + embedded
    kind:    Annotated[MemoryKind, SQLite]                         # column
    tags:    Annotated[tuple[str, ...], SQLite]                    # multi-value column
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import InitVar, dataclass
from functools import cached_property
from types import UnionType
from typing import Any, ClassVar, Union, get_args, get_origin

from pydantic.fields import FieldInfo

from alpha_lab.models.utils import AbstractModel


def _strip_none(annotation: Any) -> Any:
    """Strip ``None`` out of an optional union (leave non-union types untouched).

    ``T | None`` collapses to ``T``; a wider optional such as ``int | float | None``
    collapses to the remaining union ``int | float``.
    """
    if get_origin(annotation) in (Union, UnionType):
        non_none = tuple(a for a in get_args(annotation) if a is not type(None))
        if len(non_none) == 1:
            return non_none[0]
        if non_none:
            return Union[non_none]
    return annotation


@dataclass(frozen=True)
class SQLite:
    embedded: bool = False
    fulltext: bool = False


@dataclass(frozen=True)
class SQLiteFieldInfo:
    base: SQLite
    info: FieldInfo
    eager: InitVar[bool] = True

    def __post_init__(self, eager: bool):
        if eager:
            self.multi_value
            self.sqlite_type

    @property
    def embedded(self) -> bool:
        return self.base.embedded

    @property
    def fulltext(self) -> bool:
        return self.base.fulltext
    
    @cached_property
    def multi_value(self) -> bool:
        """Whether this is a membership (list/tuple) column."""
        return get_origin(_strip_none(self.info.annotation)) in (list, tuple)
    
    @cached_property
    def sqlite_type(self) -> type:
        """SQL storage type: ``str`` for membership columns, else the scalar type
        normalized to its SQL base. Raises for unsupported types."""
        annotation = _strip_none(self.info.annotation)
        if self.multi_value:
            if not all(
                arg is Ellipsis or (isinstance(arg, type) and issubclass(arg, str))
                for arg in get_args(annotation)
            ):
                msg = f"list/tuple columns must hold str, not {annotation}"
                raise TypeError(msg)
            return str
        if not (isinstance(annotation, type) and issubclass(annotation, (str, int, float, bool))):
            msg = f"unsupported column type {annotation!r} (use a scalar or a tuple/list of str)"
            raise TypeError(msg)

        return next((base for base in (bool, int, float, str) if issubclass(annotation, base)))

 
class SQLiteModel(AbstractModel):
    __abstract_model__ = True
    _sqlite_fields: ClassVar[dict[str, SQLiteFieldInfo]]

    @classmethod
    def sqlite_fields(cls) -> Iterator[tuple[str, SQLiteFieldInfo]]:
        """Yield ``(field_name, resolved column)`` for each SQLite-annotated field."""
        yield from cls._sqlite_fields.items()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        sqlite_fields: dict[str, SQLiteFieldInfo] = {}
        for name, info in cls.model_fields.items():
            markers = [m for m in info.metadata if m is SQLite or isinstance(m, SQLite)]
            if not markers:
                continue
            if len(markers) > 1:
                msg = f"field `{name}` carries {len(markers)} SQLite markers; expected one"
                raise TypeError(msg)
            
            marker = SQLite() if markers[0] is SQLite else markers[0]
            sqlite_fields[name] = SQLiteFieldInfo(base=marker, info=info)

        if not cls.is_abstract() and not sqlite_fields:
            msg = "a SQLiteModel must declare at least one SQLite-annotated field"
            raise TypeError(msg)

        cls._sqlite_fields = sqlite_fields

    def get_embedded_text(self) -> str | None:
        """Assemble the text embedded for this record.

        Returns:
            ``"<field>: <value>"`` lines (one per embedded field, blank-line
            joined), or ``None`` when the model declares no embedded fields.
        """
        parts = []
        for name, column in self.sqlite_fields():
            if not column.embedded:
                continue

            cell = getattr(self, name)
            body = (", ".join(cell or ()) if column.multi_value else str(cell or "")).strip()
            parts.append(f"{name}: {body}")
        
        if not parts:
            return None

        return "\n\n".join(parts)
