from typing import ClassVar

from pydantic import BaseModel

class AbstractModel(BaseModel):
    """Base class for abstract BaseModel subclasses."""

    __abstract_model__: ClassVar[bool] = False

    @classmethod
    def is_abstract(cls) -> bool:
        return cls.__dict__.get("__abstract_model__", False)
