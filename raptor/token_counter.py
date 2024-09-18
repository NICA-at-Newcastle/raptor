from typing import Generic, Protocol, TypeVar
import math


_CHUNK_contra = TypeVar("_CHUNK_contra", contravariant=True)


class BaseTokenCounter(Protocol, Generic[_CHUNK_contra]):
    """Base class for all token counters."""

    def __call__(self, chunk: _CHUNK_contra) -> int: ...


class BytePairTokenCounter(BaseTokenCounter):
    """Token counter which assumes that each token = 2 bytes."""

    def __call__(self, text: str) -> int:
        return math.ceil(len(text.encode()) / 2)
