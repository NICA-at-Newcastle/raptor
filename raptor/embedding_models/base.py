from typing import Generic, TypeVar, Protocol
import numpy as np

_CHUNK_contra = TypeVar("_CHUNK_contra", contravariant=True)


class IEmbeddingModel(Protocol, Generic[_CHUNK_contra]):
    """Defines the methods required for a raptor embedding model."""

    def create_embedding(self, chunk: _CHUNK_contra) -> np.ndarray: ...

    def create_text_embedding(self, text: str) -> np.ndarray: ...

    @property
    def slug(self) -> str: ...
