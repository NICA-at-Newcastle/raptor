from typing import Optional, Iterable, Protocol, Generic, TypeVar
import numpy as np
from ..tree_structures import Node


_CHUNK_contra = TypeVar("_CHUNK_contra", contravariant=False, covariant=False)


class IStorageSearch(Protocol, Generic[_CHUNK_contra]):
    """Defines the methods required for a raptor search."""

    def search(
        self,
        search_vector: np.ndarray,
        parents: Optional[set[int]] = None,
        layer: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[tuple[Node[_CHUNK_contra], float]]: ...


class IStorageSave(Protocol, Generic[_CHUNK_contra]):
    """Defines the methods required for a raptor tree build."""

    def save_node(
        self,
        node: Node[_CHUNK_contra],
        parent: Optional[int] = None,
    ): ...
