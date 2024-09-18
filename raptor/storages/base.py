from typing import Optional, Iterable, Protocol, Generic, TypeVar
import numpy as np
from ..tree_structures import Node


_CHUNK_contra = TypeVar("_CHUNK_contra", contravariant=False, covariant=False)


class IStorageSearch(Protocol, Generic[_CHUNK_contra]):
    """Defines the methods required for a raptor search."""

    def search(
        self,
        search_vector: np.ndarray,
        indices: Optional[set[int]] = None,
        layer: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[tuple[Node[_CHUNK_contra], float]]: ...


class IStorageSave(Protocol, Generic[_CHUNK_contra]):
    """Defines the methods required for a raptor tree build."""

    def create_node(
        self,
        chunk: _CHUNK_contra,
        index: int,
        layer: int,
        embedding: np.ndarray,
        children_indices: Optional[set[int]] = None,
        is_root: bool = False,
    ) -> Node[_CHUNK_contra]: ...
