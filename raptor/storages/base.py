from abc import ABC, abstractmethod
from typing import Optional, Iterable
import numpy as np
from ..tree_structures import Node


class BaseStorage(ABC):

    @abstractmethod
    def search(
        self,
        search_vector: np.ndarray,
        indices: Optional[set[int]] = None,
        layer: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[tuple[Node, float]]:
        raise NotImplementedError("Implement in subclass")
