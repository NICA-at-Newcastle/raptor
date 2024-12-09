from typing import TypeVar, Generic, List
from abc import abstractmethod

from raptor.tree_structures import Node

_CHUNK = TypeVar("_CHUNK")


class ClusteringAlgorithm(Generic[_CHUNK]):
    """
    Base clustering algorithm. Clusters the given nodes into a list of clusters.
    """

    @abstractmethod
    def __call__(
        self,
        nodes: List[Node[_CHUNK]],
    ) -> tuple[list[list[Node[_CHUNK]]], list[Node[_CHUNK]]]:
        """
        Returns
        -------
        tuple[list[list[Node[_CHUNK]]], list[Node[_CHUNK]]]
            A tuple containing the clusters and the outliers.
        """
        raise NotImplementedError("Implement in subclass")
