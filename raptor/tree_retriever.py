from __future__ import annotations
import logging
from typing import (
    List,
    Optional,
    TypedDict,
    Union,
    Iterable,
    TypeVar,
    Generic,
)
import dataclasses
import numpy as np
from .embedding_models import IEmbeddingModel
from .tree_structures import Node
from .storages import IStorageSearch

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

_CHUNK = TypeVar("_CHUNK")
_C = TypeVar("_C")


logger = logging.getLogger(__name__)


class TreeRetriever(Generic[_CHUNK]):
    """Retrieves nodes from tree using raptor search."""

    class Limit:
        """TreeRetriever limit methods"""

        class TopK(int):
            """TopK(top_k)"""

        class Threshold(float):
            """Threshold(threshold)"""

    class SearchMethod:
        """TreeRetriever search methods"""

        class Flatten:
            """Flatten()"""

        class Tree(tuple[int, bool]):
            """Tree((start_layer, include_outliers))"""

    class StringQuery(str):
        """StringQuery(query)"""

    class ImageQuery(str):
        """ImageQuery(file_path)"""

    Query = Union[StringQuery, ImageQuery]

    @dataclasses.dataclass
    class Config(Generic[_C]):
        """TreeRetriever config"""

        storage: IStorageSearch[_C] = dataclasses.field()
        limit: Union[TreeRetriever.Limit.TopK, TreeRetriever.Limit.Threshold] = (
            dataclasses.field()
        )
        embedding_model: IEmbeddingModel[TreeRetriever.Query] = dataclasses.field()
        max_iterations: int = dataclasses.field(default=100)
        start_layer: Optional[int] = dataclasses.field(default=0)

        def __post_init__(self):

            if self.start_layer is not None and self.start_layer < 0:
                raise ValueError("start_layer must be >= 0")

        def __str__(self) -> str:
            """Returns string formatted config."""
            return f"""
            TreeRetriever.Config:
                Embedding Model: {self.embedding_model}
                Start Layer: {self.start_layer}
                Limit: {self.limit}
            """

    def __init__(
        self,
        config: Config[_CHUNK],
    ) -> None:

        self.storage = config.storage
        self.embedding_model = config.embedding_model
        self.max_iterations = config.max_iterations
        self.limit = config.limit
        match self.limit:
            case self.Limit.TopK(x):
                if x < 1:
                    raise ValueError("TopK limit must specify an int >= 1")
            case self.Limit.Threshold(x):
                if x < 0:
                    raise ValueError("Threshold limit must specify a float > 0")

    def _create_embedding(self, query: Query) -> np.ndarray:
        return self.embedding_model.create_embedding(query)

    def retrieve_information_collapse_tree(
        self,
        query: Query,
    ) -> Iterable[Node]:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self._create_embedding(query)

        return self._retreive_information_from_storage(
            query_embedding,
            None,
            None,
        )

    @staticmethod
    def _select_nodes_by_threshold(
        nodes: Iterable[tuple[Node, float]],
        threshold: float,
    ) -> Iterable[Node]:
        for node, distance in nodes:
            if distance > threshold:
                yield node

    @staticmethod
    def _select_nodes_by_top_k(
        nodes: Iterable[tuple[Node, float]],
        top_k: int,
    ) -> Iterable[Node]:
        for (node, _), _ in zip(nodes, range(top_k)):
            yield node

    def _retreive_information_from_storage(
        self,
        query: np.ndarray,
        /,
        parents: Optional[set[int]],
        layer: Optional[int],
    ) -> Iterable[Node]:

        match self.limit:
            case self.Limit.TopK(x):
                limit = x
            case _:
                limit = None

        found = self.storage.search(
            query,
            parents=parents,
            layer=layer,
            limit=limit,
        )

        match self.limit:
            case self.Limit.TopK(x):
                yield from self._select_nodes_by_top_k(found, x)
            case self.Limit.Threshold(x):
                yield from self._select_nodes_by_threshold(found, x)

    def _retreive_information_from_child_nodes_recurse(
        self,
        query: np.ndarray,
        /,
        parents: List[Node],
        layer: int | None,  # Set to none to include outliers
        iteration_number: int,
    ) -> dict[int, Node]:

        parent_node_indices = {node["index"] for node in parents}

        if iteration_number >= self.max_iterations:
            return {}

        nodes_to_add: dict[int, Node] = {
            node["index"]: node
            for node in self._retreive_information_from_storage(
                query,
                parents=parent_node_indices,
                layer=layer,
            )
        }

        if len(nodes_to_add) == 0 or layer == 0:
            return nodes_to_add

        next_layer = None if layer is None else layer - 1

        return {
            **nodes_to_add,
            **self._retreive_information_from_child_nodes_recurse(
                query,
                parents=list(nodes_to_add.values()),
                layer=next_layer,
                iteration_number=iteration_number + 1,
            ),
        }

    def _retrieve_information_tree_search(
        self,
        query: Query,
        /,
        start_layer: int,
        include_outliers: bool,
    ) -> dict[int, Node[_CHUNK]]:
        """
        Retrieves the most relevant information from the tree based on the query.
        Recursively iterates to query embeddings on child nodes.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self._create_embedding(query)

        selected_nodes: dict[int, Node] = {
            node["index"]: node
            for node in self._retreive_information_from_storage(
                query_embedding,
                parents=None,
                layer=start_layer,
            )
        }

        if len(selected_nodes) == 0:
            logger.warning("No nodes returned from storage at start of tree search. Is the storage empty?")
            return {}

        next_layer = None if include_outliers else start_layer - 1
        child_selected_nodes = self._retreive_information_from_child_nodes_recurse(
            query_embedding,
            parents=list(selected_nodes.values()),
            layer=next_layer,
            iteration_number=0,
        )

        selected_nodes = {
            **selected_nodes,
            **child_selected_nodes
        }
        del child_selected_nodes

        return selected_nodes

    class LayerInformation(TypedDict):
        """Dict containing the node and layer numbers."""

        node_index: int
        layer_number: int

    def retrieve(
        self,
        query: str,
        search_method: Union[SearchMethod.Flatten, SearchMethod.Tree],
    ) -> list[Node[_CHUNK]]:
        """
        Queries the tree and returns the most relevant information.

        Args:
            query (str): The query text.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            collapse_tree (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The result of the query.
        """

        match search_method:
            case self.SearchMethod.Tree((start_layer, include_outliers)):
                selected_nodes = list(
                    self._retrieve_information_tree_search(
                        self.StringQuery(query),
                        start_layer=start_layer,
                        include_outliers=include_outliers,
                    ).values()
                )
            case self.SearchMethod.Flatten():
                selected_nodes = list(
                    self.retrieve_information_collapse_tree(self.StringQuery(query))
                )
            case _:
                raise ValueError("Invalid search_method")

        return selected_nodes
