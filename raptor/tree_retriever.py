from __future__ import annotations
import itertools
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

        class Tree(int):
            """Tree(start_layer)"""

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

        def log_config(self):
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

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

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
        indices: Optional[set[int]],
        layer: Optional[int],
    ) -> Iterable[Node]:

        match self.limit:
            case self.Limit.TopK(x):
                limit = x
            case _:
                limit = None

        found = self.storage.search(
            query,
            indices=indices,
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
        parent_nodes: List[Node],
        iteration_number: int,
    ) -> Iterable[Node]:

        child_node_indices: set[int] = set().union(
            *(node["children"] for node in parent_nodes)
        )

        nodes_to_add = list(
            self._retreive_information_from_storage(
                query,
                child_node_indices,
                None,
            )
        )

        if iteration_number >= self.max_iterations or len(child_node_indices) == 0:
            return nodes_to_add

        return itertools.chain(
            nodes_to_add,
            self._retreive_information_from_child_nodes_recurse(
                query,
                nodes_to_add,
                iteration_number + 1,
            ),
        )

    def _retrieve_information_tree_search(
        self,
        query: str,
        start_layer: int,
    ) -> Iterable[Node[_CHUNK]]:
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

        selected_nodes = list(
            self._retreive_information_from_storage(
                query_embedding,
                None,
                start_layer,
            )
        )

        child_selected_nodes = self._retreive_information_from_child_nodes_recurse(
            query_embedding,
            selected_nodes,
            0,
        )

        selected_nodes.extend(child_selected_nodes)
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
            case self.SearchMethod.Tree(start_layer):
                selected_nodes = list(
                    self._retrieve_information_tree_search(
                        query,
                        start_layer,
                    )
                )
            case self.SearchMethod.Flatten():
                logging.info("Using collapsed_tree")
                selected_nodes = list(self.retrieve_information_collapse_tree(query))

        return selected_nodes
