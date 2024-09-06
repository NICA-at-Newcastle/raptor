from __future__ import annotations
import itertools
import logging
from typing import List, Optional, Literal, TypedDict, overload, Union, Iterable
import dataclasses
import numpy as np
from .embedding_models import BaseEmbeddingModel
from .token_counter import BaseTokenCounter, BytePairTokenCounter
from .tree_structures import Node, Tree
from .utils import (
    distances_from_embeddings,
    get_embeddings,
    get_node_list,
    get_text,
    indices_of_nearest_neighbors_from_distances,
    reverse_mapping,
)
from .storages import BaseStorage

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeRetriever:

    class Limit:
        class TopK(int):
            """TopK(top_k)"""

        class Threshold(float):
            """Threshold(threshold)"""

    class SearchMethod:
        class Flatten(tuple[int, int]):
            """Flatten((top_k, max_tokens))"""

        class Tree(int):
            """Tree(start_layer)"""

    @dataclasses.dataclass
    class Config:

        storage: BaseStorage = dataclasses.field()
        limit: Union[TreeRetriever.Limit.TopK, TreeRetriever.Limit.Threshold] = (
            dataclasses.field()
        )
        embedding_model: BaseEmbeddingModel = dataclasses.field()
        token_counter: BaseTokenCounter = dataclasses.field(
            default=BytePairTokenCounter()
        )
        max_iterations: int = dataclasses.field(default=100)
        start_layer: Optional[int] = dataclasses.field(default=0)

        def __post_init__(self):

            if self.start_layer is not None and self.start_layer < 0:
                raise ValueError("start_layer must be >= 0")

        def log_config(self):
            return f"""
            TreeRetriever.Config:
                TokenCounter: {self.token_counter}
                Embedding Model: {self.embedding_model}
                Start Layer: {self.start_layer}
                Limit: {self.limit}
            """

    def __init__(self, config: Config) -> None:

        self.storage = config.storage
        self.token_counter = config.token_counter
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

    def _create_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.create_embedding(text)

    def retrieve_information_collapse_tree(
        self,
        query: str,
    ) -> Iterable[Node]:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

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
    ) -> tuple[list[Node], str]:
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

        context = get_text(selected_nodes)
        return selected_nodes, context

    class LayerInformation(TypedDict):
        node_index: int
        layer_number: int

    @overload
    def retrieve(
        self,
        query: str,
        search_method: Union[SearchMethod.Flatten, SearchMethod.Tree],
        return_layer_information: Literal[False] = False,
    ) -> str:
        pass

    @overload
    def retrieve(
        self,
        query: str,
        search_method: Union[SearchMethod.Flatten, SearchMethod.Tree],
        return_layer_information: Literal[True] = True,
    ) -> tuple[str, list[LayerInformation]]:
        pass

    def retrieve(
        self,
        query: str,
        search_method: Union[SearchMethod.Flatten, SearchMethod.Tree],
        return_layer_information: bool = False,
    ) -> str | tuple[str, list[LayerInformation]]:
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
                if start_layer < 0:
                    raise ValueError("Start layer must be >= 0")
                selected_nodes, context = self._retrieve_information_tree_search(
                    query,
                    start_layer,
                )
            case self.SearchMethod.Flatten((top_k, max_tokens)):
                if top_k < 1:
                    raise ValueError("top_k must be >= 0")
                if max_tokens < 1:
                    raise ValueError("max_tokens must be an integer and at least 1")
                logging.info(f"Using collapsed_tree")
                selected_nodes, context = self.retrieve_information_collapse_tree(
                    query,
                    top_k,
                    max_tokens,
                )

        if return_layer_information:

            layer_information: list[TreeRetriever.LayerInformation] = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node["index"],
                        "layer_number": self.tree_node_index_to_layer[node["index"]],
                    }
                )

            return context, layer_information

        return context
