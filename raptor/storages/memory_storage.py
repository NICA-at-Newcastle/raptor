from typing import Iterable
import numpy as np
from ..tree_structures import Node, Tree, Nodes
from .base import BaseStorage
from ..utils import (
    get_node_list,
    get_embeddings,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)


class MemoryStorage(BaseStorage):

    def __init__(
        self,
        tree: Tree,
        embedding_model_key: str,
    ):
        self.tree = tree
        self.embedding_model_key = embedding_model_key

    def search(
        self,
        search_vector: np.ndarray,
        indices: set[int] | None = None,
        layer: int | None = None,
        limit: int | None = None,
    ) -> Iterable[tuple[Node, float]]:

        if layer is not None:
            nodes_to_search: Nodes = {
                node["index"]: node for node in self.tree["layer_to_nodes"][layer]
            }
        else:
            nodes_to_search: Nodes = self.tree["all_nodes"]

        node_list = get_node_list(nodes_to_search, indices)

        embeddings = get_embeddings(node_list, self.embedding_model_key)
        distances = distances_from_embeddings(search_vector, embeddings)
        found_indices = indices_of_nearest_neighbors_from_distances(distances)

        if limit:
            best_indices = found_indices[:limit]
        else:
            best_indices = found_indices

        return ((node_list[i], distances[i]) for i in best_indices)
