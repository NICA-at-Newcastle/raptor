"""Simple in memory storage for testing."""

from __future__ import annotations
from typing import Iterable, TypeVar, Generic, Optional, TypedDict
import numpy as np
from ..tree_structures import Node, Nodes
from ..utils import (
    get_node_list,
    get_embeddings,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)

_CHUNK = TypeVar("_CHUNK")
_C = TypeVar("_C")


class MemoryStorage(Generic[_CHUNK]):

    class DictFormat(TypedDict, Generic[_C]):
        layers: dict[int, list[Node[_C]]]
        root_nodes: list[Node[_C]]

    def __init__(
        self,
        load_from: DictFormat[_CHUNK] | None = None,
    ):
        self._layers: dict[int, list[Node[_CHUNK]]] = {}
        self._root_nodes = []
        if load_from:
            self.load_dict(load_from)

    def to_dict(self) -> DictFormat[_CHUNK]:
        return {
            "layers": self._layers,
            "root_nodes": self._root_nodes,
        }

    def load_dict(self, dict_format: DictFormat):
        self._layers = {
            int(layer_id): nodes for layer_id, nodes in dict_format["layers"].items()
        }
        self._root_nodes = dict_format["root_nodes"]

    def search(
        self,
        search_vector: np.ndarray,
        indices: set[int] | None = None,
        layer: int | None = None,
        limit: int | None = None,
    ) -> Iterable[tuple[Node, float]]:

        if layer is not None and layer >= 0:
            nodes_to_search: Nodes = {
                node["index"]: node for node in self._layers[layer]
            }
        elif layer is not None and layer < 0:
            nodes_to_search: Nodes = {node["index"]: node for node in self._root_nodes}
        else:
            nodes_to_search: Nodes = {
                node["index"]: node for layer in self._layers.values() for node in layer
            }

        node_list = get_node_list(nodes_to_search, indices)

        embeddings = get_embeddings(node_list)
        distances = distances_from_embeddings(search_vector, embeddings)
        found_indices = indices_of_nearest_neighbors_from_distances(distances)

        if limit:
            best_indices = found_indices[:limit]
        else:
            best_indices = found_indices

        return ((node_list[i], distances[i]) for i in best_indices)

    def create_node(
        self,
        chunk: _CHUNK,
        index: int,
        layer: int,
        embedding: np.ndarray,
        children_indices: Optional[set[int]] = None,
        is_root: bool = False,
    ) -> Node[_CHUNK]:
        node: Node[_CHUNK] = {
            "index": index,
            "chunk": chunk,
            "embedding": embedding,
            "children": children_indices or set(),
        }
        if layer not in self._layers:
            self._layers[layer] = []
        self._layers[layer].append(node)

        if is_root:
            self._root_nodes.append(node)

        return node
