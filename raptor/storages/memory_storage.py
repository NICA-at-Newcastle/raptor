"""Simple in memory storage for testing."""

from __future__ import annotations
import logging
from typing import Iterable, TypeVar, Generic, Optional, TypedDict
import numpy as np
from ..tree_structures import Node
from ..utils import (
    get_embeddings,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)

_CHUNK = TypeVar("_CHUNK")
_C = TypeVar("_C")


class MemoryStorage(Generic[_CHUNK]):

    class DictFormat(TypedDict, Generic[_C]):
        layers: dict[int, list[Node[_C]]]
        index_to_node: dict[int, Node[_C]]
        parent_to_children: dict[int | None, set[int]]
        child_to_parent: dict[int, int | None]

    def __init__(
        self,
        load_from: DictFormat[_CHUNK] | None = None,
    ):
        self._index_to_node: dict[int, Node[_CHUNK]] = {}
        self._layers: dict[int, list[Node[_CHUNK]]] = {}
        self._parent_to_children: dict[int | None, set[int]] = dict()
        self._child_to_parent: dict[int, int | None] = dict()
        if load_from:
            self.load_dict(load_from)

    def to_dict(self) -> DictFormat[_CHUNK]:
        return {
            "layers": self._layers,
            "parent_to_children": self._parent_to_children,
            "child_to_parent": self._child_to_parent,
            "index_to_node": self._index_to_node,
        }

    def load_dict(self, dict_format: DictFormat):
        self._layers = {
            int(layer_id): nodes for layer_id, nodes in dict_format["layers"].items()
        }
        self._index_to_node = {
            int(node_id): node for node_id, node in dict_format["index_to_node"].items()
        }
        self._parent_to_children = dict_format["parent_to_children"]
        self._child_to_parent = dict_format["child_to_parent"]

    def search(
        self,
        search_vector: np.ndarray,
        parents: set[int] | None = None,
        layer: int | None = None,
        limit: int | None = None,
    ) -> Iterable[tuple[Node, float]]:

        if layer is not None and layer >= 0:
            nodes_to_search: set[int] = {node["index"] for node in self._layers[layer]}
        elif layer is not None and layer < 0:
            nodes_to_search: set[int] = self._parent_to_children.get(None, set())
        else:
            nodes_to_search: set[int] = {
                node["index"] for layer in self._layers.values() for node in layer
            }

        if parents is not None:
            child_nodes = set(
                [
                    child
                    for parent in parents
                    for child in self._parent_to_children.get(parent, set())
                ]
            )
            nodes_to_search = nodes_to_search.intersection(child_nodes)

        node_list = [self._index_to_node[i] for i in nodes_to_search]

        embeddings = get_embeddings(node_list)
        distances = distances_from_embeddings(search_vector, embeddings)
        found_indices = indices_of_nearest_neighbors_from_distances(distances)

        if limit:
            best_indices = found_indices[:limit]
        else:
            best_indices = found_indices

        return ((node_list[i], distances[i]) for i in best_indices)

    def save_node(
        self,
        node: Node[_CHUNK],
        parent: Optional[int] = None,
    ):
        logging.info("Saving node %s with parent %s", node, parent)
        self._index_to_node[node["index"]] = node
        layer = node["layer"]
        if layer not in self._layers:
            self._layers[layer] = []
        self._layers[layer].append(node)

        if parent not in self._parent_to_children:
            self._parent_to_children[parent] = set()

        self._parent_to_children[parent].add(node["index"])
        self._child_to_parent[node["index"]] = parent
