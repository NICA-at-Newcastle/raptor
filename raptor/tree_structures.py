from typing import TypedDict, TypeVar, Generic
import numpy as np

_C = TypeVar("_C")


class Node(TypedDict, Generic[_C]):
    """
    Represents a node in the hierarchical tree structure.
    """

    chunk: _C
    index: int
    children: set[int]
    embedding: np.ndarray


Nodes = dict[int, Node[_C]]


class Tree(TypedDict, Generic[_C]):
    """
    Represents the entire hierarchical tree structure.
    """

    all_nodes: Nodes[_C]
    root_nodes: Nodes[_C]
    leaf_nodes: Nodes[_C]
    num_layers: int
    layer_to_nodes: dict[int, list[Node[_C]]]
