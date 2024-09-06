from typing import TypedDict
import numpy as np


class Node(TypedDict):
    """
    Represents a node in the hierarchical tree structure.
    """

    text: str
    index: int
    children: set[int]
    embeddings: dict[str, np.ndarray]


Nodes = dict[int, Node]


class Tree(TypedDict):
    """
    Represents the entire hierarchical tree structure.
    """

    all_nodes: Nodes
    root_nodes: Nodes
    leaf_nodes: Nodes
    num_layers: int
    layer_to_nodes: dict[int, list[Node]]
