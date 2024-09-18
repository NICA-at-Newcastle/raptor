"""Tree builder"""
import copy
import logging
import dataclasses
from typing import Literal, TypeVar, Dict, List, Optional, Set, Tuple, Generic
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor, as_completed

from .embedding_models import IEmbeddingModel
from .summarization_models import ISummarizationModel
from .storages import IStorageSave
from .tree_structures import Node, Tree
from .token_counter import BaseTokenCounter, BytePairTokenCounter
from .utils import (
    distances_from_embeddings,
    get_embeddings,
    indices_of_nearest_neighbors_from_distances,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

_CHUNK = TypeVar("_CHUNK")
_C = TypeVar("_C")

class TreeBuilder(ABC, Generic[_CHUNK]):
    """
    The TreeBuilder class is responsible for building a hierarchical text abstraction
    structure, known as a "tree," using summarization models and
    embedding models.
    """

    @dataclasses.dataclass
    class Config(Generic[_C]):
        """TreeBuilder config"""

        summarization_model: ISummarizationModel[_C] = dataclasses.field()
        embedding_model: IEmbeddingModel[_C] = dataclasses.field()
        storage: IStorageSave[_C] = dataclasses.field()
        cluster_embedding_model: int = dataclasses.field(default=0)
        token_counter: BaseTokenCounter = dataclasses.field(
            default=BytePairTokenCounter()
        )

        max_tokens: int = dataclasses.field(default=100)
        num_layers: int = dataclasses.field(default=5)
        threshold: float = dataclasses.field(default=0.5)
        top_k: int = dataclasses.field(default=5)
        selection_mode: Literal["top_k", "threshold"] = dataclasses.field(
            default="top_k"
        )
        summarization_length: int = dataclasses.field(default=100)

        def __post_init__(self):
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be at least 1")

            if self.num_layers < 1:
                raise ValueError("num_layers must be at least 1")

            if not (0 <= self.threshold <= 1):
                raise ValueError("threshold must be between 0 and 1")

            if self.top_k < 1:
                raise ValueError("top_k must be at least 1")

            if self.summarization_length < 1:
                raise ValueError("summarization length must be at least 1")

        def log_config(self) -> str:
            """Return a formatted string of the config."""
            return f"""
            TreeBuilderConfig:
                Max Tokens: {self.max_tokens}
                Num Layers: {self.num_layers}
                Threshold: {self.threshold}
                Top K: {self.top_k}
                Selection Mode: {self.selection_mode}
                Summarization Length: {self.summarization_length}
                Summarization Model: {self.summarization_model}
                Embedding Model: {self.embedding_model}
                Cluster Embedding Model: {self.cluster_embedding_model}
            """

    def __init__(self, config: Config[_CHUNK]) -> None:
        """Initializes the tokenizer, maximum tokens, number of layers, top-k value, threshold, and selection mode."""

        self.config: TreeBuilder.Config[_CHUNK] = config

        logging.info(
            "Successfully initialized TreeBuilder with Config %s", config.log_config()
        )

    def create_node(
        self,
        index: int,
        chunk: _CHUNK,
        layer: int,
        children_indices: Optional[Set[int]] = None,
        is_root: bool = False,
    ) -> Tuple[int, Node]:
        """Creates a new node with the given index, text, and (optionally) children indices.

        Args:
            index (int): The index of the new node.
            text (str): The text associated with the new node.
            children_indices (Optional[Set[int]]): A set of indices representing the children of the new node.
                If not provided, an empty set will be used.

        Returns:
            Tuple[int, Node]: A tuple containing the index and the newly created node.
        """
        if children_indices is None:
            children_indices = set()

        embedding = self.config.embedding_model.create_embedding(chunk)

        node = self.config.storage.create_node(
            chunk,
            index,
            layer,
            embedding,
            children_indices,
            is_root,
        )
        return (index, node)

    def summarize(self, context: list[_CHUNK], max_tokens=150) -> _CHUNK:
        """
        Generates a summary of the input context using the specified summarization model.

        Args:
            context (str, optional): The context to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.o

        Returns:
            str: The generated summary.
        """
        return self.config.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node: Node, list_nodes: list[Node]) -> List[Node]:
        """
        Retrieves the top-k most relevant nodes to the current node from the list of nodes
        based on cosine distance in the embedding space.

        Args:
            current_node (Node): The current node.
            list_nodes (List[Node]): The list of nodes.

        Returns:
            List[Node]: The top-k most relevant nodes.
        """
        embeddings = get_embeddings(list_nodes)
        distances = distances_from_embeddings(
            current_node["embedding"], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        match self.config.selection_mode:
            case "threshold":
                best_indices = [
                    index
                    for index in indices
                    if distances[index] > self.config.threshold
                ]
            case "top_k":
                best_indices = indices[: self.config.top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]

        return nodes_to_add

    def multithreaded_create_leaf_nodes(self, chunks: List[_CHUNK]) -> Dict[int, Node]:
        """Creates leaf nodes using multithreading from the given list of text chunks.

        Args:
            chunks (List[str]): A list of text chunks to be turned into leaf nodes.

        Returns:
            Dict[int, Node]: A dictionary mapping node indices to the corresponding leaf nodes.
        """
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text, 0): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_chunks(
        self,
        chunks: list[_CHUNK],
        use_multithreading: bool = True
    ) -> Tree:
        """Builds a tree using the pre-computed chunks."""
        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks):
                _, node = self.create_node(index, text, 0)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info("Created %s Leaf Embeddings", len(leaf_nodes))

        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        root_nodes_list = root_nodes.values()
        summarized_text = self.summarize(
            context=[node["chunk"] for node in root_nodes_list],
            max_tokens=self.config.summarization_length,
        )
        next_node_index = max(root_nodes.keys()) + 1
        _, new_root_node = self.create_node(
            next_node_index,
            summarized_text,
            len(layer_to_nodes),
            {node["index"] for node in root_nodes_list},
            True,
        )
        all_nodes.update({next_node_index: new_root_node})
        layer_to_nodes[len(layer_to_nodes)] = [new_root_node]

        return {
            "all_nodes": all_nodes,
            "root_nodes": root_nodes,
            "leaf_nodes": leaf_nodes,
            "num_layers": self.config.num_layers,
            "layer_to_nodes": layer_to_nodes,
        }


    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
        of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.

        Args:
            current_level_nodes (Dict[int, Node]): The current set of nodes.
            all_tree_nodes (Dict[int, Node]): The dictionary of all nodes.
            use_multithreading (bool): Whether to use multithreading to speed up the process.

        Returns:
            Dict[int, Node]: The final set of root nodes.
        """
        raise NotImplementedError("Implement in subclass")
