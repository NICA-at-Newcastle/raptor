from typing import override, Dict, List, TypeVar
import logging
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder
from .utils import (
    get_node_list,
)
from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


_CHUNK = TypeVar("_CHUNK")
_C = TypeVar("_C")


class ClusterTreeBuilder(TreeBuilder[_CHUNK]):
    """Tree builder which uses clustering."""

    @dataclasses.dataclass
    class Config(TreeBuilder.Config[_C]):
        """ClusterTreeBuilder config"""

        clustering_algorithm: ClusteringAlgorithm = dataclasses.field(
            default_factory=lambda: RAPTOR_Clustering(RAPTOR_Clustering.Config())
        )

        def log_config(self):
            base_summary = super().log_config()
            cluster_tree_summary = f"""
            Clustering Algorithm: {self.clustering_algorithm}
            """
            return base_summary + cluster_tree_summary

    def __init__(self, config: Config[_CHUNK]) -> None:
        super().__init__(config)

        self.clustering_algorithm = config.clustering_algorithm

        logging.info(
            "Successfully initialized ClusterTreeBuilder with Config %s",
            config.log_config(),
        )

    @override
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster: list[Node],
            layer: int,
            new_level_nodes: dict[int, Node],
            next_node_index: int,
            lock: Lock,
        ):

            summarized_text = self.summarize(
                context=[node["chunk"] for node in cluster],
            )

            _, new_parent_node = self.create_node(
                next_node_index,
                summarized_text,
                layer,
                {node["index"] for node in cluster},
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.config.num_layers):
            true_layer_number = layer + 1
            new_level_nodes = {}

            logging.info(f"Constructing Layer {true_layer_number}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if (
                len(node_list_current_layer)
                <= self.clustering_algorithm.reduction_dimension + 1
            ):
                self.num_layers = true_layer_number
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {true_layer_number}"
                )
                break

            clusters = self.clustering_algorithm(node_list_current_layer)

            lock = Lock()

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            true_layer_number,
                            new_level_nodes,
                            next_node_index,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        true_layer_number,
                        new_level_nodes,
                        next_node_index,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[true_layer_number] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return current_level_nodes
