from typing import override, Dict, List, TypeVar
import logging
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .tree_builder import TreeBuilder
from .utils import (
    get_node_list,
)
from .clustering_algorithms.base import ClusteringAlgorithm
from .clustering_algorithms.umap_gmm import UMAPGMMClusteringAlgorithm
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
            default_factory=lambda: UMAPGMMClusteringAlgorithm(
                UMAPGMMClusteringAlgorithm.Config()
            )
        )
        target_reduction_factor: float = dataclasses.field(default=2)

        def log_config(self):
            base_summary = super().log_config()
            cluster_tree_summary = f"""
            Clustering Algorithm: {self.clustering_algorithm}
            """
            return base_summary + cluster_tree_summary

    def __init__(self, config: Config[_CHUNK]) -> None:
        super().__init__(config)

        self.clustering_algorithm = config.clustering_algorithm
        self.target_reduction_factor = config.target_reduction_factor

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
    ) -> tuple[Dict[int, Node], list[Node]]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)
        all_outliers = []

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
            )

            for node in cluster:
                self.config.storage.save_node(
                    node,
                    next_node_index,
                )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.config.num_layers):
            true_layer_number = layer + 1
            new_level_nodes = {}

            logging.info("Constructing Layer %s", true_layer_number)
            node_list_current_layer = get_node_list(current_level_nodes)

            logging.info("Clustering %s nodes", len(node_list_current_layer))
            clusters, outliers = self.clustering_algorithm(node_list_current_layer)
            logging.info("Got %s clusters", len(clusters))
            all_outliers.extend(outliers)

            reduction_factor = (
                0
                if len(clusters) == 0
                else len(node_list_current_layer) / len(clusters)
            )

            if reduction_factor < self.target_reduction_factor:
                self.num_layers = true_layer_number
                logging.info(
                    "Stopping Layer construction: Reduction factor of %s reached. Total Layers in tree: %s",
                    reduction_factor,
                    true_layer_number,
                )
                break

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

        return current_level_nodes, all_outliers
