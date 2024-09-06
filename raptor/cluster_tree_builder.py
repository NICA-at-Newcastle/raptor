from typing import override, Dict, List
import logging
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder
from .utils import (
    get_node_list,
    get_text,
)
from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeBuilder(TreeBuilder):

    @dataclasses.dataclass
    class Config(TreeBuilder.Config):

        clustering_algorithm: ClusteringAlgorithm = dataclasses.field(
            default_factory=lambda: RAPTOR_Clustering(RAPTOR_Clustering.Config())
        )

        def log_config(self):
            base_summary = super().log_config()
            cluster_tree_summary = f"""
            Clustering Algorithm: {self.clustering_algorithm}
            """
            return base_summary + cluster_tree_summary

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.clustering_algorithm = config.clustering_algorithm

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
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
            new_level_nodes: dict[int, Node],
            next_node_index: int,
            summarization_length: int,
            lock: Lock,
        ):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {self.config.token_counter(node_texts)}, Summarized Text Length: {self.config.token_counter(summarized_text)}"
            )

            _, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node["index"] for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.config.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if (
                len(node_list_current_layer)
                <= self.clustering_algorithm.reduction_dimension + 1
            ):
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm(
                node_list_current_layer,
                self.config.embedding_models[self.config.cluster_embedding_model].slug,
            )

            lock = Lock()

            summarization_length = self.config.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return current_level_nodes
