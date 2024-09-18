import logging
import random
from abc import abstractmethod
import dataclasses
from typing import List, Optional, override, TypeVar, Generic

import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from .token_counter import BaseTokenCounter, BytePairTokenCounter
from .tree_structures import Node

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# Import necessary methods from other modules

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    assert isinstance(reduced_embeddings, np.ndarray)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    assert isinstance(reduced_embeddings, np.ndarray)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info("Total Clusters: %s", total_clusters)
    return all_local_clusters


_CHUNK = TypeVar("_CHUNK")
_C = TypeVar("_C")


class ClusteringAlgorithm(Generic[_CHUNK]):
    """
    Base clustering algorithm. Clusters the given nodes into a list of clusters.
    """

    @abstractmethod
    def __call__(
        self,
        nodes: List[Node[_CHUNK]],
    ) -> list[list[Node[_CHUNK]]]:
        raise NotImplementedError("Implement in subclass")

    @property
    @abstractmethod
    def reduction_dimension(self) -> int:
        """Returns the reduction dimension of the clustering algorithm."""
        raise NotImplementedError("Implement in subclass")


class RAPTOR_Clustering(ClusteringAlgorithm[_CHUNK]):
    """Base Raptor clustering algorithm"""

    @dataclasses.dataclass
    class Config(Generic[_C]):
        """Raptor clustering algorithm config."""

        max_length_in_cluster: int = dataclasses.field(default=3500)
        token_counter: BaseTokenCounter = dataclasses.field(
            default_factory=BytePairTokenCounter
        )
        reduction_dimension: int = dataclasses.field(default=10)
        threshold: float = dataclasses.field(default=0.1)
        verbose: bool = dataclasses.field(default=False)

    def __init__(self, config: Config[_CHUNK]):
        self.config = config

    @override
    def __call__(
        self,
        nodes: List[Node[_CHUNK]],
    ) -> list[list[Node[_CHUNK]]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node["embedding"] for node in nodes])

        # Perform the clustering
        clusters = perform_clustering(
            embeddings,
            dim=self.config.reduction_dimension,
            threshold=self.config.threshold,
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [self.config.token_counter(node["chunk"]) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > self.config.max_length_in_cluster:
                if self.config.verbose:
                    logging.info(
                        "reclustering cluster with %s nodes", len(cluster_nodes)
                    )
                node_clusters.extend(self(cluster_nodes))
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters

    @property
    @override
    def reduction_dimension(self) -> int:
        return self.config.reduction_dimension
