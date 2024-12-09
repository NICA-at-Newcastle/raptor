"""
Raptor clustering algorithm using UMAP and Gaussian Mixture Models (GMM).
"""

import logging
import random
import dataclasses
from typing import List, Optional, override, TypeVar, Generic

import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from raptor.tree_structures import Node

from .base import ClusteringAlgorithm

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# Import necessary methods from other modules

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def _global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    assert len(embeddings) > 0, "Cannot cluster empty embeddings"
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    assert isinstance(reduced_embeddings, np.ndarray)
    return reduced_embeddings


def _local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    assert isinstance(reduced_embeddings, np.ndarray)
    return reduced_embeddings


def _get_optimal_clusters(
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


def _gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = _get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def _perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = _global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = _gmm_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info("Global Clusters: %s", n_global_clusters)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                "Nodes in Global Cluster %s: %s",
                i,
                len(global_cluster_embeddings_),
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = _local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = _gmm_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(
                "Local Clusters in Global Cluster %s: %s",
                i,
                n_local_clusters,
            )

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


class UMAPGMMClusteringAlgorithm(ClusteringAlgorithm[_CHUNK]):
    """Base Raptor clustering algorithm"""

    @dataclasses.dataclass
    class Config(Generic[_C]):
        """Raptor clustering algorithm config."""

        reduction_dimension: int = dataclasses.field(default=10)
        max_cluster_size: int = dataclasses.field(default=10)
        threshold: float = dataclasses.field(default=0.1)
        verbose: bool = dataclasses.field(default=False)

    def __init__(self, config: Config[_CHUNK]):
        self.config = config

    @override
    def __call__(
        self,
        nodes: List[Node[_CHUNK]],
    ) -> tuple[list[list[Node[_CHUNK]]], list[Node[_CHUNK]]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node["embedding"] for node in nodes])

        # Perform the clustering
        clusters = _perform_clustering(
            embeddings,
            dim=self.config.reduction_dimension,
            threshold=self.config.threshold,
        )

        logging.info("UMAP got %s clusters", len(clusters))

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 0:
                continue

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # If the cluster exceeds maximum cluster size, recluster this cluster
            if len(cluster_nodes) > self.config.max_cluster_size:
                if self.config.verbose:
                    logging.info(
                        "reclustering cluster with %s nodes",
                        len(cluster_nodes),
                    )
                reclustered_nodes, reclustered_outliers = self(cluster_nodes)
                node_clusters.extend(reclustered_nodes)
                reclustered_outliers.extend(reclustered_outliers)
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters, []
