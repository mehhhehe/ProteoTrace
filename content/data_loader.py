"""Data loading and graph utilities for the ogbn‑proteins project.

This module provides a minimal interface to load the ogbn‑proteins
dataset, construct node features by aggregating incident edge
attributes and expose helper methods to support the interactive
dashboard.  It mirrors the design of the TruthTrace `data_loader` by
wrapping the underlying OGB and PyTorch Geometric logic in a clean
Python class.

Key features:

* **Lazy loading** – the dataset is only downloaded or read from disk
  when explicitly requested.  If the files exist under
  ``root/ogbn-proteins/raw``, no network access is attempted.
* **Node feature construction** – the eight‑dimensional edge features
  of the dataset are aggregated to create a per‑node feature vector.
  Optionally, node degree and log‑degree are appended.
* **Neighbourhood extraction** – given a node ID, return a small
  induced subgraph of its neighbours for D3 visualisation.
* **Prediction management** – load precomputed predictions from
  `.npy` files for the baseline and GNN models.

To keep memory usage low on CPU‑only machines, the aggregation and
graph construction use NumPy and NetworkX rather than PyTorch until
model training is invoked.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

LOGGER = logging.getLogger(__name__)


def load_raw_ogbn_proteins(root: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """Load the ogbn‑proteins dataset via OGB.

    This function attempts to use the OGB downloader.  If the raw files
    already exist under ``root/ogbn-proteins/raw``, the OGB
    downloader will reuse them.  If you encounter issues with the
    official download (e.g. 404 errors), manually place the raw files
    from OpenDataLab into this directory (see README).

    Parameters
    ----------
    root : str
        Directory in which the dataset will be cached.  The expected
        structure is ``root/ogbn-proteins/raw`` containing ``edge_index.npy``,
        ``edge_feat.npy``, ``node_label.npy`` and other files.

    Returns
    -------
    graph : dict
        Dictionary with keys ``edge_index`` (shape [2, num_edges]),
        ``edge_feat`` (shape [num_edges, 8]) and ``num_nodes``.
    labels : np.ndarray
        Array of shape [num_nodes, 112] containing binary labels.
    split_idx : dict
        Dictionary with keys ``'train'``, ``'valid'``, ``'test'`` mapping
        to NumPy arrays of node indices for each split.
    """
    LOGGER.info("Loading ogbn-proteins from %s", root)
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as e:
        raise ImportError(
            "The ogb package is required to load ogbn-proteins. Please install it via pip."
        ) from e

    dataset = NodePropPredDataset(name="ogbn-proteins", root=root)
    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    return graph, labels, split_idx


def aggregate_edge_features(graph: Dict[str, np.ndarray], method: str = "mean",
                            add_degree: bool = True) -> np.ndarray:
    """Aggregate 8‑dimensional edge features to create node features.

    For each node, we aggregate the features of all incident edges.
    When ``method='mean'``, the arithmetic mean of edge features is
    computed; when ``method='sum'``, a simple sum is used.  Optionally
    append the node degree and log(1+degree) as additional features.

    Parameters
    ----------
    graph : dict
        Raw graph dictionary from :func:`load_raw_ogbn_proteins`.
    method : {"mean", "sum"}, default="mean"
        Aggregation operation.
    add_degree : bool, default=True
        Whether to append degree and log‑degree features.

    Returns
    -------
    node_features : np.ndarray
        Array of shape [num_nodes, 8 (+2)] containing aggregated
        features.  The optional last two columns are degree and
        log‑degree when ``add_degree`` is True.
    """
    num_nodes = graph['num_nodes']
    edge_index = graph['edge_index']
    edge_feat = graph['edge_feat']

    # Initialize accumulators
    agg = np.zeros((num_nodes, edge_feat.shape[1]), dtype=np.float32)
    deg = np.zeros(num_nodes, dtype=np.int64)

    src, dst = edge_index
    # Aggregate edge features for both endpoints since the graph is undirected
    for u, v, feat in zip(src, dst, edge_feat):
        agg[u] += feat
        agg[v] += feat
        deg[u] += 1
        deg[v] += 1

    if method == "mean":
        # Avoid division by zero
        deg_nonzero = np.maximum(deg[:, None], 1)
        agg = agg / deg_nonzero
    elif method == "sum":
        pass
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

    if add_degree:
        deg_feat = deg.astype(np.float32)[:, None]
        log_deg = np.log1p(deg_feat)
        node_features = np.concatenate([agg, deg_feat, log_deg], axis=1)
    else:
        node_features = agg
    return node_features


class ProteinsDataset:
    """Wrapper class exposing nodes, graphs and predictions for the dashboard.

    This class encapsulates the raw graph, aggregated node features,
    labels and split indices.  It also holds predictions from the
    baseline and GNN models which can be loaded via
    :meth:`load_predictions`.  Methods like :meth:`list_nodes` and
    :meth:`get_graph_json` are used by the Flask backend.
    """

    def __init__(self, root: str, agg_method: str = "mean", add_degree: bool = True) -> None:
        self.root = os.path.expanduser(root)
        graph, labels, split_idx = load_raw_ogbn_proteins(self.root)
        self.graph = graph
        self.labels = labels
        self.split_idx = split_idx
        self.features = aggregate_edge_features(graph, method=agg_method, add_degree=add_degree)
        self.num_nodes = graph['num_nodes']
        self.baseline_pred: Optional[np.ndarray] = None
        self.gnn_pred: Optional[np.ndarray] = None
        # Build an undirected NetworkX graph for neighbourhood queries
        LOGGER.info("Constructing NetworkX graph for neighbourhood queries...")
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_nodes))
        src, dst = graph['edge_index']
        edges = [(int(u), int(v)) for u, v in zip(src, dst)]
        self.G.add_edges_from(edges)
        LOGGER.info("Graph with %d nodes and %d edges constructed", self.num_nodes, len(edges))

    def load_predictions(self, baseline_path: str, gnn_path: str) -> None:
        """Load precomputed predictions from .npy files.

        Parameters
        ----------
        baseline_path : str
            Path to a NumPy file containing baseline probabilities of
            shape [num_nodes, 112].
        gnn_path : str
            Path to a NumPy file containing GNN probabilities of
            shape [num_nodes, 112].
        """
        self.baseline_pred = np.load(baseline_path)
        self.gnn_pred = np.load(gnn_path)
        LOGGER.info(
            "Loaded baseline predictions from %s and GNN predictions from %s",
            baseline_path, gnn_path
        )

    def list_nodes(self, limit: Optional[int] = None) -> List[int]:
        """Return a list of node indices for populating the dropdown.

        Parameters
        ----------
        limit : int, optional
            Maximum number of nodes to return.  If None (default), all
            nodes are returned.  Limiting the number of nodes can make
            the dropdown more responsive.

        Returns
        -------
        List[int]
            Node indices.
        """
        if limit is None or limit >= self.num_nodes:
            return list(range(self.num_nodes))
        return list(range(limit))

    def get_prediction(self, node_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return baseline and GNN probabilities for a given node.

        Parameters
        ----------
        node_id : int
            Node index.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Tuple of two arrays of shape [112] containing predicted
            probabilities from the baseline and GNN models.  If
            predictions have not been loaded yet, zeros are returned.
        """
        if self.baseline_pred is None or self.gnn_pred is None:
            # Return zeros as placeholders if predictions are missing
            empty = np.zeros(self.labels.shape[1], dtype=np.float32)
            return empty, empty
        return self.baseline_pred[node_id], self.gnn_pred[node_id]

    def get_graph_json(self, node_id: int, hop: int = 1, max_nodes: int = 50) -> Dict[str, List[Dict[str, int]]]:
        """Return a small subgraph around a node in JSON format.

        The subgraph is induced by breadth‑first search up to ``hop``
        steps from the source node.  To prevent overly large graphs in
        the dashboard, the number of nodes is capped at ``max_nodes``.

        Parameters
        ----------
        node_id : int
            The central node for which to extract the neighbourhood.
        hop : int, default=1
            Number of hops to include.  ``hop=1`` includes immediate
            neighbours; ``hop=2`` includes neighbours of neighbours.
        max_nodes : int, default=50
            Maximum number of nodes in the returned subgraph.

        Returns
        -------
        dict
            A dictionary with two keys: ``'nodes'`` (a list of node
            dictionaries) and ``'links'`` (a list of edge dictionaries).
            Each node dictionary has ``id`` and ``group`` fields (group
            is 0 for the root and 1 for neighbours).  Each edge
            dictionary has ``source`` and ``target``.
        """
        # BFS to collect up to max_nodes nodes within hop hops
        visited = {node_id}
        frontier = {node_id}
        next_frontier = set()
        nodes_collected = {node_id}
        for _ in range(hop):
            for u in frontier:
                for v in self.G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
                        nodes_collected.add(v)
                        if len(nodes_collected) >= max_nodes:
                            break
                if len(nodes_collected) >= max_nodes:
                    break
            frontier = next_frontier
            next_frontier = set()
            if len(nodes_collected) >= max_nodes:
                break
        # Build node list
        nodes = []
        for n in nodes_collected:
            group = 0 if n == node_id else 1
            nodes.append({"id": int(n), "group": group})
        # Build edge list (only keep edges within subgraph)
        links = []
        for u in nodes_collected:
            for v in self.G.neighbors(u):
                if v in nodes_collected and u < v:
                    links.append({"source": int(u), "target": int(v)})
        return {"nodes": nodes, "links": links}