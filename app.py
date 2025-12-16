"""Flask web application for the ogbn‑proteins dashboard.

This application provides an interactive interface to explore
predictions from both a simple baseline model and a GAT model on the
ogbn‑proteins dataset.  Users can select a protein node from a
dropdown, inspect the top predicted functions (labels) from each
model, and visualise a small neighbourhood of the node.  The layout
and structure mirror the TruthTrace dashboard but are adapted for
protein function prediction.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, render_template, jsonify

from data_loader import ProteinsDataset


def load_prediction_arrays(model_dir: str, dataset: ProteinsDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct full prediction matrices from split files.

    The training script writes predictions separately for train, val
    and test splits.  This helper reads those files (if present) and
    merges them into arrays of shape [num_nodes, 112] according to
    ``dataset.split_idx``.

    Parameters
    ----------
    model_dir : str
        Directory containing prediction files.
    dataset : ProteinsDataset
        Dataset object providing ``split_idx`` and ``num_nodes``.

    Returns
    -------
    baseline_pred, gnn_pred : np.ndarray
        Matrices of shape [num_nodes, 112] containing probabilities
        from the baseline and GAT models.  If files are missing,
        arrays of zeros are returned.
    """
    num_nodes = dataset.num_nodes
    num_labels = dataset.labels.shape[1]
    # Initialise with zeros
    baseline_pred = np.zeros((num_nodes, num_labels), dtype=np.float32)
    gnn_pred = np.zeros((num_nodes, num_labels), dtype=np.float32)
    # Load baseline probabilities
    base_train_path = os.path.join(model_dir, "baseline_probs_train.npy")
    base_val_path = os.path.join(model_dir, "baseline_probs_val.npy")
    base_test_path = os.path.join(model_dir, "baseline_probs_test.npy")
    if os.path.exists(base_train_path):
        train_probs = np.load(base_train_path)
        baseline_pred[dataset.split_idx["train"]] = train_probs
    if os.path.exists(base_val_path):
        val_probs = np.load(base_val_path)
        baseline_pred[dataset.split_idx["valid"]] = val_probs
    if os.path.exists(base_test_path):
        test_probs = np.load(base_test_path)
        baseline_pred[dataset.split_idx["test"]] = test_probs
    # Load GNN probabilities
    gnn_train_path = os.path.join(model_dir, "gat_probs_train.npy")
    gnn_val_path = os.path.join(model_dir, "gat_probs_val.npy")
    gnn_test_path = os.path.join(model_dir, "gat_probs_test.npy")
    if os.path.exists(gnn_train_path):
        train_probs = np.load(gnn_train_path)
        gnn_pred[dataset.split_idx["train"]] = train_probs
    if os.path.exists(gnn_val_path):
        val_probs = np.load(gnn_val_path)
        gnn_pred[dataset.split_idx["valid"]] = val_probs
    if os.path.exists(gnn_test_path):
        test_probs = np.load(gnn_test_path)
        gnn_pred[dataset.split_idx["test"]] = test_probs
    return baseline_pred, gnn_pred


def create_app() -> Flask:
    app = Flask(__name__)
    # Configuration via environment variables
    root = os.environ.get("PROTEINS_ROOT", "./data")
    agg_method = os.environ.get("PROTEINS_AGG_METHOD", "mean")
    add_degree = os.environ.get("PROTEINS_ADD_DEGREE", "1") != "0"
    model_dir = os.environ.get("PROTEINS_MODEL_DIR", "models")
    # Limit number of nodes to show in dropdown to improve responsiveness
    limit = int(os.environ.get("PROTEINS_NODE_LIMIT", "200"))
    # Load dataset
    dataset = ProteinsDataset(root=root, agg_method=agg_method, add_degree=add_degree)
    # Reconstruct full prediction matrices
    baseline_pred, gnn_pred = load_prediction_arrays(model_dir, dataset)
    dataset.baseline_pred = baseline_pred
    dataset.gnn_pred = gnn_pred
    # Build list of node IDs (strings) for dropdown
    node_list = dataset.list_nodes(limit=limit)
    node_ids: List[str] = [str(n) for n in node_list]

    @app.route("/")
    def index() -> str:
        return render_template("index.html", node_ids=node_ids)

    @app.route("/data")
    def data_endpoint() -> str:
        # Provide simple metadata for each node: id and number of positive labels
        data_list = []
        for nid in node_list:
            label_vec = dataset.labels[nid]
            pos_count = int(label_vec.sum())
            data_list.append({"id": str(nid), "pos_labels": pos_count})
        return jsonify(data_list)

    @app.route("/predict/<node_id>")
    def predict(node_id: str):
        try:
            nid = int(node_id)
        except ValueError:
            return jsonify({"error": "Invalid node ID"}), 400
        if nid < 0 or nid >= dataset.num_nodes:
            return jsonify({"error": "Node ID out of range"}), 404
        baseline_vec, gnn_vec = dataset.get_prediction(nid)
        # Compute top 5 labels for each model
        topk = 5
        # Baseline top labels
        baseline_indices = np.argsort(baseline_vec)[::-1][:topk]
        baseline_scores = [(int(idx), float(baseline_vec[idx])) for idx in baseline_indices]
        # GNN top labels
        gnn_indices = np.argsort(gnn_vec)[::-1][:topk]
        gnn_scores = [(int(idx), float(gnn_vec[idx])) for idx in gnn_indices]
        return jsonify({"baseline": baseline_scores, "gnn": gnn_scores})

    @app.route("/graph_json/<node_id>")
    def graph_json(node_id: str):
        try:
            nid = int(node_id)
        except ValueError:
            return jsonify({"error": "Invalid node ID"}), 400
        if nid < 0 or nid >= dataset.num_nodes:
            return jsonify({"error": "Node ID out of range"}), 404
        graph_data = dataset.get_graph_json(nid, hop=1, max_nodes=50)
        return jsonify(graph_data)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
