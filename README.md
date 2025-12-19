## Overview

ProteoTrace provides an interactive Flask dashboard for exploring protein function predictions on the `ogbn-proteins` benchmark. The application mirrors the TruthTrace visualisation pattern: a dropdown for selecting protein nodes, side-by-side comparison of baseline and GraphSAGE predictions, and a neighbourhood graph extracted for D3-based rendering. Core components include:

- `app.py`: Flask entry point that loads data, reconstructs prediction matrices, and exposes endpoints for node metadata, per-node predictions, and graph JSON payloads.
- `data_loader.py`: Data access layer that downloads or reads the OGB dataset, aggregates edge features into node representations, maintains train/validation/test splits, and provides graph neighbourhood sampling for the frontend.
- Analysis scripts (`ablation_study.py`, `sensitivity_analysis.py`, `aggregate_analysis.py`, `shap_analysis.py`, `significance_tests.py`) and training utilities (`train.py`, `train_hybrid.py`, `train_script.sh`) for model experimentation and evaluation.

## System requirements

- **Operating system:** Linux or macOS recommended.
- **Python:** 3.9+ (matching the dependencies in `requirements.txt`).
- **Memory:** At least **16 GB of system RAM is mandatory**. The `ogbn-proteins` graph is large; loading node features, prediction matrices, and NetworkX subgraphs simultaneously will exceed the capacity of lower-memory machines. Systems below this threshold will fail to render neighbourhood graphs and are likely to terminate the process due to out-of-memory conditions. For reliable interactivity, 32 GB or more is preferred.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ProteoTrace
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Fetch the ogbn-proteins data**
   - By default, files are cached under `./data/ogbn-proteins`. The first run will use the OGB downloader.
   - If network access is restricted, manually place `edge_index.npy`, `edge_feat.npy`, `node_label.npy`, and related raw files into `./data/ogbn-proteins/raw`.

## Running the dashboard

The application is configured through environment variables that select data and model locations:

- `PROTEINS_ROOT`: Path to the dataset cache (default: `./data`).
- `PROTEINS_MODEL_DIR`: Directory containing prediction files such as `baseline_probs_[split].npy` and `graphsage_probs_[split].npy` (default: `models`).
- `PROTEINS_AGG_METHOD`: Edge aggregation strategy for node features (`mean` or `sum`; default: `mean`).
- `PROTEINS_ADD_DEGREE`: Set to `0` to omit degree-based features (default: `1`).
- `PROTEINS_NODE_LIMIT`: Number of node IDs to expose in the dropdown to keep the UI responsive (default: `200`).

Start the Flask server (development mode):
```bash
export PROTEINS_ROOT=./data
export PROTEINS_MODEL_DIR=./models
python app.py
```

Then open http://127.0.0.1:5000/ to interact with the dashboard. When selecting a node, the backend retrieves the top predicted functions from both models and emits a 1-hop neighbourhood for visualisation.

## Training and experimentation

Model training and analysis scripts assume the dataset is already cached:

- **Train baseline and GraphSAGE models**: `python train.py` or the hybrid variant `python train_hybrid.py`.
- **Ablation and sensitivity studies**: run `ablation_study.py` or `sensitivity_analysis.py` after training outputs are saved in `models/`.
- **Explainability and significance testing**: `shap_analysis.py` and `significance_tests.py` operate on the stored predictions and labels to quantify model behaviour.

Generated prediction arrays are stored under `models/` and loaded automatically by `app.py` during startup.

## Troubleshooting

- **Out-of-memory crashes or missing graphs:** Ensure the host machine has at least 16 GB of available RAM; otherwise, NetworkX neighbourhood extraction and NumPy prediction matrices cannot be materialised simultaneously.
- **Dataset download errors:** Verify the raw files exist at `PROTEINS_ROOT/ogbn-proteins/raw` or re-run the download with a clean cache.
- **Slow dropdown population:** Lower `PROTEINS_NODE_LIMIT` to reduce initial payload size for the client.
