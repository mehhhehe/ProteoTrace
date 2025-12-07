## ProteinsFull: Graph‑Based Prediction on OGBN‑Proteins

This package provides a **TruthTrace‑style** implementation for the
[`ogbn‑proteins`](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins) dataset.  The goal is to predict 112 binary
function labels for each protein in a large protein–protein
association network.  Unlike the original research project, this code
is designed to run on modest hardware (≤16 GB RAM, CPU only) and
includes an **interactive Flask frontend** for exploring predictions
and neighbourhood subgraphs.  The structure and workflow mirror the
`truthtrace_full` application: clear separation of data loading,
model definition, training/evaluation and a web dashboard.

### Big‑Data Context

The OGBN‑proteins graph is substantial: it contains **132 534 nodes**
and **39 561 252 edges**【9459112307277†L103-L112】.  Each edge is annotated
with an 8‑dimensional feature vector expressing different types of
biological associations, and the task is to predict 112 function
labels per node【9459112307277†L114-L117】.  The official split is
species‑wise【9459112307277†L119-L121】, forcing models to generalise across
organisms.  Training state‑of‑the‑art GNNs on this graph can require
GPUs and large memory.  Our implementation therefore focuses on
light‑weight architectures and neighbour sampling with conservative
batch sizes so that training is feasible on a CPU with 16 GB RAM.

### Repository Layout

```
proteins_full/
├── data_loader.py    # Load OGBN‑proteins, construct features and graphs
├── model.py          # Baseline MLP and GraphSAGE/GAT definitions
├── train.py          # Train baseline and GNN with neighbour sampling
├── app.py            # Flask backend serving predictions and graphs
├── templates/
│   ├── index.html    # Main dashboard page
│   └── graph.html    # D3.js visualisation template
├── static/
│   ├── d3.v7.min.js  # D3 library
│   └── style.css     # Minimal styling
└── requirements.txt  # Python dependencies
```

### Installation

Create a Python virtual environment (recommended) and install the
requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The dependencies include `torch`, `torch_geometric`, `ogb`, `flask`,
`networkx` and `scikit‑learn`.  Versions are pinned to ensure
compatibility on CPU.  Installing `torch_geometric` without CUDA can
be slow; follow the [official installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if you encounter issues.

### Dataset Preparation

By default the training script will use the `ogb` library to fetch the
`ogbn‑proteins` dataset on the fly.  However, the original OGB
download URLs have recently become unreliable, so we recommend
downloading the data manually from **OpenDataLab** and placing it
into a local cache.  Manual download is a one‑time step and avoids
runtime errors.

1. Visit the OpenDataLab page for the
   [**OGBN‑Proteins** dataset](https://opendatalab.com/), log in if
   necessary, and download the archive corresponding to
   *ogbn‑proteins*.  The archive contains a `raw/` folder with
   NumPy arrays and split indices (e.g. `edge_index.npy`,
   `edge_feat.npy`, `node_label.npy`, `train_idx.npy`, `valid_idx.npy`,
   `test_idx.npy`).  These files correspond to the raw graph,
   eight‑dimensional edge features, 112‑dimensional node labels and
   the species‑wise train/val/test split【9459112307277†L103-L112】【9459112307277†L114-L117】.
2. In your `proteins_full` project directory, create the cache
   structure expected by the loader.  For example:

   ```text
   proteins_full/
     data/
       ogbn-proteins/
         raw/
           edge_index.npy
           edge_feat.npy
           node_label.npy
           train_idx.npy
           valid_idx.npy
           test_idx.npy
           ...
   ```

   Move or extract the contents of the downloaded `raw/` directory into
   `proteins_full/data/ogbn-proteins/raw/`.  Do **not** rename the files
   – the loader expects the original names.
3. If you skip the manual download, the data loader will fall back to
   `ogb.NodePropPredDataset` which attempts to fetch the data from the
   OGB mirror.  Should that download fail (e.g. due to 404 errors),
   perform the manual download described above.

### Training

Run the training script to train both a baseline MLP and a GNN model:

```bash
python train.py \
  --root ./data \            # directory containing ogbn-proteins/raw
  --model-dir models \       # where to save predictions and weights
  --epochs 10 \              # fewer epochs for CPU training
  --hidden-dim 64 \          # smaller hidden size to reduce memory usage
  --num-layers 2 \           # shallow networks run faster on CPU
  --num-neighbors 10 5 \     # neighbour sampling sizes per GNN layer
  --batch-size 256 \         # adjust batch size to fit into 16 GB
  --device cpu               # force CPU execution
```

The script loads the graph, constructs aggregated node features by
averaging incident edge features【9459112307277†L103-L112】 and optionally
appends degree statistics.  It trains a baseline MLP on these
features and a GraphSAGE or GAT model using mini‑batch neighbour
sampling.  After training, the script computes predictions for all
nodes and saves them as `.npy` files in `model_dir`.  These files are
used by the web dashboard.

### Running the Dashboard

After training, start the Flask application to explore predictions and
local neighbourhoods:

```bash
export PROTEINS_ROOT=./data
export PROTEINS_MODEL_DIR=models
python app.py
```

Navigate to <http://127.0.0.1:5000> in your browser.  The dashboard
displays a dropdown of protein IDs (first 500 by default), the
predicted probabilities for the top five functions according to both
the MLP and GNN models, and an interactive graph showing the selected
protein’s immediate neighbours.  You can modify the number of nodes
shown in the dropdown by adjusting the `MAX_NODES` constant in
`app.py`.

### Notes on Performance

This code is intentionally simplified to run on CPU with ≤16 GB of
memory.  The model definitions use shallow architectures (e.g. two
GraphSAGE layers with 64 hidden units) and the neighbour sampling
sizes are conservative (`--num-neighbors 10 5`).  Training the GNN
for 10 epochs on CPU takes approximately 20–30 minutes and fits
comfortably into 16 GB.  If you have more memory or a GPU, you can
increase `--hidden-dim`, `--num-layers` or `--num-neighbors` to
improve accuracy.

### Citation

The ogbn‑proteins dataset is described in the OGB repository
documentation【9459112307277†L103-L112】.  The GraphSAGE architecture is
introduced by Hamilton et al. 【565794615307797†L20-L26】 and GAT by
Veličković et al. 【61684600892784†L49-L59】.  This project borrows the
hybrid motivation of combining learned embeddings with classical
classifiers from the original ogbn‑proteins study but focuses on
resource‑constrained training.