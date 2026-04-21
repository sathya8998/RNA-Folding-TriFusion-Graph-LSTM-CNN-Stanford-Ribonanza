# Tri-Branch Graph-LSTM-CNN Fusion Network for RNA Reactivity Prediction — Stanford Ribonanza RNA Folding

## PROJECT OVERVIEW

This repository implements a state-of-the-art RNA chemical reactivity prediction pipeline for the Stanford Ribonanza RNA Folding Kaggle competition. The core architecture — TriFusionRNAModel — simultaneously processes each RNA sequence through three parallel branches (Graph Transformer, BiLSTM, and 1D CNN) fused into a single regression head. Structural priors from ViennaRNA base-pair probability matrices are embedded directly into the graph topology, enabling biologically grounded message passing beyond simple sequence proximity.

**Biological Graph Construction**: Each RNA sequence is converted to a PyTorch Geometric `Data` object with two biologically distinct edge types:
- **Backbone Edges**: Sequential phosphodiester bond connectivity (every adjacent nucleotide pair, bidirectional)
- **Structural BPP Edges**: Top-K dynamic edges per node derived from ViennaRNA base-pair probability (BPP) matrices, filtered at `p > 0.01` to suppress noise. Fully symmetric via `np.maximum(bpp, bpp.T)`

**Node Features**: One-hot nucleotide encoding (A/C/G/U, 4-dim) concatenated with 12-dimensional sinusoidal positional encodings, giving a 16-dimensional input feature per nucleotide node.

**Edge Features**: Per-edge embedding of type (backbone vs. structural), sequence distance bucket (log-scale, 8 bins), and raw BPP probability — projected into the transformer edge attention space.

**TriFusion Architecture**:
- **Graph Branch**: 5-layer `TransformerConv` with `GraphNorm` and residual connections; stochastic edge dropout (`p=0.1`) during training for structural regularisation
- **BiLSTM Branch**: 2-layer bidirectional LSTM with `pack_padded_sequence` to handle variable-length sequences without padding contamination
- **CNN Branch**: Dual `Conv1d` layers (kernel 3 + kernel 5) with GELU activations for local sequence motif extraction
- **Global Context**: `global_mean_pool` over graph node embeddings, broadcast back to node level and concatenated to the fusion vector

**Confidence-Weighted Loss**: Per-nucleotide `SmoothL1Loss` (β=0.5) weighted by inverse reactivity error (`1 / (error + 0.05)`), down-weighting unreliable experimental measurements during training.

**Test-Time Augmentation (TTA)**: Inference averages forward-sequence and reverse-sequence predictions (with reverse-flip alignment) for ensemble-level uncertainty reduction without additional model training.

## DATASET

**Competition — Stanford Ribonanza RNA Folding**
- **Source**: Kaggle Competition — `stanford-ribonanza-rna-folding`
- **Link**: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding

## MODEL ARCHITECTURE

**Input per Node**: 16-dimensional feature vector (4-dim one-hot + 12-dim sinusoidal positional encoding)

**Node Embedding**: `Linear(16 → 192)`

**Edge Embedding Pipeline**:
- `Embedding(3, 16)` for edge type + `Embedding(8, 16)` for distance bucket + raw BPP scalar
- Projected via `Linear(33 → 24)` (192 // 8 heads) into transformer edge attention space

**Graph Branch — 5× TransformerConv Layers**:
- `TransformerConv(192 → 24, heads=8, edge_dim=24)` with `GraphNorm(192)` and residual addition
- Stochastic `dropout_edge(p=0.1)` applied to edge index and attributes during training

**BiLSTM Branch**:
- `LSTM(192 → 96, num_layers=2, bidirectional=True, dropout=0.1)` with `pack_padded_sequence` / `pad_packed_sequence`
- Output re-indexed to node level via boolean mask

**CNN Branch**:
- `Conv1d(192 → 96, k=3)` → `GELU` → `Conv1d(96 → 96, k=5)` on dense-batched sequence tensor
- Output re-indexed to node level via boolean mask

**Global Context**: `global_mean_pool` over graph branch output → expanded back to per-node via `batch` index

**Fusion Dimension**: `192 (graph) + 192 (LSTM) + 96 (CNN) + 192 (global) = 672`

**Regression Head**: `Linear(672→256)` → `LayerNorm` → `GELU` → `Dropout(0.2)` → `Linear(256→128)` → `GELU` → `Linear(128→1)`


## KEY FEATURES

**BPP Caching System**: ViennaRNA base-pair probability matrices are computed once per unique sequence and persisted to `bpp_cache.pkl` via Python `pickle`. On re-run, cached matrices are loaded instantly, eliminating repeated O(n²) folding computation — critical for sequences up to thousands of nucleotides.

**ViennaRNA Fallback**: If the `RNA` (ViennaRNA) package is unavailable, the pipeline falls back gracefully to zero BPP matrices, retaining full backbone graph connectivity and all other features without crashing.

**Dynamic Top-K BPP Edges**: Rather than thresholding at a fixed probability, the top-3 highest-probability pairing partners per nucleotide are selected first, then filtered by `p > 0.01`. This adapts the graph density to each sequence's structural confidence rather than imposing a global cutoff.

**Distance Bucketing**: Sequence separation between any two connected nodes is encoded into 8 logarithmic distance bins (0, 1–2, 3–4, 5–8, 9–16, 17–32, 33–64, 64+) and embedded as a learned `Embedding(8, 16)`, giving the model awareness of local vs. long-range interactions.

**Gradient Accumulation**: 2-step accumulation with AMP `autocast('cuda')` and `clip_grad_norm_(max_norm=1.0)` stabilises training with effective batch size 32 while keeping GPU memory footprint at batch size 16.

**Early Stopping**: Patience-based stopping (patience=5 epochs) on validation weighted-MAE with automatic best-chec
