# ==========================================
# 1. SETUP, DEPENDENCIES & COMPUTE ENVIRONMENT
# ==========================================

import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.io as pio

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch_geometric.nn import TransformerConv, GraphNorm, global_mean_pool
from torch_geometric.utils import to_dense_batch, dropout_edge
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    import RNA
    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False
    print("[WARNING] ViennaRNA failed to load. Will fallback to distance-based heuristics.")

pio.renderers.default = 'iframe'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SYSTEM] Execution Environment: {device.type.upper()}")

# ==========================================
# 2. STRICT DATA INGESTION & BPP CACHING
# ==========================================

def get_real_path():
    print("[SYSTEM] Probing Kaggle mount points for REAL dataset...")
    paths_to_check = [
        '/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv',
        '/kaggle/input/stanford-ribonanza-rna-folding/train_data_QUICK_START.csv',
        '/kaggle/input/competitions/stanford-ribonanza-rna-folding/train_data.csv',
        'train_data.csv'
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            print(f"[SYSTEM]  Target acquired: {p}")
            return p
    raise FileNotFoundError("[ERROR] Real Kaggle dataset not found! NO DUMMY DATA ALLOWED.")

DATASET_PATH = get_real_path()
BPP_CACHE_FILE = "bpp_cache.pkl"

def get_bpp_matrix(sequence, cache):
    """Computes and strictly caches ViennaRNA BPP to fix extreme slowdowns."""
    if sequence in cache:
        return cache[sequence]
    
    if VIENNA_AVAILABLE:
        try:
            fc = RNA.fold_compound(sequence)
            fc.pf()
            bpp = np.array(fc.bpp())[1:, 1:] # 1-indexed in ViennaRNA
        except:
            bpp = np.zeros((len(sequence), len(sequence)))
    else:
        bpp = np.zeros((len(sequence), len(sequence)))
        
    cache[sequence] = bpp
    return bpp

def load_and_preprocess_data(filepath, sample_size=2000):
    print(f"[SYSTEM] Ingesting biological sequence and confidence data...")
    df = pd.read_csv(filepath, nrows=sample_size)
    sequences = df['sequence'].tolist()
    
    reactivity_cols = [c for c in df.columns if c.startswith('reactivity_0')]
    raw_targets = df[reactivity_cols].values.tolist()
    targets = [target[:len(seq)] for seq, target in zip(sequences, raw_targets)]
    
    error_cols = [c for c in df.columns if c.startswith('reactivity_error_0')]
    if len(error_cols) > 0:
        raw_errors = df[error_cols].values.tolist()
        weights = [[1.0 / (e + 0.05) if not math.isnan(e) else 0.0 for e in err[:len(seq)]] 
                   for seq, err in zip(sequences, raw_errors)]
    else:
        weights = [[1.0] * len(seq) for seq in sequences]
        
    return sequences, targets, weights

sequences, target_reactivities, confidence_weights = load_and_preprocess_data(DATASET_PATH, sample_size=2000)

if os.path.exists(BPP_CACHE_FILE):
    with open(BPP_CACHE_FILE, 'rb') as f:
        bpp_cache = pickle.load(f)
else:
    bpp_cache = {}

# ==========================================
# 3. BIOLOGICAL GRAPH CONSTRUCTION 
# ==========================================

def get_dist_bucket(dist):
    if dist == 0: return 0
    elif dist <= 2: return 1
    elif dist <= 4: return 2
    elif dist <= 8: return 3
    elif dist <= 16: return 4
    elif dist <= 32: return 5
    elif dist <= 64: return 6
    else: return 7

def get_sinusoidal_positional_encoding(seq_len, d_model=12):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def sequence_to_graph(sequence: str, reactivities: list = None, weights: list = None) -> Data:
    seq_len = len(sequence)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    
    # 1. NODE FEATURES (Sequence + Pos Encoding)
    base_feats = np.zeros((seq_len, 4))
    for i, base in enumerate(sequence):
        if base in mapping: base_feats[i, mapping[base]] = 1.0
            
    pos_feats = get_sinusoidal_positional_encoding(seq_len, d_model=12)
    x = torch.tensor(np.concatenate([base_feats, pos_feats], axis=-1), dtype=torch.float)
    
    # 2. EDGE CONSTRUCTION
    sources, targets, edge_types, edge_dists, edge_bpps = [], [], [], [], []
    
    # Backbone Edges
    for i in range(seq_len - 1):
        sources.extend([i, i + 1])
        targets.extend([i + 1, i])
        edge_types.extend([0, 0])
        edge_dists.extend([get_dist_bucket(1), get_dist_bucket(1)])
        edge_bpps.extend([1.0, 1.0])

    # Dynamic Top-K BPP Structural Edges 
    bpp_matrix = get_bpp_matrix(sequence, bpp_cache)
    bpp_matrix = np.maximum(bpp_matrix, bpp_matrix.T) # Ensure symmetry
    np.fill_diagonal(bpp_matrix, 0)
    
    K = min(3, seq_len - 1) # Top 3 interacting nodes
    for i in range(seq_len):
        row_probs = bpp_matrix[i]
        top_indices = np.argsort(row_probs)[-K:]
        for j in top_indices:
            prob = row_probs[j]
            if prob > 0.01: # Dynamic noise filter
                dist_bucket = get_dist_bucket(abs(i - j))
                sources.append(i)
                targets.append(j)
                edge_types.append(1)
                edge_dists.append(dist_bucket)
                edge_bpps.append(prob)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_dist = torch.tensor(edge_dists, dtype=torch.long)
    edge_bpp = torch.tensor(edge_bpps, dtype=torch.float).unsqueeze(1)
    
    y = torch.tensor(reactivities, dtype=torch.float).unsqueeze(1) if reactivities else None
    w = torch.tensor(weights, dtype=torch.float).unsqueeze(1) if weights else None

    return Data(x=x, edge_index=edge_index, y=y, weights=w,
                edge_type=edge_type, edge_dist=edge_dist, edge_bpp=edge_bpp)

print("[SYSTEM] Compiling RNA Graphs...")
graph_dataset = [sequence_to_graph(seq, react, w) for seq, react, w in zip(sequences, target_reactivities, confidence_weights)]

with open(BPP_CACHE_FILE, 'wb') as f: pickle.dump(bpp_cache, f)

split_idx = int(len(graph_dataset) * 0.9) # 90/10 Split
BATCH_SIZE = 16 # Stabilized batch size

train_loader = DataLoader(graph_dataset[:split_idx], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(graph_dataset[split_idx:], batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. KAGGLE SOTA TRI-FUSION ARCHITECTURE
# ==========================================

class TriFusionRNAModel(nn.Module):
    def __init__(self, node_in_dim=16, hidden_dim=192, heads=8, num_layers=5):
        super(TriFusionRNAModel, self).__init__()
        
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        
        # --- GRAPH BRANCH ---
        self.edge_type_emb = nn.Embedding(3, 16) 
        self.edge_dist_emb = nn.Embedding(8, 16)
        self.edge_proj = nn.Linear(16 + 16 + 1, hidden_dim // heads)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim // heads))
            self.norms.append(GraphNorm(hidden_dim))
            
        # --- SEQUENCE BRANCH 1 (1D BiLSTM w/ Padding Avoidance) ---
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        
        # --- SEQUENCE BRANCH 2 (1D CNN for local motifs) ---
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2)
        )
        
        # --- FUSION & REGRESSION ---
        fuse_dim = hidden_dim * 3 + (hidden_dim // 2)
        self.regressor = nn.Sequential(
            nn.Linear(fuse_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_type, edge_dist, edge_bpp, batch):
        x = self.node_emb(x)
        
        # 1. GRAPH BRANCH
        e = self.edge_proj(torch.cat([self.edge_type_emb(edge_type), self.edge_dist_emb(edge_dist), edge_bpp], dim=-1))
        
        edge_index_drop, edge_mask = dropout_edge(edge_index, p=0.1, training=self.training)
        e_drop = e[edge_mask]
        
        g_x = x
        for conv, norm in zip(self.convs, self.norms):
            g_x = g_x + F.gelu(norm(conv(g_x, edge_index_drop, edge_attr=e_drop), batch))
            
        # 2. SEQUENCE BRANCHES
        dense_x, mask = to_dense_batch(x, batch) 
        
        # CNN Processing
        cnn_in = dense_x.transpose(1, 2)
        c_x = self.cnn(cnn_in).transpose(1, 2)[mask]
        
        lengths = mask.sum(dim=1).cpu() 
        packed_x = pack_padded_sequence(dense_x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        s_x_dense, _ = pad_packed_sequence(packed_out, batch_first=True)
        s_x = s_x_dense[mask] 
        
        # 3. GLOBAL CONTEXT (Attention/Mean Pooling)
        global_context = global_mean_pool(g_x, batch)
        global_x_expanded = global_context[batch]
        
        # 4. FUSION
        fused_features = torch.cat([g_x, s_x, c_x, global_x_expanded], dim=-1)
        return self.regressor(fused_features)

model = TriFusionRNAModel().to(device)
print("[SYSTEM] Tri-Branch Graph-LSTM-CNN Fusion Model Initialized.")

# ==========================================
# 5. TRAINING ENGINE (AMP, GRAD ACCUMULATION)
# ==========================================

EPOCHS = 3
ACCUM_STEPS = 2 # Gradient Accumulation (16 * 2 = 32 Effective Batch Size)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.SmoothL1Loss(beta=0.5, reduction='none') 
scaler = torch.amp.GradScaler('cuda')

def train_engine(epochs=30):
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5

    for epoch in range(epochs):
        model.train()
        total_train_loss = train_nodes = 0
        optimizer.zero_grad()
        
        for i, b in enumerate(train_loader):
            b = b.to(device)
            
            with torch.amp.autocast('cuda'):
                preds = model(b.x, b.edge_index, b.edge_type, b.edge_dist, b.edge_bpp, b.batch)
                valid_mask = ~torch.isnan(b.y)
                if valid_mask.sum() == 0: continue
                
                loss = (criterion(preds[valid_mask], b.y[valid_mask]) * b.weights[valid_mask]).mean()
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += (loss.item() * ACCUM_STEPS) * b.num_graphs
            train_nodes += b.num_graphs
            
        # --- VALIDATION ---
        model.eval()
        total_val_loss = total_weight_sum = 0
        
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                with torch.amp.autocast('cuda'):
                    preds = model(b.x, b.edge_index, b.edge_type, b.edge_dist, b.edge_bpp, b.batch)
                    valid_mask = ~torch.isnan(b.y)
                    if valid_mask.sum() == 0: continue
                    
                    abs_err = torch.abs(preds[valid_mask] - b.y[valid_mask])
                    total_val_loss += torch.sum(abs_err * b.weights[valid_mask]).item()
                    total_weight_sum += torch.sum(b.weights[valid_mask]).item()
                
        val_mae = total_val_loss / max(1e-6, total_weight_sum)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_train_loss/max(1, train_nodes):.4f} | Val MAE: {val_mae:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), 'best_rna_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("[SYSTEM] Early Stopping Triggered.")
                break

train_engine(EPOCHS)
model.load_state_dict(torch.load('best_rna_model.pth', weights_only=True))

# ==========================================
# 6. INFERENCE & TTA (TEST-TIME AUGMENTATION)
# ==========================================

def predict_with_tta(sequence, model):
    """Predicts using Test-Time Augmentation (Forward + Reverse Sequence Averaging)"""
    model.eval()
    with torch.no_grad():
        # 1. Forward Pass
        g_fwd = sequence_to_graph(sequence).to(device)
        batch_fwd = torch.zeros(g_fwd.x.size(0), dtype=torch.long, device=device)
        pred_fwd = model(g_fwd.x, g_fwd.edge_index, g_fwd.edge_type, g_fwd.edge_dist, g_fwd.edge_bpp, batch_fwd)
        
        # 2. Reverse Pass (TTA)
        seq_rev = sequence[::-1]
        g_rev = sequence_to_graph(seq_rev).to(device)
        batch_rev = torch.zeros(g_rev.x.size(0), dtype=torch.long, device=device)
        pred_rev = model(g_rev.x, g_rev.edge_index, g_rev.edge_type, g_rev.edge_dist, g_rev.edge_bpp, batch_rev)
        
        pred_rev_aligned = torch.flip(pred_rev, dims=[0])
        
        # 3. Ensemble Average (Multi-Branch Arch + TTA)
        final_pred = (pred_fwd + pred_rev_aligned) / 2.0
    return final_pred.flatten().cpu().numpy()

test_sequence = "GGAAAACCCCCUUUUUUGGGGGAAAAACCCC"
reactivities = predict_with_tta(test_sequence, model)

# Visualization
bases = list(test_sequence)
x_indices = np.arange(len(bases))
max_idx, max_val = np.argmax(reactivities), np.max(reactivities)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_indices, y=reactivities, fill='tozeroy', fillcolor='rgba(255, 0, 102, 0.1)', mode='lines+markers',
    line=dict(color='#FF0066', width=3, shape='spline'), 
    marker=dict(size=8, color='#FF0066', symbol='circle', line=dict(color='#0F0F0F', width=2)),
    name='TTA Ensembled Exposure',
    hoverinfo='text',
    text=[f"Base: <b>{b}</b><br>Position: {i}<br>Reactivity: {r:.4f}" for i, (b, r) in enumerate(zip(bases, reactivities))]
))

fig.add_trace(go.Scatter(
    x=[max_idx], y=[max_val], mode='markers',
    marker=dict(size=16, color='#00FF88', symbol='diamond', line=dict(color='#FFFFFF', width=2)),
    name=f'Optimal Target (Base {bases[max_idx]})'
))

fig.update_layout(
    title=dict(text='<b>SOTA AI RNA Prediction</b><br><sup>Tri-Fusion (Graph+LSTM+CNN) w/ TTA</sup>', font=dict(size=22, color="#FFFFFF"), x=0.05),
    paper_bgcolor='#0F0F0F', plot_bgcolor='#0F0F0F',
    xaxis=dict(title='Nucleotide Sequence', tickmode='array', tickvals=x_indices, ticktext=bases, gridcolor='#222222', color='#888888'),
    yaxis=dict(title='Reactivity Index', gridcolor='#222222', color='#888888'), hovermode='x unified'
)
fig.show()
