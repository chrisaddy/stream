"""Graph Convolutional Network for illicit transaction detection.

Leverages the UTXO transaction graph: edges represent Bitcoin payment flows.
GCN learns from both node features AND graph structure, unlike XGBoost which
only sees individual transaction features.
"""

import pickle
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class IllicitGCN(nn.Module):
    """3-layer GCN for binary classification (licit vs illicit)."""

    def __init__(self, in_channels: int = 165, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """Get probability of illicit class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=1)
        return probs[:, 1].cpu().numpy()


def train_gcn(
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    class_weights: torch.Tensor | None = None,
    patience: int = 20,
) -> tuple[IllicitGCN, dict]:
    """Train GCN with temporal masks and class-weighted loss.

    Args:
        data: PyG Data object with x, edge_index, y
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
        epochs: Max training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        class_weights: Tensor of class weights for imbalanced loss
        patience: Early stopping patience

    Returns:
        model: Trained GCN
        history: Training history dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Skip timestep feature (feature_0) — GCN uses 165 features
    x = data.x[:, 1:].to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    model = IllicitGCN(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_prauc = 0.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_prauc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_loss = criterion(val_out[val_mask], y[val_mask])
            val_probs = F.softmax(val_out[val_mask], dim=1)[:, 1].cpu().numpy()
            val_labels = y[val_mask].cpu().numpy()

        from sklearn.metrics import average_precision_score
        val_prauc = average_precision_score(val_labels, val_probs) if val_labels.sum() > 0 else 0.0

        history["train_loss"].append(float(loss))
        history["val_loss"].append(float(val_loss))
        history["val_prauc"].append(float(val_prauc))

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def serialize_gcn(model: IllicitGCN) -> bytes:
    buf = BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def deserialize_gcn(data: bytes, in_channels: int = 165) -> IllicitGCN:
    buf = BytesIO(data)
    model = IllicitGCN(in_channels=in_channels)
    model.load_state_dict(torch.load(buf, weights_only=True))
    model.eval()
    return model
