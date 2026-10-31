"""Autoencoder utilities used by the DAPS pipeline."""

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap

warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def set_global_seed(seed):
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def set_seed(seed):
    """Set deterministic PyTorch-related seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class AE(nn.Module):
    """Symmetric autoencoder used for SLOPE and AE-UMAP baselines."""

    def __init__(self, dims, activation="tanh", dropout=0.0, batch_norm=False):
        super(AE, self).__init__()
        self.dims = dims
        self.encoder = self.build_layers(dims[: dims.index(min(dims)) + 1], activation, dropout, batch_norm)
        self.decoder = self.build_layers(dims[dims.index(min(dims)) :], activation, dropout, batch_norm, reverse=True)

    def build_layers(self, layer_dims, activation, dropout, batch_norm, reverse=False):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if not reverse or i < len(layer_dims) - 2:
                if isinstance(activation, str):
                    if activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "leakyrelu":
                        layers.append(nn.LeakyReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    else:
                        layers.append(nn.ReLU())
                else:
                    layers.append(activation())
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view((-1, self.encoder[0].in_features))
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = x.view((-1, self.encoder[0].in_features))
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def get_encoded_dim(self):
        return min(self.dims)


def compute_directional_loss_tensor(encoded_data, train_data_identifiers):
    """Compute the longitudinal directional consistency loss."""
    sorted_indices = train_data_identifiers.sort_values(["RID", "PHC_Age_PET"]).index
    sorted_rids = train_data_identifiers.loc[sorted_indices, "RID"].values
    sorted_encoded_data = encoded_data[sorted_indices]

    directional_losses = []
    for i in range(sorted_encoded_data.size(0) - 2):
        if sorted_rids[i] == sorted_rids[i + 1] and sorted_rids[i + 1] == sorted_rids[i + 2]:
            z1 = sorted_encoded_data[i]
            z2 = sorted_encoded_data[i + 1]
            z3 = sorted_encoded_data[i + 2]

            vector_12 = z2 - z1
            vector_23 = z3 - z2
            vector_12_norm = torch.norm(vector_12) + 1e-12
            vector_23_norm = torch.norm(vector_23) + 1e-12
            cos = torch.sum(vector_12 * vector_23) / (vector_12_norm * vector_23_norm)
            directional_losses.append(1.0 - cos)

    if directional_losses:
        return torch.stack(directional_losses).mean()
    return torch.tensor(0.0, requires_grad=True)


def initialize_weights(model, seed=42):
    """Initialize shallow linear layers for the baseline autoencoder."""
    torch.manual_seed(seed)
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)


def train_AE(
    model,
    X_train_tensor,
    X_test_tensor,
    train_data_identifiers,
    test_data_identifiers,
    criterion,
    optimizer_class,
    lr,
    epochs=500,
    lambda_reconstruction=1.0,
    lambda_directional=1.0,
    directional_loss_enabled=True,
    return_losses=False,
):
    """Train the baseline autoencoder and return UMAP embeddings."""
    torch.use_deterministic_algorithms(True)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    set_global_seed(42)

    print(f"Training for {epochs} epochs with optimizer: {str(optimizer_class)}, learning rate: {lr}")
    print()

    initialize_weights(model, seed=42)
    torch.use_deterministic_algorithms(True)

    optimizer = optimizer_class(list(model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_history = []

    for epoch in range(epochs):
        set_seed(42)
        optimizer.zero_grad()

        encoded = model.encode(X_train_tensor)
        reconstructed = model.decode(encoded)
        recon_loss = criterion(reconstructed, X_train_tensor)

        directional_loss = 0.0
        if directional_loss_enabled:
            directional_loss = compute_directional_loss_tensor(encoded, train_data_identifiers)

        total_loss = lambda_reconstruction * recon_loss + lambda_directional * directional_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 5 == 0:
            loss_entry = {
                "epoch": epoch,
                "total_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
            }
            print(loss_entry)

        if directional_loss_enabled:
            loss_entry["directional_loss"] = lambda_directional * directional_loss.item()

        if epoch % 5 == 0:
            loss_history.append(loss_entry)

    train_loss_df = pd.DataFrame(loss_history)

    model.eval()
    with torch.no_grad():
        encoded_test = model.encode(X_test_tensor)
        reconstructed_test = model.decode(encoded_test)
        test_recon_loss = criterion(reconstructed_test, X_test_tensor)

        directional_loss_test = torch.tensor(0.0)
        if directional_loss_enabled:
            directional_loss_test = compute_directional_loss_tensor(encoded_test, test_data_identifiers)

        total_loss_test = (lambda_reconstruction * test_recon_loss) + (lambda_directional * directional_loss_test)

    final_test_losses = {
        "Metric": ["Reconstruction Loss", "Directional Loss", "Total Loss"],
        "Train": [
            train_loss_df["recon_loss"].iloc[-1],
            train_loss_df["directional_loss"].iloc[-1] if directional_loss_enabled else 0.0,
            train_loss_df["total_loss"].iloc[-1],
        ],
        "Test": [
            test_recon_loss.item(),
            directional_loss_test.item() if directional_loss_enabled else 0.0,
            total_loss_test.item(),
        ],
    }
    print(final_test_losses)

    umap_model = umap.UMAP(n_components=2, random_state=42)
    data_2d_umap_train = umap_model.fit_transform(encoded.detach().cpu().numpy())

    train_encodings_umap = pd.DataFrame(data_2d_umap_train, columns=["UMAP1", "UMAP2"])
    train_encodings_umap = pd.merge(train_encodings_umap, train_data_identifiers, left_index=True, right_index=True)

    if return_losses:
        return model, umap_model, train_encodings_umap, encoded.detach().cpu().numpy(), loss_history
    return model, umap_model, train_encodings_umap, encoded.detach().cpu().numpy()


def test_AE(model, X_test_tensor, test_data_identifiers, umap_model):
    """Encode test data and project it with the fitted UMAP model."""
    model.eval()
    with torch.no_grad():
        encoded_test = model.encode(X_test_tensor)

    data_2d_umap_test = umap_model.transform(encoded_test.detach().cpu().numpy())
    test_encodings_umap = pd.DataFrame(data_2d_umap_test, columns=["UMAP1", "UMAP2"])
    test_encodings_umap = pd.merge(test_encodings_umap, test_data_identifiers, left_index=True, right_index=True)

    return test_encodings_umap, encoded_test.detach().cpu().numpy()
