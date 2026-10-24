import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_gridsearch import (
    BEST_PARAMS_PATH,
    FEATURE_COLUMNS,
    RegressionMLP,
    ROOT,
    load_regression_data,
    run_grid_search,
)


HISTORY_PATH = ROOT / "training_history.csv"
PLOT_PATH = ROOT / "training_curves.png"
SEED = 42


def get_best_params() -> dict[str, Any]:
    if not BEST_PARAMS_PATH.is_file():
        print("Aucun best_params.json trouvé ; lancement automatique du GridSearchCV.")
        return run_grid_search()
    try:
        params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Impossible de lire {BEST_PARAMS_PATH}: {error}") from error
    required = {"lr", "optimizer", "hidden_dim", "num_layers", "batch_size"}
    if not isinstance(params, dict) or not required.issubset(params):
        raise ValueError(
            f"{BEST_PARAMS_PATH} doit contenir : {', '.join(sorted(required))}."
        )
    return params


def make_optimizer(
    model: nn.Module, optimizer_name: str, lr: float, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    optimizers: dict[str, type[torch.optim.Optimizer]] = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }
    if optimizer_name not in optimizers:
        raise ValueError(
            f"Optimiseur non pris en charge : {optimizer_name}. "
            f"Choisissez parmi {', '.join(optimizers)}."
        )
    return optimizers[optimizer_name](
        model.parameters(), lr=lr, weight_decay=weight_decay
    )


def train_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_name: str,
    weight_decay: float = 0.0,
) -> dict[str, list[float]]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs et batch_size doivent être positifs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    criterion = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32)),
        torch.from_numpy(train_target.astype(np.float32)),
    )
    effective_batch_size = min(batch_size, len(dataset))
    uses_batch_norm = any(
        isinstance(layer, nn.BatchNorm1d) for layer in model.modules()
    )
    if uses_batch_norm and len(dataset) > 1:
        effective_batch_size = max(2, effective_batch_size)
        if len(dataset) > effective_batch_size and len(dataset) % effective_batch_size == 1:
            effective_batch_size -= 1
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
    )
    train_x = torch.from_numpy(train_features.astype(np.float32)).to(device)
    train_y = torch.from_numpy(train_target.astype(np.float32)).to(device)
    test_x = torch.from_numpy(test_features.astype(np.float32)).to(device)
    test_y = torch.from_numpy(test_target.astype(np.float32)).to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_r2": [],
        "test_r2": [],
    }
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_target in loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_predictions = model(train_x)
            test_predictions = model(test_x)
            train_loss = criterion(train_predictions, train_y).item()
            test_loss = criterion(test_predictions, test_y).item()

        train_predictions_np = train_predictions.cpu().numpy().reshape(-1)
        test_predictions_np = test_predictions.cpu().numpy().reshape(-1)
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        history["train_r2"].append(
            float(r2_score(train_target.reshape(-1), train_predictions_np))
        )
        history["test_r2"].append(
            float(r2_score(test_target.reshape(-1), test_predictions_np))
        )
        print(
            f"Époque {epoch + 1:03d}/{epochs} — "
            f"train_loss={train_loss:.6f}, test_loss={test_loss:.6f}, "
            f"train_R²={history['train_r2'][-1]:.4f}, "
            f"test_R²={history['test_r2'][-1]:.4f}"
        )
    return history


def plot_history(history: dict[str, list[float]]) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set(title="MSE Loss vs Epochs", xlabel="Époque", ylabel="MSE")
    axes[0].legend()

    axes[1].plot(epochs, history["train_r2"], label="Train")
    axes[1].plot(epochs, history["test_r2"], label="Test")
    axes[1].set(title="R² Score vs Epochs", xlabel="Époque", ylabel="R²")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Graphiques enregistrés dans : {PLOT_PATH}")


def run_training() -> dict[str, list[float]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    best_params = get_best_params()
    features, target = load_regression_data()
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    model = RegressionMLP(
        hidden_dim=int(best_params["hidden_dim"]),
        num_layers=int(best_params["num_layers"]),
    )
    history = train_model(
        model,
        train_features,
        train_target,
        test_features,
        test_target,
        epochs=100,
        batch_size=int(best_params["batch_size"]),
        lr=float(best_params["lr"]),
        optimizer_name=str(best_params["optimizer"]),
    )
    pd.DataFrame(history).to_csv(HISTORY_PATH, index_label="epoch")
    print(f"Métriques par époque enregistrées dans : {HISTORY_PATH}")
    plot_history(history)
    return history


if __name__ == "__main__":
    run_training()
