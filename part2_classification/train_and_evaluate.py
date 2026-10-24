import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from balance_and_preprocess import FEATURE_COLUMNS, get_data
from baseline_gridsearch import BEST_PARAMS_PATH, ClassificationMLP


ROOT = Path(__file__).resolve().parent
PLOTS_PATH = ROOT / "plots"
HISTORY_PATH = ROOT / "classification_training_history.csv"
LEARNING_CURVES_PATH = PLOTS_PATH / "classification_learning_curves.png"
CONFUSION_MATRIX_PATH = PLOTS_PATH / "confusion_matrix_baseline.png"
SEED = 42
DEFAULT_PARAMS: dict[str, Any] = {
    "lr": 0.001,
    "optimizer": "Adam",
    "hidden_dim": 64,
    "num_layers": 2,
    "batch_size": 64,
}


def get_best_params() -> dict[str, Any]:
    if not BEST_PARAMS_PATH.is_file():
        print("Fichier de paramètres absent ; utilisation des valeurs par défaut.")
        return DEFAULT_PARAMS.copy()
    try:
        params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Impossible de lire {BEST_PARAMS_PATH}: {error}") from error
    required = set(DEFAULT_PARAMS)
    if not isinstance(params, dict) or not required.issubset(params):
        raise ValueError(
            f"{BEST_PARAMS_PATH} doit contenir : {', '.join(sorted(required))}."
        )
    return params


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, list[float]]:
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size et lr doivent être positifs.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    criterion = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    effective_batch_size = min(batch_size, len(train_dataset))
    uses_batch_norm = any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules())
    drop_last = (
        uses_batch_norm
        and len(train_dataset) > effective_batch_size
        and len(train_dataset) % effective_batch_size == 1
    )
    if uses_batch_norm and effective_batch_size == 1:
        raise ValueError("BatchNorm nécessite au moins deux observations par lot.")
    loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_x = torch.from_numpy(x_train.astype(np.float32)).to(device)
    train_y = torch.from_numpy(y_train.astype(np.int64)).to(device)
    test_x = torch.from_numpy(x_test.astype(np.float32)).to(device)
    test_y = torch.from_numpy(y_test.astype(np.int64)).to(device)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(train_x)
            test_logits = model(test_x)
            train_loss = criterion(train_logits, train_y).item()
            test_loss = criterion(test_logits, test_y).item()
            train_predictions = train_logits.argmax(dim=1)
            test_predictions = test_logits.argmax(dim=1)
            train_acc = (train_predictions == train_y).float().mean().item()
            test_acc = (test_predictions == test_y).float().mean().item()
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(test_acc))
        print(
            f"Époque {epoch + 1:02d}/{epochs} — "
            f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
            f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
        )
    return history


def plot_history(history: dict[str, list[float]]) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set(title="Loss (Cross-Entropy) vs Epochs", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["test_acc"], label="Test")
    axes[1].set(title="Accuracy vs Epochs", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(LEARNING_CURVES_PATH, dpi=150)
    plt.close(fig)


def predict(model: nn.Module, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32)).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set(title="Matrice de confusion — modèle baseline", xlabel="Prédit", ylabel="Réel")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_training() -> tuple[nn.Module, dict[str, list[float]]]:
    os.makedirs(PLOTS_PATH, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    params = get_best_params()
    x_train, x_test, y_train, y_test, encoder = get_data()
    model = ClassificationMLP(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        num_classes=len(encoder.classes_),
    )
    history = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=35,
        batch_size=int(params["batch_size"]),
        lr=float(params["lr"]),
    )
    pd.DataFrame(history).to_csv(HISTORY_PATH, index_label="epoch")
    plot_history(history)

    predictions = predict(model, x_test)
    print(f"\nAccuracy globale (test) : {accuracy_score(y_test, predictions):.4f}")
    print("\n=== Rapport de classification (test) ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )
    save_confusion_matrix(
        y_test, predictions, encoder.classes_, CONFUSION_MATRIX_PATH
    )
    print(f"Courbes : {LEARNING_CURVES_PATH}")
    print(f"Matrice de confusion : {CONFUSION_MATRIX_PATH}")
    print(f"Historique : {HISTORY_PATH}")
    return model, history


if __name__ == "__main__":
    run_training()
