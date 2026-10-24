import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from balance_and_preprocess import FEATURE_COLUMNS, get_data
from baseline_gridsearch import ClassificationMLP
from regularized_model import build_regularized_model
from train_and_evaluate import get_best_params, predict, train_model


ROOT = Path(__file__).resolve().parent
PLOTS_PATH = ROOT / "plots"
COMPARISON_PATH = PLOTS_PATH / "models_comparison.png"
SEED = 42


def run_comparison() -> dict[str, dict[str, float]]:
    os.makedirs(PLOTS_PATH, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    params = get_best_params()
    x_train, x_test, y_train, y_test, encoder = get_data()

    model_metrics: dict[str, dict[str, float]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    models: list[tuple[str, torch.nn.Module, torch.optim.Optimizer | None]] = []

    baseline = ClassificationMLP(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        num_classes=len(encoder.classes_),
    )
    models.append(("Baseline", baseline, None))
    regularized, regularized_optimizer = build_regularized_model(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        num_classes=len(encoder.classes_),
        lr=float(params["lr"]),
        weight_decay=1e-4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    models.append(("Régularisé", regularized, regularized_optimizer))

    for name, model, optimizer in models:
        print(f"\n=== Entraînement du modèle {name} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        history = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=35,
            batch_size=int(params["batch_size"]),
            lr=float(params["lr"]),
            weight_decay=1e-4 if name == "Régularisé" else 0.0,
            optimizer=optimizer,
        )
        predictions = predict(model, x_test)
        train_predictions = predict(model, x_train)
        predictions_by_model[name] = predictions
        train_loss = history["train_loss"][-1]
        test_loss = history["test_loss"][-1]
        model_metrics[name] = {
            "train_accuracy": float(accuracy_score(y_train, train_predictions)),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_macro_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    labels=np.arange(len(encoder.classes_)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "generalization_loss": float(test_loss - train_loss),
        }

    print("\n=== Comparaison finale ===")
    print(
        f"{'Modèle':<15} {'Train Accuracy':>15} {'Test Accuracy':>15} "
        f"{'Test Macro F1':>15} {'Perte généralisation':>22}"
    )
    print("-" * 85)
    for name, metrics in model_metrics.items():
        print(
            f"{name:<15} {metrics['train_accuracy']:>15.4f} "
            f"{metrics['test_accuracy']:>15.4f} "
            f"{metrics['test_macro_f1']:>15.4f} "
            f"{metrics['generalization_loss']:>22.4f}"
        )

    labels = np.arange(len(encoder.classes_))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (name, predictions) in zip(axes, predictions_by_model.items()):
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_,
            ax=ax,
        )
        ax.set(title=f"Matrice de confusion — {name}", xlabel="Prédit", ylabel="Réel")
    fig.tight_layout()
    fig.savefig(COMPARISON_PATH, dpi=150)
    plt.close(fig)
    print(f"\nComparaison des matrices enregistrée dans : {COMPARISON_PATH}")
    return model_metrics


if __name__ == "__main__":
    run_comparison()
