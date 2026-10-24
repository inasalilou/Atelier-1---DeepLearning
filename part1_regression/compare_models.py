import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from baseline_gridsearch import (
    FEATURE_COLUMNS,
    RegressionMLP,
    ROOT,
    load_regression_data,
)
from regularized_model import RegularizedRegressionMLP
from train_and_evaluate import get_best_params, train_model


PLOT_PATH = ROOT / "model_comparison.png"
SEED = 42


def run_comparison() -> dict[str, dict[str, list[float]]]:
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

    histories: dict[str, dict[str, list[float]]] = {}
    for name in ("Baseline", "Régularisé"):
        print(f"\n=== Entraînement du modèle {name} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model_arguments = {
            "hidden_dim": int(best_params["hidden_dim"]),
            "num_layers": int(best_params["num_layers"]),
        }
        if name == "Baseline":
            model = RegressionMLP(**model_arguments)
        else:
            model = RegularizedRegressionMLP(
                input_dim=len(FEATURE_COLUMNS), **model_arguments
            )
        histories[name] = train_model(
            model,
            train_features,
            train_target,
            test_features,
            test_target,
            epochs=100,
            batch_size=int(best_params["batch_size"]),
            lr=float(best_params["lr"]),
            optimizer_name=str(best_params["optimizer"]),
            weight_decay=1e-4 if name == "Régularisé" else 0.0,
        )

    epochs = np.arange(1, len(histories["Baseline"]["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name, history in histories.items():
        axes[0, 0].plot(epochs, history["train_loss"], label=name)
        axes[0, 1].plot(epochs, history["test_loss"], label=name)
        axes[1, 0].plot(epochs, history["train_r2"], label=name)
        axes[1, 1].plot(epochs, history["test_r2"], label=name)
    axes[0, 0].set(title="MSE Loss — Train", xlabel="Époque", ylabel="MSE")
    axes[0, 1].set(title="MSE Loss — Test", xlabel="Époque", ylabel="MSE")
    axes[1, 0].set(title="R² Score — Train", xlabel="Époque", ylabel="R²")
    axes[1, 1].set(title="R² Score — Test", xlabel="Époque", ylabel="R²")
    for ax in axes.flat:
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print("\n=== Métriques finales ===")
    print(
        f"{'Modèle':<15} {'Train MSE':>12} {'Test MSE':>12} "
        f"{'Train R²':>12} {'Test R²':>12} {'Gap généralisation':>20}"
    )
    print("-" * 87)
    for name, history in histories.items():
        train_loss = history["train_loss"][-1]
        test_loss = history["test_loss"][-1]
        print(
            f"{name:<15} {train_loss:>12.6f} {test_loss:>12.6f} "
            f"{history['train_r2'][-1]:>12.4f} "
            f"{history['test_r2'][-1]:>12.4f} "
            f"{test_loss - train_loss:>20.6f}"
        )
    print(f"\nCourbes enregistrées dans : {PLOT_PATH}")
    return histories


if __name__ == "__main__":
    run_comparison()
