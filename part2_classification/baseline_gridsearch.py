import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from skorch import NeuralNetClassifier
from torch import nn

from balance_and_preprocess import get_data


ROOT = Path(__file__).resolve().parent
PLOTS_PATH = ROOT / "plots"
BEST_PARAMS_PATH = ROOT / "best_params_classification.json"
SEED = 42


class ClassificationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, num_classes) < 1:
            raise ValueError("Les dimensions du réseau doivent être positives.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def run_grid_search() -> dict[str, Any]:
    os.makedirs(PLOTS_PATH, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    x_train, _, y_train, _, _ = get_data()

    # Sous-échantillonnage représentatif pour accélérer le GridSearch sur CPU
    if len(x_train) > 8000:
        x_search, _, y_search, _ = train_test_split(
            x_train, y_train, train_size=8000, stratify=y_train, random_state=SEED
        )
    else:
        x_search, y_search = x_train, y_train

    print(f"Lancement du GridSearchCV sur {len(x_search)} échantillons équilibrés...")

    net = NeuralNetClassifier(
        module=ClassificationMLP,
        criterion=nn.CrossEntropyLoss,
        max_epochs=10,
        train_split=None,
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    parameter_grid = {
        "lr": [0.01, 0.005],
        "optimizer": [torch.optim.Adam],
        "module__hidden_dim": [64, 128],
        "module__num_layers": [2],
        "batch_size": [128],
    }

    search = GridSearchCV(
        estimator=net,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_search, y_search)

    best_params = {
        "lr": float(search.best_params_["lr"]),
        "optimizer": search.best_params_["optimizer"].__name__,
        "hidden_dim": int(search.best_params_["module__hidden_dim"]),
        "num_layers": int(search.best_params_["module__num_layers"]),
        "batch_size": int(search.best_params_["batch_size"]),
    }

    BEST_PARAMS_PATH.write_text(
        json.dumps(best_params, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== Recherche terminée ===")
    print("Meilleurs paramètres :", best_params)
    print(f"Meilleur F1 macro CV : {search.best_score_:.4f}")
    print(f"Paramètres enregistrés dans : {BEST_PARAMS_PATH}")
    return best_params


if __name__ == "__main__":
    run_grid_search()