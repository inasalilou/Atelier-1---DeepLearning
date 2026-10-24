import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "prices-split-adjusted.csv"
BEST_PARAMS_PATH = ROOT / "best_params.json"
FEATURE_COLUMNS = ["open", "high", "low", "volume"]
TARGET_COLUMN = "close"
SYMBOL = "AAPL"
SEED = 42


class RegressionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim et num_layers doivent être positifs.")

        layers: list[nn.Module] = []
        input_dim = len(FEATURE_COLUMNS)
        for _ in range(num_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_regression_data() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Placez prices-split-adjusted.csv dans le dossier data/."
        )

    frame = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "symbol"])
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Colonnes requises absentes du CSV : " + ", ".join(missing_columns)
        )

    # Filtrage sur le ticker pour un entraînement stable et rapide
    if "symbol" in frame.columns and SYMBOL in frame["symbol"].values:
        frame = frame[frame["symbol"] == SYMBOL].copy()

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date", kind="stable")

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    if len(frame) < 10:
        raise ValueError("Le dataset doit contenir au moins 10 observations.")

    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = frame[[TARGET_COLUMN]].to_numpy(dtype=np.float32)

    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Les données contiennent des valeurs infinies.")

    return features, target


def get_train_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, target = load_regression_data()
    return train_test_split(features, target, test_size=0.2, shuffle=False)


def run_grid_search() -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    features, target = load_regression_data()
    train_features, _, train_target, _ = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    print(f"Lancement du GridSearch sur {len(train_features)} échantillons ({SYMBOL})...")

    # max_epochs réduit à 15 pour une sélection rapide des meilleurs hyperparamètres
    net = NeuralNetRegressor(
        module=RegressionMLP,
        criterion=nn.MSELoss,
        max_epochs=15,
        train_split=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
    )

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("net", net),
        ]
    )

    # Grille représentative et ciblée
    parameter_grid = {
        "net__lr": [0.01, 0.001],
        "net__optimizer": [torch.optim.Adam],
        "net__module__hidden_dim": [32, 64],
        "net__module__num_layers": [2],
        "net__batch_size": [32],
    }

    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(train_features, train_target)

    best_params = {
        "lr": float(search.best_params_["net__lr"]),
        "optimizer": search.best_params_["net__optimizer"].__name__,
        "hidden_dim": int(search.best_params_["net__module__hidden_dim"]),
        "num_layers": int(search.best_params_["net__module__num_layers"]),
        "batch_size": int(search.best_params_["net__batch_size"]),
    }

    BEST_PARAMS_PATH.write_text(
        json.dumps(best_params, indent=2) + "\n", encoding="utf-8"
    )
    best_mse = -float(search.best_score_)
    print("\n--- Recherche terminée avec succès ---")
    print("Meilleurs hyperparamètres :", best_params)
    print(f"Meilleur MSE CV : {best_mse:.6f}")
    print(f"Configuration enregistrée dans : {BEST_PARAMS_PATH}")
    return best_params


if __name__ == "__main__":
    run_grid_search()