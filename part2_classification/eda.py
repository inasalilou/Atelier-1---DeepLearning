import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "predictive_maintenance.csv"
PLOTS_PATH = ROOT / "plots"
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def run_eda() -> None:
    os.makedirs(PLOTS_PATH, exist_ok=True)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Placez predictive_maintenance.csv dans le dossier data/."
        )

    frame = pd.read_csv(DATA_PATH)
    required_columns = {"UDI", "Product ID", "Target", "Failure Type", *SENSOR_COLUMNS}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError("Colonnes requises absentes du CSV : " + ", ".join(missing))

    frame = frame.drop(columns=["UDI", "Product ID", "Target"])
    print("=== Statistiques descriptives ===")
    print(frame.describe(include="all").to_string())
    print("\n=== Types de données ===")
    print(frame.dtypes.to_string())
    print("\n=== Valeurs nulles ===")
    print(frame.isnull().sum().to_string())
    class_counts = frame["Failure Type"].value_counts(dropna=False)
    print("\n=== Comptage des modalités de Failure Type ===")
    print(class_counts.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=class_counts.values,
        y=class_counts.index.astype(str),
        orient="h",
        ax=ax,
        color="steelblue",
    )
    ax.set(title="Distribution des classes de Failure Type", xlabel="Nombre", ylabel="Classe")
    fig.tight_layout()
    fig.savefig(PLOTS_PATH / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        frame[SENSOR_COLUMNS].corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Corrélation des capteurs continus")
    fig.tight_layout()
    fig.savefig(PLOTS_PATH / "sensors_correlation.png", dpi=150)
    plt.close(fig)
    print(f"\nGraphiques enregistrés dans : {PLOTS_PATH}")


if __name__ == "__main__":
    run_eda()
