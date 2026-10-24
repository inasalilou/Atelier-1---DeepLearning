from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "prices-split-adjusted.csv"
PLOT_DIR = ROOT / "plots"
PRICE_COLUMNS = ["open", "high", "low", "volume", "close"]


def load_prices() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Placez prices-split-adjusted.csv dans le dossier data/."
        )

    frame = pd.read_csv(DATA_PATH)
    required = {"date", "close", *PRICE_COLUMNS}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Colonnes requises absentes du CSV : " + ", ".join(missing_columns)
        )
    return frame


def run_eda() -> None:
    prices = load_prices()

    print("=== Aperçu ===")
    print(prices.head())
    print("\n=== Statistiques descriptives ===")
    print(prices.describe(include="all").transpose())
    print("\n=== Types des colonnes ===")
    print(prices.dtypes)
    print("\n=== Valeurs manquantes par colonne ===")
    print(prices.isna().sum())

    if "symbol" in prices.columns:
        selected = prices.loc[prices["symbol"].astype(str).str.upper() == "AAPL"].copy()
        if selected.empty:
            raise ValueError("Aucune ligne AAPL trouvée dans le dataset.")
    else:
        print("\nColonne 'symbol' absente ; les graphiques utilisent toutes les lignes.")
        selected = prices.copy()

    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected["close"] = pd.to_numeric(selected["close"], errors="coerce")
    selected = selected.dropna(subset=["date", "close"]).sort_values("date")
    if selected.empty:
        raise ValueError("Aucune observation valide disponible pour le graphique.")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(selected["date"], selected["close"], linewidth=1)
    ax.set(title="AAPL — prix de clôture", xlabel="Date", ylabel="Prix de clôture")
    fig.tight_layout()
    close_path = PLOT_DIR / "close_price_aapl.png"
    fig.savefig(close_path, dpi=150)
    plt.close(fig)

    correlation_data = selected.reindex(columns=PRICE_COLUMNS).apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if correlation_data.empty:
        raise ValueError("Aucune ligne complète disponible pour calculer la corrélation.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation_data.corr(),
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Matrice de corrélation — AAPL")
    fig.tight_layout()
    correlation_path = PLOT_DIR / "price_correlation_aapl.png"
    fig.savefig(correlation_path, dpi=150)
    plt.close(fig)

    print(f"\nGraphiques enregistrés :\n- {close_path}\n- {correlation_path}")


if __name__ == "__main__":
    run_eda()
