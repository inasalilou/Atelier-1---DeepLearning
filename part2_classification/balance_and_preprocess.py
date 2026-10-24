import os
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "predictive_maintenance.csv"
PLOTS_PATH = ROOT / "plots"
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Failure Type"
SEED = 42


def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Load, split, scale, and balance predictive-maintenance classification data."""
    os.makedirs(PLOTS_PATH, exist_ok=True)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Placez predictive_maintenance.csv dans le dossier data/."
        )

    frame = pd.read_csv(DATA_PATH)
    required_columns = {
        "UDI",
        "Product ID",
        "Target",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError("Colonnes requises absentes du CSV : " + ", ".join(missing))

    frame = frame.drop(columns=["UDI", "Product ID", "Target"]).copy()
    type_mapping = {"L": 0, "M": 1, "H": 2}
    unexpected_types = sorted(set(frame["Type"].dropna()) - set(type_mapping))
    if unexpected_types:
        raise ValueError(
            "Valeurs inattendues dans la colonne Type : "
            + ", ".join(map(str, unexpected_types))
        )
    frame["Type"] = frame["Type"].map(type_mapping)
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if frame[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("Le dataset contient des valeurs nulles ou non numériques.")

    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(frame[TARGET_COLUMN])
    if len(label_encoder.classes_) != 6:
        raise ValueError(
            "La cible Failure Type doit contenir exactement 6 classes ; "
            f"{len(label_encoder.classes_)} trouvée(s)."
        )
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Les features contiennent des valeurs infinies.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    train_class_counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
    if np.any(train_class_counts <= 3):
        raise ValueError(
            "SMOTE(k_neighbors=3) exige au moins 4 observations d'entraînement "
            "dans chaque classe."
        )
    smote = SMOTE(random_state=SEED, k_neighbors=3)
    x_train_balanced, y_train_balanced = smote.fit_resample(
        x_train_scaled, y_train
    )
    return (
        x_train_balanced.astype(np.float32),
        x_test_scaled.astype(np.float32),
        y_train_balanced.astype(np.int64),
        y_test.astype(np.int64),
        label_encoder,
    )


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, encoder = get_data()
    print(f"Classes : {list(encoder.classes_)}")
    print(f"Train équilibré : {x_train.shape}, distribution {np.bincount(y_train)}")
    print(f"Test : {x_test.shape}, distribution {np.bincount(y_test)}")
