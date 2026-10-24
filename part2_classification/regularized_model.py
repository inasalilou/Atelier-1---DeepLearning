import os
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
PLOTS_PATH = ROOT / "plots"


class RegularizedClassificationMLP(nn.Module):
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
            layers.extend(
                (
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                )
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_regularized_model(
    *,
    input_dim: int = 6,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_classes: int = 6,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: torch.device | str | None = None,
) -> tuple[RegularizedClassificationMLP, torch.optim.Optimizer]:
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr doit être positif et weight_decay non négatif.")
    model = RegularizedClassificationMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    if device is not None:
        model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return model, optimizer


if __name__ == "__main__":
    os.makedirs(PLOTS_PATH, exist_ok=True)
    model, optimizer = build_regularized_model()
    print(model)
    print(f"Optimiseur : {optimizer.__class__.__name__}, weight_decay=1e-4")
