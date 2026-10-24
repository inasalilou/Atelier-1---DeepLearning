import torch
from torch import nn


class RegularizedRegressionMLP(nn.Module):
    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 32, num_layers: int = 2
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("input_dim, hidden_dim et num_layers doivent être positifs.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                (
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                )
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_regularized_model(
    *,
    input_dim: int = 4,
    hidden_dim: int = 32,
    num_layers: int = 2,
    lr: float = 0.001,
    optimizer_name: str = "Adam",
    weight_decay: float = 1e-4,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr doit être positif et weight_decay ne peut pas être négatif.")
    model = RegularizedRegressionMLP(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
    )
    criterion = nn.MSELoss()
    optimizers: dict[str, type[torch.optim.Optimizer]] = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }
    if optimizer_name not in optimizers:
        raise ValueError(
            f"Optimiseur non pris en charge : {optimizer_name}. "
            f"Choisissez parmi {', '.join(optimizers)}."
        )
    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return model, criterion, optimizer
