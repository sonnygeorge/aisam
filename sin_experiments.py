import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Callable, Dict, List
from datetime import datetime


def generate_sinusoidal_data(n_samples: int, frequency: float = 1, input_range: float = 1):
    """Generate data for a sinusoidal function with given frequency."""
    x = np.random.uniform(0, input_range, n_samples)
    y = np.sin(2 * np.pi * frequency * x)
    return x.reshape(-1, 1), y


class MLP(nn.Module):
    """A multi-layer perceptron with configurable layer structure and activation."""

    def __init__(self, hidden_dim: int, layers_config: List[nn.Module], activation: str = "relu"):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(layers_config)
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


def create_simple_mlp(hidden_dim: int, n_layers: int, **kwargs) -> MLP:
    """Create a SimpleMLP with specified number of layers."""
    layers = [nn.Linear(1, hidden_dim)]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    layers.append(nn.Linear(hidden_dim, 1))
    return MLP(hidden_dim, layers, **kwargs)


def create_recursive_mlp(
    hidden_dim: int, n_inner_layers: int = 2, n_recursions: int = 2, **kwargs
) -> MLP:
    """Create an MLP with recursive inner layers."""
    layers = [nn.Linear(1, hidden_dim)]
    inner_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_inner_layers)]
    layers.extend(inner_layers * n_recursions)
    layers.append(nn.Linear(hidden_dim, 1))
    return MLP(hidden_dim, layers, **kwargs)


def train_model(X_train, y_train, X_val, y_val, model_constructor: Callable, **model_kwargs):
    """Train a model and return the validation loss."""
    model = model_constructor(**model_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    best_val_loss = float("inf")
    patience, max_epochs = 100, 1000

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


def train_model_n_times_and_min_loss(
    X_train, y_train, X_val, y_val, model_constructor: Callable, n_runs: int = 5, **model_kwargs
):
    """Run train_model multiple times and return all validation losses."""
    losses = [
        train_model(X_train, y_train, X_val, y_val, model_constructor, **model_kwargs)
        for _ in range(n_runs)
    ]
    return losses  # Return all losses instead of just the minimum


def plot_results(
    frequencies: List[float],
    x_values: range,
    results: Dict[float, List[List[float]]],  # Updated to handle list of losses per config
    success_threshold: float = 0.01,
    x_label: str = "Number of Layers",
    title_prefix: str = "Layer Requirements",
    n_runs: int = 5,
    extra_info: str = "",
):
    """Plot results with min loss and range of losses for different frequencies."""
    fig, axes = plt.subplots(1, len(frequencies), figsize=(4 * len(frequencies), 4))
    if len(frequencies) == 1:
        axes = [axes]

    for i, freq in enumerate(frequencies):
        losses = results[freq]  # List of lists: losses for each x_value
        min_losses = [min(run_losses) for run_losses in losses]  # Minimum loss for each x_value
        max_losses = [max(run_losses) for run_losses in losses]  # Maximum loss for each x_value
        colors = ["green" if min_loss < success_threshold else "red" for min_loss in min_losses]

        # Plot min losses as points and lines
        axes[i].plot(x_values, min_losses, "o-", color="gray", alpha=0.3, label="Min Loss")
        for j, x in enumerate(x_values):
            axes[i].scatter(x, min_losses[j], c=colors[j], s=80)

        # Plot error bars to show the range of losses (min to max)
        axes[i].errorbar(
            x_values,
            min_losses,
            yerr=[
                np.zeros(len(min_losses)),
                [max_loss - min_loss for max_loss, min_loss in zip(max_losses, min_losses)],
            ],
            fmt="none",
            ecolor="blue",
            alpha=0.3,
            capsize=3,
            label="Loss Range",
        )

        axes[i].axhline(
            y=success_threshold, color="blue", linestyle="--", alpha=0.5, label="Success threshold"
        )
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel("Validation Loss (log scale)")
        axes[i].set_yscale("log")
        axes[i].set_title(f"Frequency = {freq} Hz")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(list(x_values))
        axes[i].set_ylim(
            min(min(min_losses + [success_threshold]) * 0.5, success_threshold * 0.5),
            max(max_losses) * 2,
        )
        axes[i].legend()

    plt.tight_layout()
    plt.suptitle(
        f"{title_prefix} vs. Sinusoidal Frequency (Min and Range of {n_runs} Runs{extra_info})",
        y=1.05,
        fontsize=12,
    )

    # Save plot with datetime in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title_prefix.lower().replace(' ', '_')}_{timestamp}.png"
    plt.savefig(filename, bbox_inches="tight")

    print(f"\nSummary of Minimum Viable {x_label}:")
    for freq in frequencies:
        min_losses = [min(run_losses) for run_losses in results[freq]]
        successful = [i + 1 for i, loss in enumerate(min_losses) if loss < success_threshold]
        min_value = min(successful) if successful else "None"
        print(f"Frequency {freq} Hz: Minimum {x_label.lower()} = {min_value}")


def plot_requirements(
    frequencies: List[float],
    model_constructor: Callable,
    max_value: int,
    n_samples: int = 1000,
    n_runs: int = 5,
    x_label: str = "Number of Layers",
    title_prefix: str = "Layer Requirements",
    **model_kwargs,
):
    """Evaluate and plot requirements for different frequencies."""
    param_name = "n_layers" if model_constructor == create_simple_mlp else "n_recursions"

    results = {}
    for freq in frequencies:
        X, y = generate_sinusoidal_data(n_samples, frequency=freq)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        results[freq] = []
        for value in range(1, max_value + 1):
            losses = train_model_n_times_and_min_loss(
                X_train,
                y_train,
                X_val,
                y_val,
                model_constructor,
                n_runs=n_runs,
                **{**model_kwargs, param_name: value},
            )
            results[freq].append(losses)  # Store all losses for this configuration
            print(
                f"Freq {freq} Hz, {x_label.lower()}: {value} Min Loss = {min(losses):.6f}, Range = [{min(losses):.6f}, {max(losses):.6f}]"
            )

    extra_info = (
        f", {model_kwargs.get('n_inner_layers', '')} Inner Layers"
        if "n_inner_layers" in model_kwargs
        else ""
    )
    plot_results(
        frequencies,
        range(1, max_value + 1),
        results,
        x_label=x_label,
        title_prefix=title_prefix,
        n_runs=n_runs,
        extra_info=extra_info,
    )


if __name__ == "__main__":
    frequencies = [2, 5, 9]
    print("Running Layer Requirements Experiment...")
    plot_requirements(
        frequencies,
        create_simple_mlp,
        max_value=11,
        n_runs=4,
        x_label="Number of Layers",
        title_prefix="Layer Requirements",
        hidden_dim=64,
        activation="silu",
    )
    print("\nRunning Recursion Requirements Experiment...")
    plot_requirements(
        frequencies,
        create_recursive_mlp,
        max_value=5,
        n_runs=4,
        x_label="Number of Recursions",
        title_prefix="Recursion Requirements",
        hidden_dim=64,
        n_inner_layers=2,
        activation="silu",
    )
