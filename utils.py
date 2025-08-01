import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class SinPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attention_heads: int,
        ff_hidden_dim: int | None = None,
        ff_output_dim: int | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        # Default ff_hidden_dim to 4 * embed_dim if not specified
        ff_hidden_dim = 4 * embed_dim if ff_hidden_dim is None else ff_hidden_dim
        # Default ff_output_dim to embed_dim if not specified
        ff_output_dim = embed_dim if ff_output_dim is None else ff_output_dim

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim, num_attention_heads, dropout=dropout_rate, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, ff_output_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, input_tensor, attention_mask=None):
        # Self-attention with residual connection
        attention_output, _ = self.multihead_attention(
            input_tensor, input_tensor, input_tensor, attn_mask=attention_mask
        )
        normed_attention_output = self.layer_norm1(input_tensor + attention_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward_network(normed_attention_output)
        final_output = self.layer_norm2(normed_attention_output + ff_output)

        return final_output


def apply_speculative_update(
    original_model: nn.Module, original_optimizer: optim.Optimizer, from_loss: torch.Tensor
) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Applies a speculative update of a PyTorch model after a forward pass.

    This function computes gradients and applies them to a copy of the model, allowing
    you to preview what the model would look like after the update without modifying the
    original model.

    Args:
        original_model: The original PyTorch model
        original_optimizer: The optimizer for the original model
        from_loss: The loss computed from a previous forward pass output

    Returns:
        tuple: (speculative_model, speculative_optimizer) - The updated model copy and
            its optimizer
    """
    # Step 1: Compute gradients on the original model
    # This populates .grad attributes on the original model's parameters
    from_loss.backward(retain_graph=True)  # retain_graph=True to allow multiple backward passes

    # Step 2: Create a deep copy of the model for speculative update
    speculative_model = copy.deepcopy(original_model)

    # Step 3: Copy gradients from original model to the speculative model
    # This is necessary because the computational graph was built with the original model
    for (name, param), (name_spec, param_spec) in zip(
        original_model.named_parameters(), speculative_model.named_parameters()
    ):
        if param.grad is not None:
            param_spec.grad = param.grad.clone()

    # Step 4: Create a new optimizer for the speculative model
    # We need to match the optimizer type and parameters
    optimizer_type = type(original_optimizer)
    optimizer_state = original_optimizer.state_dict()
    speculative_optimizer = optimizer_type(
        speculative_model.parameters(), lr=optimizer_state["param_groups"][0]["lr"]
    )

    # Step 5: Apply the speculative update
    speculative_optimizer.step()

    # Step 6: Clear gradients on the original model
    # This ensures the original model is ready for its own update
    original_optimizer.zero_grad()

    return speculative_model, speculative_optimizer


if __name__ == "__main__":

    def test_apply_speculative_update() -> None:
        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.output = nn.Linear(5, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.linear(x))
                return self.output(x)

        model: nn.Module = SimpleModel()
        optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Store original parameters for comparison
        original_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        # Run forward pass
        input_data: torch.Tensor = torch.randn(32, 10)
        output: torch.Tensor = model(input_data)

        # Compute two different losses from the same output
        target1: torch.Tensor = torch.randn(32, 1)
        target2: torch.Tensor = torch.randn(32, 1)
        loss_for_speculative_update: torch.Tensor = nn.MSELoss()(output, target1)
        loss_for_final_update: torch.Tensor = nn.MSELoss()(output, target2)

        # Create speculative update
        speculative_model, _ = apply_speculative_update(
            model, optimizer, loss_for_speculative_update
        )

        # Verify speculative model has updated parameters
        speculative_updated = False
        for (name, param), (name_spec, param_spec) in zip(
            model.named_parameters(), speculative_model.named_parameters()
        ):
            if not torch.allclose(param, param_spec):
                speculative_updated = True
        assert speculative_updated, "Speculative model parameters should have been updated"

        # Verify original model parameters are unchanged
        for name, param in model.named_parameters():
            assert torch.allclose(
                param, original_params[name]
            ), f"Original model parameter '{name}' should not have changed"

        # Store speculative params BEFORE original model's update
        speculative_params_before = {
            name: param.clone().detach() for name, param in speculative_model.named_parameters()
        }

        # Update original model
        loss_for_final_update.backward()
        optimizer.step()

        # Verify original model params changed
        original_updated = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, original_params[name]):
                original_updated = True
        assert original_updated, "Original model parameters should have been updated"

        # Verify speculative model params didn't change from the original model's update
        for name, param in speculative_model.named_parameters():
            assert torch.allclose(
                param, speculative_params_before[name]
            ), f"Speculative model parameter '{name}' should not have changed from original's update"

    test_apply_speculative_update()
