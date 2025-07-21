import torch
import torch.nn as nn
import torch.distributions as dist

from src.ppo.ppo import PolicyNetwork, ValueNetwork


class BlockGamePolicyNet(PolicyNetwork):
    """
    Policy network for the BlockGameEnv using attention and token embeddings.

    The network takes an observation tensor of shape (batch_size, n_blocks, n_stacks)
    with integer token IDs (0 for empty, 1 to n_blocks for block IDs), applies
    self-attention, and outputs a probability distribution over possible token IDs
    for each grid position to form the target state.
    """

    def __init__(
        self,
        n_blocks: int,
        n_stacks: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """
        Initialize the policy network.

        Args:
            n_blocks: Number of blocks in the game.
            n_stacks: Number of stacks in the game.
            embed_dim: Dimension of the token embeddings.
            num_heads: Number of attention heads in the transformer.
            num_layers: Number of transformer layers.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self.seq_length = n_blocks * n_stacks
        self.vocab_size = n_blocks + 1  # Block IDs 0 (empty) to n_blocks
        # Token embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        # Positional encoding for (height, stack) positions
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, embed_dim) * 0.02, requires_grad=True
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predicts logits for each position's state
        self.output_head = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute logits for each grid position.

        Args:
            states: Tensor of shape (batch_size, n_blocks, n_stacks)
                    with integer token IDs (0 for empty, 1 to n_blocks).

        Returns:
            logits: Tensor of shape (batch_size, n_blocks, n_stacks, n_blocks + 1)
                    containing logits for each position's state.
        """
        batch_size = states.shape[0]
        # Flatten to sequence: (batch_size, n_blocks * n_stacks)
        token_ids = states.view(batch_size, self.seq_length)
        # Embed tokens
        x = self.token_embedding(token_ids)  # Shape: (batch_size, seq_length, embed_dim)
        # Add positional encoding
        x = x + self.positional_encoding
        # Apply transformer
        x = self.transformer(x)  # Shape: (batch_size, seq_length, embed_dim)
        # Compute logits for each position
        logits = self.output_head(x)  # Shape: (batch_size, seq_length, vocab_size)
        # Reshape back to grid
        logits = logits.view(batch_size, self.n_blocks, self.n_stacks, self.vocab_size)
        return logits

    def get_dist(self, states: torch.Tensor) -> dist.Distribution:
        """
        Get the action distribution for the given states.

        Args:
            states: Tensor of shape (batch_size, n_blocks, n_stacks, n_blocks + 1).

        Returns:
            dist.Independent: Distribution over the action space, combining
                             categorical distributions for each grid position.
        """
        logits = self.forward(states)  # Shape: (batch_size, n_blocks, n_stacks, vocab_size)
        # Create categorical distribution for each position
        probs = torch.softmax(logits, dim=-1)
        categorical_dist = dist.Categorical(probs=probs)
        # Combine distributions across grid positions
        # Use Independent to treat the grid as a single action
        return dist.Independent(
            categorical_dist, 2
        )  # Reinterpreted as 2D (n_blocks, n_stacks)


class BlockGameValueNet(ValueNetwork):
    """
    Value network for the BlockGameEnv, estimating the value of each state.

    Shares initial feature extraction with the policy network but outputs a scalar value.
    """

    def __init__(
        self,
        n_blocks: int,
        n_stacks: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """
        Initialize the value network.

        Args:
            n_blocks: Number of blocks in the game.
            n_stacks: Number of stacks in the game.
            embed_dim: Dimension of the token embeddings (must match policy).
            num_heads: Number of attention heads in the transformer.
            num_layers: Number of transformer layers.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self.seq_length = n_blocks * n_stacks
        self.vocab_size = n_blocks + 1
        # Token embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, embed_dim) * 0.02, requires_grad=True
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Value head: reduce to a single value
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * self.seq_length, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute state values.

        Args:
            states: Tensor of shape (batch_size, n_blocks, n_stacks) with token IDs.

        Returns:
            values: Tensor of shape (batch_size,) containing state values.
        """
        batch_size = states.shape[0]
        # Flatten to sequence
        token_ids = states.view(batch_size, self.seq_length)
        # Embed tokens
        x = self.token_embedding(token_ids)  # Shape: (batch_size, seq_length, embed_dim)
        # Add positional encoding
        x = x + self.positional_encoding
        # Apply transformer
        x = self.transformer(x)  # Shape: (batch_size, seq_length, embed_dim)
        # Flatten for value head
        x = x.view(batch_size, -1)  # Shape: (batch_size, seq_length * embed_dim)
        # Compute value
        values = self.value_head(x).squeeze(-1)  # Shape: (batch_size,)
        return values
