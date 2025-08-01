import torch
import torch.nn as nn
import torch.distributions as dist

from src.ppo.ppo import PolicyNetwork, ValueNetwork


class BlockGamePolicyNet(PolicyNetwork):
    """
    Policy network for the BlockGameEnv using encoder-decoder architecture with cross-attention.

    Architecture:
    - Encoder: Processes goal state (what we want to achieve)
    - Decoder: Processes current state while attending to encoded goal state
    - Cross-attention: Current state positions attend to goal state positions

    This is analogous to machine translation where:
    - Goal state = source language
    - Current state = target language being generated
    - Cross-attention learns "what in the goal relates to my current position"
    """

    def __init__(
        self,
        n_blocks: int,
        n_stacks: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
    ):
        """
        Initialize the policy network.

        Args:
            n_blocks: Number of blocks in the game.
            n_stacks: Number of stacks in the game.
            embed_dim: Dimension of the token embeddings.
            num_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers (for goal state).
            num_decoder_layers: Number of decoder layers (for current state).
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self.seq_length = n_blocks * n_stacks
        self.vocab_size = n_blocks + 1  # Block IDs 0 (empty) to n_blocks

        # Shared token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Positional encodings (same structure for both, but learned separately)
        self.current_state_pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, embed_dim) * 0.02, requires_grad=True
        )
        self.goal_state_pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, embed_dim) * 0.02, requires_grad=True
        )

        # Encoder: processes goal state with self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.goal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder: processes current state with self-attention + cross-attention to goal
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.current_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output head: predicts logits for each current state position
        self.output_head = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with encoder-decoder cross-attention.

        Args:
            states: Tensor of shape (batch_size, 2*n_blocks, n_stacks) with concatenated
                   current and goal states.

        Returns:
            logits: Tensor of shape (batch_size, n_blocks, n_stacks, n_blocks + 1)
                    containing logits for each position's state.
        """
        batch_size = states.shape[0]

        # Split concatenated observation
        current_state = states[:, : self.n_blocks, :]  # First n_blocks rows
        goal_state = states[:, self.n_blocks :, :]  # Next n_blocks rows

        # Flatten to sequences
        current_tokens = current_state.view(batch_size, self.seq_length)
        goal_tokens = goal_state.view(batch_size, self.seq_length)

        # Embed and add positional encodings
        current_embedded = (
            self.token_embedding(current_tokens) + self.current_state_pos_encoding
        )
        goal_embedded = self.token_embedding(goal_tokens) + self.goal_state_pos_encoding

        # Encoder: process goal state
        # Shape: (batch_size, seq_length, embed_dim)
        encoded_goal = self.goal_encoder(goal_embedded)

        # Decoder: process current state with cross-attention to encoded goal
        # TransformerDecoder expects (tgt, memory) where:
        # - tgt = current state (what we're generating actions for)
        # - memory = encoded goal state (what we're attending to)
        decoded_current = self.current_decoder(
            tgt=current_embedded,  # Current state positions
            memory=encoded_goal,  # Cross-attend to goal state
        )

        # Compute logits for each current state position
        logits = self.output_head(decoded_current)
        # Shape: (batch_size, seq_length, vocab_size)

        # Reshape back to grid
        logits = logits.view(batch_size, self.n_blocks, self.n_stacks, self.vocab_size)
        return logits

    def get_dist(self, states: torch.Tensor) -> dist.Distribution:
        """
        Get the action distribution for the given states.

        Args:
            states: Tensor of shape (batch_size, 2*n_blocks, n_stacks).

        Returns:
            dist.Independent: Distribution over the action space.
        """
        logits = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        categorical_dist = dist.Categorical(probs=probs)
        return dist.Independent(categorical_dist, 2)


class BlockGameValueNet(ValueNetwork):
    """
    Value network using encoder-decoder architecture with cross-attention.

    Similar to policy network but outputs a single value estimate.
    """

    def __init__(
        self,
        n_blocks: int,
        n_stacks: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
    ):
        """
        Initialize the value network.

        Args:
            n_blocks: Number of blocks in the game.
            n_stacks: Number of stacks in the game.
            embed_dim: Dimension of the token embeddings.
            num_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.embed_dim = embed_dim
        self.seq_length = n_blocks * n_stacks
        self.vocab_size = n_blocks + 1

        # Shared token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Positional encodings
        self.current_state_pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, embed_dim) * 0.02, requires_grad=True
        )
        self.goal_state_pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, embed_dim) * 0.02, requires_grad=True
        )

        # Encoder for goal state
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.goal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Decoder for current state with cross-attention to goal
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.current_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Value head: aggregate cross-attended features to single value
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * self.seq_length, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute state values using cross-attention.

        Args:
            states: Tensor of shape (batch_size, 2*n_blocks, n_stacks).

        Returns:
            values: Tensor of shape (batch_size,) containing state values.
        """
        batch_size = states.shape[0]

        # Split concatenated observation
        current_state = states[:, : self.n_blocks, :]
        goal_state = states[:, self.n_blocks :, :]

        # Flatten to sequences and embed
        current_tokens = current_state.view(batch_size, self.seq_length)
        goal_tokens = goal_state.view(batch_size, self.seq_length)

        current_embedded = (
            self.token_embedding(current_tokens) + self.current_state_pos_encoding
        )
        goal_embedded = self.token_embedding(goal_tokens) + self.goal_state_pos_encoding

        # Encode goal state
        encoded_goal = self.goal_encoder(goal_embedded)

        # Decode current state with cross-attention to goal
        decoded_current = self.current_decoder(tgt=current_embedded, memory=encoded_goal)

        # Aggregate to single value
        x = decoded_current.view(batch_size, -1)  # Flatten sequence
        values = self.value_head(x).squeeze(-1)

        return values
