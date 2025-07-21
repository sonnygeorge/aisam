"""
Proximal Policy Optimization (PPO) implementation with Generalized Advantage Estimation (GAE).

This module provides a policy-agnostic PPO implementation that works with both discrete
and continuous action spaces. It includes GAE for advantage estimation and the standard
PPO clipped objective with value function and entropy regularization.
"""

from typing import Tuple, Union, Protocol
import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor


class PolicyProtocol(Protocol):
    """Protocol defining the interface for policies used with PPO."""

    def get_dist(self, states: Tensor) -> dist.Distribution:
        """Get the action distribution for given states."""
        ...

    def parameters(self):
        """Return policy parameters for optimization."""
        ...


class PolicyNetwork(nn.Module, PolicyProtocol):
    pass


class ValueNetworkProtocol(Protocol):
    """Protocol defining the interface for value networks used with PPO."""

    def __call__(self, states: Tensor) -> Tensor:
        """Compute state values for given states."""
        ...

    def parameters(self):
        """Return value network parameters for optimization."""
        ...


class ValueNetwork(nn.Module, ValueNetworkProtocol):
    pass


def compute_gae(
    rewards: Union[Tensor, list],
    values: Union[Tensor, list],
    next_values: Union[Tensor, list],
    dones: Union[Tensor, list],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE reduces variance in policy gradient estimates by using a bias-variance tradeoff
    controlled by the lambda parameter. The advantage function is computed as:
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        rewards: Reward values for each timestep [T]
        values: State value estimates V(s_t) for each timestep [T]
        next_values: State value estimates V(s_{t+1}) for each timestep [T]
        dones: Boolean flags indicating episode termination [T]
        gamma: Discount factor for future rewards (0 ≤ γ ≤ 1)
        lam: GAE lambda parameter controlling bias-variance tradeoff (0 ≤ λ ≤ 1)

    Returns:
        advantages: Normalized advantage estimates [T]
        returns: Value targets (advantages + values) [T]

    Note:
        Higher lambda values reduce bias but increase variance.
        Lambda = 0 gives TD(0), lambda = 1 gives Monte Carlo estimates.
    """
    # Ensure inputs are tensors with correct dtype
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    next_values = torch.as_tensor(next_values, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)

    advantages = []
    gae = 0.0

    # Compute GAE backwards through time
    for t in reversed(range(len(rewards))):
        # TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]

        # GAE recursion: A_t = δ_t + γλ(1-done)A_{t+1}
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32, device=rewards.device)

    # Compute returns as advantages + baseline values
    returns = advantages + values.detach().clone()

    # Normalize advantages for training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_ppo_loss(
    policy: PolicyProtocol,
    value_net: ValueNetworkProtocol,
    states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    epsilon: float = 0.2,
    c1: float = 0.5,
    c2: float = 0.01,
) -> Tensor:
    """
    Compute the PPO clipped surrogate loss.

    The PPO loss combines three components:
    1. Policy loss: Clipped surrogate objective to prevent large policy updates
    2. Value loss: MSE between predicted and target values
    3. Entropy bonus: Encourages exploration

    Args:
        policy: Policy network implementing PolicyProtocol
        value_net: Value network implementing ValueNetworkProtocol
        states: State observations [batch_size, state_dim]
        actions: Actions taken [batch_size, action_dim] or [batch_size] for discrete
        old_log_probs: Log probabilities from behavior policy [batch_size]
        advantages: Advantage estimates [batch_size]
        returns: Value targets [batch_size]
        epsilon: PPO clipping parameter (typical range: 0.1-0.3)
        c1: Value loss coefficient (typical range: 0.25-1.0)
        c2: Entropy bonus coefficient (typical range: 0.001-0.1)

    Returns:
        total_loss: Combined PPO loss for optimization

    Note:
        The clipping prevents the policy from changing too dramatically,
        which helps training stability.
    """
    # Get current policy distribution and log probabilities
    dist = policy.get_dist(states)
    log_probs = dist.log_prob(actions)

    # Handle continuous actions (sum over action dimensions)
    if log_probs.dim() > 1:
        log_probs = log_probs.sum(dim=-1)

    # Compute probability ratio: π_θ(a|s) / π_θ_old(a|s)
    ratio = torch.exp(log_probs - old_log_probs)

    # PPO clipped surrogate objective
    surr1 = ratio * advantages  # Unclipped objective
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages  # Clipped objective
    policy_loss = -torch.min(surr1, surr2).mean()  # Take minimum (pessimistic bound)

    # Value function loss (MSE)
    values = value_net(states)
    value_loss = ((values - returns) ** 2).mean()

    # Entropy bonus for exploration
    entropy = dist.entropy().mean()

    # Combined loss
    total_loss = policy_loss + c1 * value_loss - c2 * entropy

    return total_loss


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent destructively large policy updates. It alternates between
    collecting experience and updating the policy multiple times on the
    collected data.

    Attributes:
        policy: Policy network (discrete or continuous)
        value_net: Value function network
        optimizer: Adam optimizer for both networks
    """

    def __init__(
        self, policy: PolicyProtocol, value_net: ValueNetworkProtocol, lr: float = 3e-4
    ) -> None:
        """
        Initialize PPO agent.

        Args:
            policy: Policy network implementing PolicyProtocol
            value_net: Value network implementing ValueNetworkProtocol
            lr: Learning rate for Adam optimizer
        """
        self.policy = policy
        self.value_net = value_net

        # Combined optimizer for both policy and value networks
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=lr
        )

    def update(
        self,
        states: Union[Tensor, list],
        actions: Union[Tensor, list],
        old_log_probs: Union[Tensor, list],
        rewards: Union[Tensor, list],
        next_states: Union[Tensor, list],
        dones: Union[Tensor, list],
        epochs: int = 10,
    ) -> None:
        """
        Update policy and value networks using collected experience.

        This method performs multiple epochs of optimization on the same batch
        of experience data, which is a key feature of PPO that improves sample
        efficiency compared to other policy gradient methods.

        Args:
            states: State observations [batch_size, state_dim]
            actions: Actions taken [batch_size, action_dim] or [batch_size] for discrete
            old_log_probs: Log probabilities from behavior policy [batch_size]
            rewards: Rewards received [batch_size]
            next_states: Next state observations [batch_size, state_dim]
            dones: Episode termination flags [batch_size]
            epochs: Number of optimization epochs on the data

        Note:
            The same batch of data is used for multiple epochs, which is safe
            due to the clipped objective that prevents large policy changes.
        """
        # Convert inputs to tensors
        states = torch.as_tensor(states)
        actions = torch.as_tensor(actions)
        old_log_probs = torch.as_tensor(old_log_probs)
        rewards = torch.as_tensor(rewards)
        next_states = torch.as_tensor(next_states)
        dones = torch.as_tensor(dones)

        # Compute current value estimates (detached to avoid gradients)
        with torch.no_grad():
            values = self.value_net(states)
            next_values = self.value_net(next_states)

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(rewards, values, next_values, dones)

        # Multiple epochs of optimization on the same data
        for _ in range(epochs):
            # Compute PPO loss
            loss = compute_ppo_loss(
                self.policy,
                self.value_net,
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class ExperienceBuffer:
    """Buffer to store experience for PPO training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, log_prob, reward, next_state, done):
        """Add a single experience tuple."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_batch(self):
        """Get all experiences as tensors."""
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards),
            torch.stack(self.next_states),
            torch.tensor(self.dones),
        )


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating how to use the PPO implementation.

    This example shows both discrete and continuous action space usage.
    In practice, you would collect this data from environment interaction.
    """

    # Example Policy Network for Discrete Actions (MLP with Categorical distribution)
    class DiscretePolicy(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=64):
            super(DiscretePolicy, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, state):
            logits = self.net(state)
            return logits

        def get_dist(self, state):
            logits = self.forward(state)
            return dist.Categorical(logits=logits)

        def get_action(self, state):
            dist = self.get_dist(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

    # Example Policy Network for Continuous Actions (MLP with Normal distribution)
    class ContinuousPolicy(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=64):
            super(ContinuousPolicy, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.log_std = nn.Parameter(
                torch.zeros(action_dim)
            )  # Learnable standard deviation

        def forward(self, state):
            mean = self.net(state)
            std = torch.exp(self.log_std)  # Ensure positive std
            return mean, std

        def get_dist(self, state):
            mean, std = self.forward(state)
            return dist.Normal(mean, std)

        def get_action(self, state):
            dist = self.get_dist(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(
                dim=-1
            )  # Sum log probs for multi-dimensional actions
            return action, log_prob

    # Example Value Network
    class MLPValueNetwork(nn.Module):
        def __init__(self, state_dim, hidden_dim=64):
            super(MLPValueNetwork, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state):
            value = self.net(state)
            return value.squeeze(-1)

    # Environment parameters
    state_dim = 4
    action_dim = 2
    batch_size = 100

    # Initialize networks
    discrete_policy = DiscretePolicy(state_dim, action_dim)
    continuous_policy = ContinuousPolicy(state_dim, action_dim)
    value_net = MLPValueNetwork(state_dim)

    # Create PPO agent (choose discrete or continuous policy)
    ppo_discrete = PPO(policy=discrete_policy, value_net=value_net)
    ppo_continuous = PPO(policy=continuous_policy, value_net=value_net)

    # Generate dummy experience data (replace with actual environment data)
    states = torch.randn(batch_size, state_dim)
    next_states = torch.randn(batch_size, state_dim)
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)  # No episodes end in this dummy data
    old_log_probs = torch.randn(batch_size)

    # Discrete actions example
    discrete_actions = torch.randint(0, action_dim, (batch_size,))
    ppo_discrete.update(
        states, discrete_actions, old_log_probs, rewards, next_states, dones, epochs=5
    )
    print("PPO discrete action update completed successfully.")

    # Continuous actions example
    continuous_actions = torch.randn(batch_size, action_dim)
    ppo_continuous.update(
        states, continuous_actions, old_log_probs, rewards, next_states, dones, epochs=5
    )
    print("PPO continuous action update completed successfully.")
