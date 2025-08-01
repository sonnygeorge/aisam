import random
from functools import partial
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.block_game.env import BlockGameEnv
from src.block_game.models import BlockGamePolicyNet, BlockGameValueNet
from src.ppo.train import train_ppo, evaluate_policy


def get_sampled_action_or_curriculum_valid_action(
    env: BlockGameEnv,
    policy: BlockGamePolicyNet,
    state: torch.Tensor,
    previous_completed_timesteps: int,
    force_valid_action_starting_prob: float = 0.5,
    force_valid_action_min_prob: float = 0.0,
    force_valid_action_decay_rate: float = 1e-4,
    begin_decay_at: int = 100_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get sampled action or a curriculum-guided valid action."""

    if previous_completed_timesteps < begin_decay_at:
        curriculum_probability = force_valid_action_starting_prob
    else:
        curriculum_probability = (
            force_valid_action_starting_prob - force_valid_action_min_prob
        ) * np.exp(
            -force_valid_action_decay_rate * previous_completed_timesteps
        ) + force_valid_action_min_prob

    # if random.random() < 0.001:
    #     print(f"Curriculum probability: {curriculum_probability:.3f}, ")

    action_dist = policy.get_dist(state.unsqueeze(0))
    if random.random() < curriculum_probability:
        # Curriculum:
        # Use curriculum: sample from valid actions weighted by policy probabilities
        valid_actions_list = list(env.get_valid_actions())
        if not valid_actions_list:
            raise ValueError("A state with no valid actions should not be possible")
        valid_actions_tensor = torch.stack(
            [torch.tensor(action, dtype=torch.int32) for action in valid_actions_list]
        )
        # Get log probabilities for each valid action
        valid_log_probs = action_dist.log_prob(valid_actions_tensor)
        # Sample from valid actions weighted by their policy probabilities
        valid_probs = torch.softmax(valid_log_probs, dim=0)
        chosen_idx = torch.multinomial(valid_probs, 1).item()
        action = valid_actions_tensor[chosen_idx]
        log_prob = valid_log_probs[chosen_idx]
    else:
        # Sample normally from policy (may be an invalid action)
        action = action_dist.sample().squeeze(0)
        log_prob = action_dist.log_prob(action.unsqueeze(0)).squeeze(0)
    return action, log_prob


if __name__ == "__main__":
    n_blocks = 2
    n_stacks = 3

    env = BlockGameEnv(n_blocks=n_blocks, n_stacks=n_stacks, max_episode_length=18)
    policy = BlockGamePolicyNet(
        n_blocks=env.n_blocks,
        n_stacks=env.n_stacks,
        embed_dim=24,
        num_heads=4,
        num_decoder_layers=2,
        num_encoder_layers=2,
    )
    value_net = BlockGameValueNet(
        n_blocks=env.n_blocks,
        n_stacks=env.n_stacks,
        embed_dim=24,
        num_heads=4,
        num_decoder_layers=2,
        num_encoder_layers=2,
    )

    get_sampled_action = partial(
        get_sampled_action_or_curriculum_valid_action,
        force_valid_action_starting_prob=0.53,
        force_valid_action_min_prob=0.09,
        force_valid_action_decay_rate=3e-8,
        begin_decay_at=2_000_000,
    )

    start = datetime.now()
    print(f"Starting training at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    policy, value_net, rewards = train_ppo(
        env=env,
        policy=policy,
        value_net=value_net,
        total_timesteps=10_000_000,
        steps_per_update=1024,
        learning_rate=3e-4,
        ppo_epochs=10,
        save_interval=500_000,
        log_interval=50_000,
        save_path=f"blockgame_{n_blocks}:{n_stacks}_ppo.pt",
        get_sampled_action=get_sampled_action,
    )

    end = datetime.now()
    elapsed = end - start
    print(f"Training completed at {end.strftime('%Y-%m-%d %H:%M:%S')}")
    total_seconds = int(elapsed.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"Total training time: {hours}h {minutes}m {seconds}s")

    print("\nEvaluating trained policy...")
    evaluate_policy(env, policy, num_episodes=10)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Update")
    plt.ylabel("Average Episode Reward")
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.show()

    env.close()
