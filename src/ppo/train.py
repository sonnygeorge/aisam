from collections import deque
from typing import Protocol

import torch
import gymnasium as gym
import numpy as np

from src.ppo.ppo import PPO, PolicyNetwork, ValueNetwork, ExperienceBuffer


class GetSampledAction(Protocol):
    def __call__(
        self,
        env: gym.Env,
        policy: PolicyNetwork,
        state: torch.Tensor,
        prev_completed_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets sampled action and associated logprob as a tuple."""
        pass


def default_get_sampled_action(
    env: gym.Env,
    policy: PolicyNetwork,
    state: torch.Tensor,
    prev_completed_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla implementation to sample an action from the policy."""
    action_dist = policy.get_dist(state.unsqueeze(0))
    action = action_dist.sample().squeeze(0)
    log_prob = action_dist.log_prob(action.unsqueeze(0)).squeeze(0)
    return action, log_prob


def collect_experience(
    env: gym.Env,
    policy: PolicyNetwork,
    buffer: ExperienceBuffer,
    prev_completed_timesteps: int = 0,
    steps_to_collect: int = 1028,
    get_sampled_action: GetSampledAction = default_get_sampled_action,
) -> float:
    policy.eval()  # Set to evaluation mode
    episode_avg_rewards = []
    current_episode_step_rewards = []

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.int32)

    for step in range(steps_to_collect):
        # Sanmple action from policy
        with torch.no_grad():
            action, log_prob = get_sampled_action(
                env, policy, state, prev_completed_timesteps + step
            )
        # Take step in environment
        next_state, reward, done, truncated, _ = env.step(action.numpy())
        current_episode_step_rewards.append(reward)
        # Store experience
        buffer.add(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            next_state=torch.tensor(next_state, dtype=torch.int32),
            done=done or truncated,
        )
        # Update state
        state = torch.tensor(next_state, dtype=torch.int32)
        # Reset environment if episode ended
        if done or truncated:
            episode_avg_rewards.append(np.mean(current_episode_step_rewards))
            current_episode_step_rewards = []
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.int32)

    return np.mean(episode_avg_rewards) if episode_avg_rewards else 0.0


def train_ppo(
    env: gym.Env,
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    total_timesteps: int = 1_000_000,
    steps_per_update: int = 1028,
    learning_rate: float = 3e-4,
    ppo_epochs: int = 10,
    save_interval: int = 50_000,
    log_interval: int = 10_000,
    save_path: str = "ppo_checkpoint.pt",
    get_sampled_action: GetSampledAction = default_get_sampled_action,
):
    ppo_agent = PPO(policy, value_net, lr=learning_rate)
    buffer = ExperienceBuffer()
    avg_episode_avg_rewards = deque(maxlen=1_000_000)  # Store last 1M episode rewards
    completed_timesteps = 0
    update_count = 0
    last_save_interval = 0
    last_log_interval = 0

    while completed_timesteps < total_timesteps:
        avg_episode_avg_reward = collect_experience(
            env=env,
            policy=policy,
            buffer=buffer,
            prev_completed_timesteps=completed_timesteps,
            steps_to_collect=steps_per_update,
            get_sampled_action=get_sampled_action,
        )
        avg_episode_avg_rewards.append(avg_episode_avg_reward)

        # Update policy using PPO
        states, actions, log_probs, rewards, next_states, dones = buffer.get_batch()
        policy.train()  # Set to training mode
        ppo_agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            epochs=ppo_epochs,
        )
        buffer.reset()

        # Update counters
        completed_timesteps += steps_per_update
        update_count += 1

        # Logging
        if completed_timesteps - last_log_interval >= log_interval:
            last_log_interval = last_log_interval + log_interval
            avg_100_reward = (
                np.mean(avg_episode_avg_rewards) if avg_episode_avg_rewards else 0
            )
            print(
                f"Timesteps: {completed_timesteps:,} | "
                f"Updates: {update_count} | "
                f"Episode avg reward (avg over 100 eps): {avg_100_reward:.2f}"
            )

        # Save model
        if completed_timesteps - last_save_interval >= save_interval:
            last_save_interval = last_save_interval + save_interval
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "value_net_state_dict": value_net.state_dict(),
                    "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
                    "timesteps": completed_timesteps,
                    "episode_rewards": list(avg_episode_avg_rewards),
                },
                save_path.rstrip(".pt") + f"_{completed_timesteps}.pt",
            )
            print(f"Model saved at timestep {completed_timesteps}")

    print("Training completed!")

    # Save final model
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
            "optimizer_state_dict": ppo_agent.optimizer.state_dict(),
            "timesteps": completed_timesteps,
            "episode_rewards": list(avg_episode_avg_rewards),
        },
        save_path.rstrip(".pt") + f"_{completed_timesteps}_final.pt",
    )

    return policy, value_net, avg_episode_avg_rewards


def evaluate_policy(
    env: gym.Env, policy: PolicyNetwork, num_episodes: int = 10, max_steps: int = 100
):
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    episodes_truncated = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.int32)
        episode_reward = 0
        episode_length = 0
        done = False

        for step in range(max_steps):
            with torch.no_grad():
                action_dist = policy.get_dist(state.unsqueeze(0))
                action = action_dist.sample().squeeze(0)

            next_state, reward, done, truncated, _ = env.step(action.numpy())
            episode_reward += reward
            episode_length += 1
            state = torch.tensor(next_state, dtype=torch.int32)

            if done or truncated:
                break
        else:
            # This executes if the for loop completed without breaking (hit max_steps)
            episodes_truncated += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        status = (
            " (truncated)" if episode_length == max_steps and not (done or truncated) else ""
        )
        print(
            f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}{status}"
        )

    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Episodes truncated due to max_steps: {episodes_truncated}/{num_episodes}")
