import warnings
from typing import TypeAlias, Type, Literal

import numpy as np
import torch.distributions as dist
import torch
import torch.nn as nn

import gymnasium as gym

from utils import SinPositionalEmbedding, TransformerBlock, apply_speculative_update


# TODO: Practice explaining what's going on to others & simplify
# TODO: Think about why the value function can just be cosine similarlity... (initialize NN as cosine similarity?)
# TODO: Frame exploration/hill-climb of recursive stop space as genetic algorithm?
# TODO: Perform updates to the value function and latent dynamics model
# TODO: Env encoder
# TODO: Freeze different things during training?


# LatentEnvRepresentation
# - shape: (latent_space_dim)
LatentEnvRepresentation: TypeAlias = torch.Tensor
# LatentEnvRepresentationBatch
# - shape: (batch_size, latent_space_dim)
LatentEnvRepresentationBatch: TypeAlias = torch.Tensor

# GoalBuffer
# - shape: (goal_buffer_size, latent_space_dim)
# - buffer order: lower-order -> higher-order
GoalBuffer: TypeAlias = torch.Tensor
# GoalBufferBatch
# - shape: (batch_size, goal_buffer_size, latent_space_dim)
# - buffer order: lower-order -> higher-order
GoalBufferBatch: TypeAlias = torch.Tensor


class InnerGoalInferrer(nn.Module):
    def __init__(
        self,
        latent_space_dim: int,
        goal_buffer_size: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.pos_emb = SinPositionalEmbedding(latent_space_dim, goal_buffer_size + 1)
        self.middle_layers = nn.ModuleList(
            [
                TransformerBlock(latent_space_dim, num_heads, dropout_rate=dropout_rate)
                for _ in range(num_layers - 1)
            ]
        )
        self.final_layer = TransformerBlock(
            latent_space_dim,
            num_heads,
            ff_output_dim=latent_space_dim + 1,  # +1 for stop probability
            dropout_rate=dropout_rate,
        )

    def forward(
        self, env_state: LatentEnvRepresentationBatch, goal_buffer: GoalBufferBatch
    ) -> torch.Tensor:
        x = torch.cat((env_state, goal_buffer), dim=-1)  # Add env state to goal buffer
        x = self.pos_emb(x)
        for layer in self.middle_layers:
            x = layer(x)
        x = self.final_layer(x)
        x[:, -1] = torch.sigmoid(x[:, -1])  # Normalize stop probability to [0, 1]
        return x


class RecursiveGoalInferrer(nn.Module):
    should_assign_credit_for_stop_probs: bool = False
    target_recursion_stop_levels_of_batch: torch.Tensor | None = (
        None  # Shape: (batch_size,) DType: torch.int64 or torch.int32)
    )

    def __init__(
        self,
        latent_space_dim: int,
        goal_buffer_size: int = 5,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        recursion_limit: int = 10,
    ):
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.goal_buffer_size = goal_buffer_size
        self.recursion_limit = recursion_limit
        self.inner_inferrer = InnerGoalInferrer(
            latent_space_dim,
            goal_buffer_size,
            num_heads,
            num_layers,
            dropout_rate,
        )

    def get_stop_prob_gradient(
        self,
        recursion_level: int,
        pred_stop_probs: torch.Tensor,  # Shape: (batch_size,)
    ) -> torch.Tensor:
        if self.should_assign_credit_for_stop_probs:
            target_probs = (self.target_recursion_stop_levels_of_batch == recursion_level).float()
            # ∂(MSE)/∂(y_pred) = 2 * (y_pred - y_true) / N
            return 2 * (pred_stop_probs - target_probs) / pred_stop_probs.shape[0]
        else:
            return torch.full_like(pred_stop_probs, 0.0, dtype=pred_stop_probs.dtype)

    def get_stop_prob_remover(self, recursion_level: int) -> Type[torch.autograd.Function]:
        """Returns a custom autograd function that:
        1. Removes stop prob from Inner goal inferrer output tensor
        2. Dynamically injects a gradient for the removed stop prob based on recursion level
        """

        class StopProbRemover(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor) -> torch.Tensor:
                """Removes stop prob from inner goal inferrer output tensor"""
                ctx.save_for_backward(x[:, -1])  # Shape: (batch_size,)
                return x[:, :-1]  # Remove stop probability

            @staticmethod
            def backward(
                ctx, grad_output: torch.Tensor
            ) -> (
                torch.Tensor
            ):  # Will grad_output be a tensor of shape (batch_size, latent_space_dim)?
                """Injects gradient for the removed stop prob based on recursion level"""
                (stop_probs,) = ctx.saved_tensors  # Shape: (batch_size,)
                stop_prob_grad = self.get_stop_prob_gradient(
                    recursion_level=recursion_level, pred_stop_probs=stop_probs
                )  # Shape: (batch_size,)
                return torch.cat((grad_output, stop_prob_grad.unsqueeze(1)), dim=1)

        return StopProbRemover

    def forward(
        self,
        env_state: LatentEnvRepresentation,
        goal: LatentEnvRepresentation,
        fixed_num_recursions: int | None = None,
        num_recursions_past_stop: int = 0,
    ) -> tuple[list[LatentEnvRepresentation], dict[int, float]]:

        def _infer_one(recursion_level: int) -> float:
            env_state_batch = env_state.unsqueeze(0)
            goal_buffer_batch = goal_buffer.unsqueeze(0)
            out = self.inner_inferrer(env_state_batch, goal_buffer_batch).squeeze(0)
            stop_probs_by_level[recursion_level] = float(out[-1])
            StopProbRemover = self.get_stop_prob_remover(recursion_level)
            goal = StopProbRemover.apply(out)
            goal_hierarchy.append(goal)
            goal_buffer[1:] = goal_buffer[:-1]  # Shift right
            goal_buffer[0] = goal  # Add to front

        goal_buffer = torch.zeros(self.goal_buffer_size, self.latent_space_dim)
        goal_buffer[0] = goal
        goal_hierarchy = [goal]
        stop_probs_by_level = {}

        recurse_until = fixed_num_recursions or self.recursion_limit
        for recursion_lvl in range(1, recurse_until + 1):
            stop_prob = _infer_one(recursion_lvl)
            if fixed_num_recursions is None and stop_prob < 0.5:
                break

        if fixed_num_recursions is not None:
            if num_recursions_past_stop != 0:
                warnings.warn("Cannot surpass a specified fixed number of recursions.")
            num_recursions_past_stop = 0

        for extra_recursion_lvl in range(1, num_recursions_past_stop + 1):
            _infer_one(recursion_lvl + extra_recursion_lvl)

        return goal_hierarchy, stop_probs_by_level


class GoalToActionDistributionDecoder(nn.Module):
    def __init__(
        self,
        latent_space_dim: int,
        action_space_dim: int,
        action_space_type: Literal["continuous", "categorical"] = "categorical",
        hidden_dim: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_space_type = action_space_type
        self.action_dim = action_space_dim
        hidden_dim = hidden_dim or latent_space_dim * 3
        if action_space_type == "continuous":
            output_dim = action_space_dim * 2  # Continuous: output mean and log_std;
        else:
            output_dim = action_space_dim  # Categorical: output logits
        layers = []
        last_dim = latent_space_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
            last_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, goal: LatentEnvRepresentationBatch) -> torch.Tensor:
        """Maps latent goal representation(s) to action distribution parameters."""
        return self.network(goal)

    def get_dist_from_output(self, output: torch.Tensor) -> dist.Distribution:
        """Creates distribution(s) from output(s) of the MLP."""
        if self.action_space_type == "continuous":
            mean, log_std = torch.chunk(output, 2, dim=1)
            mean = torch.tanh(mean)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            return dist.Normal(mean, std)
        else:
            return dist.Categorical(logits=output)

    def sample_from_output(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples action(s), returning action(s) and log probability(s)."""
        distribution = self.get_dist_from_output(output)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        if self.action_space_type == "continuous":
            log_prob = log_prob.sum(dim=1)
        return action, log_prob


class ValueFunctionModel(nn.Module):
    # THOUGHT: Why wouldn't value function be cosine similarity between state and goal?
    def __init__(self, latent_space_dim: int, hidden_dim: int | None = None, num_layers: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or latent_space_dim * 3
        layers = []
        last_dim = latent_space_dim * 2
        for _ in range(num_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
            last_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self, state: LatentEnvRepresentationBatch, goal: LatentEnvRepresentationBatch
    ) -> torch.Tensor:
        return self.network(state).squeeze(-1)  # Shape: (batch_size,)


class EnvEncoder(nn.Module):
    pass


class ActionConditionedLatentDynamicsModel(nn.Module):
    def __init__(
        self,
        latent_space_dim: int,
        action_space_dim: int,
        action_space_type: Literal["continuous", "categorical"] = "categorical",
        hidden_dim: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_space_type = action_space_type
        self.action_dim = action_space_dim
        hidden_dim = hidden_dim or latent_space_dim * 3
        layers = []
        last_dim = latent_space_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
            last_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_space_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: LatentEnvRepresentationBatch,
        action: torch.Tensor,
    ):
        if self.action_space_type == "categorical":
            action = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat((state, action), dim=-1)
        return self.network(x)


def compute_advantage(
    method: Literal["gae", "bootstrap"],
    values: torch.Tensor | list,
    next_values: torch.Tensor | list,
    dones: torch.Tensor | list,
    rewards: torch.Tensor | list | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values, dtype=torch.float32)
    next_values = torch.as_tensor(next_values, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32) if rewards is not None else None

    # Compute advantages backwards through time
    advantages = []
    advantage = 0.0
    for t in reversed(range(len(values))):
        if method == "gae":
            # GAE: TD error δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            # GAE recursion: A_t = δ_t + γλ(1-done)A_{t+1}
            advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        else:  # method == "bootstrap"
            # Bootstrap: A_t = γV(s_{t+1})(1-done) - V(s_t)
            advantage = gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages, dtype=torch.float32, device=values.device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize
    returns = advantages + values.detach().clone()  # Returns = advantages + baselines
    return advantages, returns


def compute_clipped_surrogate_loss(
    old_log_probs: torch.Tensor | list,
    log_probs: torch.Tensor | list,
    advantages: torch.Tensor | list,
    clip_epsilon: float,
) -> torch.Tensor:
    old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32)
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    probability_ratios = torch.exp(log_probs - old_log_probs)
    surr1 = probability_ratios * advantages
    surr2 = torch.clamp(probability_ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    clipped_surrogate_loss = -torch.min(surr1, surr2).mean()
    return clipped_surrogate_loss


class RecursivePolicyExperienceBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.state_latents = []
        self.next_states = []
        self.next_state_latents = []
        self.rewards = []
        self.dones = []
        self.recursion_stop_levels = []
        # Actions
        self.actions_before_stop = []
        self.actions = []
        self.actions_after_stop = []
        # Log probabilities
        self.log_probs_before_stop = []
        self.log_probs = []
        self.log_probs_after_stop = []

    def add(
        self,
        state: torch.Tensor,
        state_latent: torch.Tensor,
        next_state: torch.Tensor,
        next_state_latent: torch.Tensor,
        reward: float,
        done: bool,
        recursion_stop_level: int,
        # Actions
        action_before_stop: torch.Tensor | None,
        action: torch.Tensor,
        action_after_stop: torch.Tensor,
        # Goal inference recursion stop probabilities
        log_prob_before_stop: float | None,
        log_prob: float,
        log_prob_after_stop: float,
    ):
        self.states.append(state)
        self.state_latents.append(state_latent)
        self.next_states.append(next_state)
        self.next_state_latents.append(next_state_latent)
        self.rewards.append(reward)
        self.dones.append(done)
        self.recursion_stop_levels.append(recursion_stop_level)
        # Actions
        self.actions_before_stop.append(action_before_stop)
        self.actions.append(action)
        self.actions_after_stop.append(action_after_stop)
        # Log probabilities
        self.log_probs_before_stop.append(log_prob_before_stop)
        self.log_probs.append(log_prob)
        self.log_probs_after_stop.append(log_prob_after_stop)


class RecursivePolicy:
    def __init__(
        self,
        env: gym.Env,
        is_discrete_action_space: bool,
        env_encoder: EnvEncoder,
        recursive_goal_inferrer: RecursiveGoalInferrer,
        action_decoder: GoalToActionDistributionDecoder,
        value_function: ValueFunctionModel,
        latent_dynamics_model: ActionConditionedLatentDynamicsModel,
    ):
        self.env = env
        self.is_discrete_action_space = is_discrete_action_space
        self.env_encoder = env_encoder
        self.recursive_goal_inferrer = recursive_goal_inferrer
        self.action_decoder = action_decoder
        self.value_function = value_function
        self.latent_dynamics_model = latent_dynamics_model
        self.latent_space_dim = recursive_goal_inferrer.latent_space_dim
        self.cur_goal_latent = torch.zeros(self.latent_space_dim, dtype=torch.float32)

    def collect_rollouts_for_rl_training(
        self,
        num_steps: int = 100_000,
    ) -> RecursivePolicyExperienceBuffer:
        # Helper function
        def process_step(state: torch.Tensor, state_latent: torch.Tensor) -> bool:
            """Steps the environment once and updates the experience buffer."""
            # Infer goal hierarchy recursively
            goal_hierarchy, stop_probs = self.recursive_goal_inferrer(
                state=state_latent.unsqueeze(0),
                goal=self.cur_goal_latent.unsqueeze(
                    0
                ),  # TODO: Have this change every episode? (add to buffer if so)
                num_recursions_past_stop=1,
            )
            goal_inference_stopped_immediately = len(stop_probs) == 2
            # Decode actions from lowest-three levels of goal hierarchy
            action_dist_outputs = self.action_decoder(torch.stack(goal_hierarchy, dim=0))
            actions, log_probs = self.action_decoder.sample_from_output(action_dist_outputs)
            # Environment step
            action_at_stop = actions[1]  # 2nd lowest-order goal (since we forced extra recursion)
            next_state, reward, terminated, truncated, _ = self.env.step(action_at_stop)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state_latent = self.env_encoder(next_state.unsqueeze(0)).squeeze(0)
            episode_is_done = terminated or truncated
            # Update experience buffer
            experience_buffer.add(
                state=state,
                state_latent=state_latent,
                next_state=next_state,
                next_state_latent=next_state_latent,
                reward=reward,
                done=episode_is_done,
                recursion_stop_level=len(
                    stop_probs
                ),  # Since stop level is 1-idxed, the stop is one before the final level (we forced an extra recursion)
                # Actions
                action_before_stop=None if goal_inference_stopped_immediately else actions[2],
                action=action_at_stop,
                action_after_stop=actions[0],
                # Log probabilities
                log_prob_before_stop=None if goal_inference_stopped_immediately else log_probs[2],
                log_prob=log_probs[1],
                log_prob_after_stop=log_probs[0],
            )
            return state, next_state_latent, episode_is_done

        # Method logic
        experience_buffer = RecursivePolicyExperienceBuffer()
        cur_state = torch.tensor(self.env.reset()[0], dtype=torch.float32)
        cur_state_latent = self.env_encoder(cur_state.unsqueeze(0)).squeeze(0)
        for _ in range(num_steps):
            cur_state, cur_state_latent, episode_is_done = process_step(cur_state, cur_state_latent)
            if episode_is_done:
                cur_state = torch.tensor(self.env.reset()[0], dtype=torch.float32)
                cur_state_latent = self.env_encoder(cur_state.unsqueeze(0)).squeeze(0)
        return experience_buffer

    def train_rl(
        self,
        num_update_batches: int = 10_000,
        num_steps_per_update: int = 100_000,
        clip_epsilon: float = 0.2,
        discount_gamma: float = 0.99,
        lr: float = 0.001,
    ):
        optimizer_all_params = torch.optim.Adam(
            list(
                self.recursive_goal_inferrer.parameters()
                + self.action_decoder.parameters()
                + self.value_function.parameters()
                + self.latent_dynamics_model.parameters()
            ),
            lr=lr,
        )
        optimizer_recursive_goal_inferrer_params_only = torch.optim.Adam(
            self.recursive_goal_inferrer.inner_inferrer.parameters(),
            lr=lr,
        )

        for update_batch_num in range(num_update_batches):
            optimizer_all_params.zero_grad()
            optimizer_recursive_goal_inferrer_params_only.zero_grad()

            experience = self.collect_rollouts_for_rl_training(num_steps=num_steps_per_update)

            ################################################################
            ## Compute GAE advantages for trajectories as they transpired ##
            ################################################################

            goal_batch = self.cur_goal_latent.unsqueeze(0).expand(len(experience.states), -1)

            values = self.value_function(
                state=torch.stack(experience.state_latents, dim=0),
                goal=goal_batch,
            )
            next_values = self.value_function(
                state=torch.stack(experience.next_state_latents, dim=0),
                goal=goal_batch,
            )
            advantages, returns = compute_advantage(
                method="gae",
                values=values,
                next_values=next_values,
                dones=torch.stack(experience.dones, dim=0),
                rewards=torch.tensor(experience.rewards, dtype=torch.float32),
                gamma=discount_gamma,
            )

            ######################################################################################
            ## Compute advs for would-have-been actions if goal inference stopped earlier/later ##
            ######################################################################################

            # Estimate would-have-been next states using action-conditioned dynamics model
            next_state_latents_early_stop = self.latent_dynamics_model(
                state=torch.stack(experience.states, dim=0),
                action=torch.stack(experience.actions_before_stop, dim=0),
            )
            next_state_latents_late_stop = self.latent_dynamics_model(
                state=torch.stack(experience.states, dim=0),
                action=torch.stack(experience.actions_after_stop, dim=0),
            )
            # Estimate would-have-been next values
            next_values_early_stop = self.value_function(
                state=next_state_latents_early_stop,
                goal=goal_batch,
            )
            next_values_late_stop = self.value_function(
                state=next_state_latents_late_stop,
                goal=goal_batch,
            )
            # Get bootstrapped would-have-been advantage estimates from values & next values
            advantages_early_stop, returns_early_stop = compute_advantage(
                method="bootstrap",
                values=values,
                next_values=next_values_early_stop,
                dones=torch.stack(experience.dones, dim=0),
                gamma=discount_gamma,
            )
            advantages_late_stop, returns_late_stop = compute_advantage(
                method="bootstrap",
                values=values,
                next_values=next_values_late_stop,
                dones=torch.stack(experience.dones, dim=0),
                gamma=discount_gamma,
            )

            #################################################################################
            ## Estimate would-have-been losses if goal inference had stopped earlier/later ##
            #################################################################################

            loss_early_stop_all = compute_clipped_surrogate_loss(
                # TODO: Old log probs?
                # E.g., from forward pass of prev update batch's policy?
                # ...or from multiple update iterations w/in update batch?
                old_log_probs=[p for p in experience.log_probs_before_stop if p is not None],
                log_probs=[p for p in experience.log_probs_before_stop if p is not None],
                advantages=[a for a in advantages_early_stop if a is not None],
                # TODO: Change design to remove need for these loops ^^
                clip_epsilon=clip_epsilon,
            )
            loss_late_stop_all = compute_clipped_surrogate_loss(
                # TODO: Old log probs?
                # E.g., from forward pass of prev update batch's policy?
                # ...or from multiple update iterations w/in update batch?
                old_log_probs=experience.log_probs_after_stop,
                log_probs=experience.log_probs_after_stop,
                advantages=advantages_late_stop,
                clip_epsilon=clip_epsilon,
            )

            #####################################################################################
            ## Speculatively update model to optimize for more & less goal-inference recursion ##
            #####################################################################################

            self.recursive_goal_inferrer.should_assign_credit_for_stop_probs = False

            recursive_goal_inferrer_earlier_stop_bias, _ = apply_speculative_update(
                original_model=self.recursive_goal_inferrer,
                original_optimizer=optimizer_recursive_goal_inferrer_params_only,
                from_loss=loss_early_stop_all,
            )
            recursive_goal_inferrer_later_stop_bias, _ = apply_speculative_update(
                original_model=self.recursive_goal_inferrer,
                original_optimizer=optimizer_recursive_goal_inferrer_params_only,
                from_loss=loss_late_stop_all,
            )

            # NOTE:
            #
            # To decide how to assign credit for when goal inference should stop, we sort
            # the batch into three buckets:
            #
            #  1. Steps where we estimate the advantage to have been higher if we had:
            #     (1) used the goal inferrer that is slightly more optimized for earlier
            #     stopping and (2) stopped one step earlier than we actually did.
            #
            #  2. Steps where the stopped-at recursion level + current goal inferrer
            #     produced the highest advantage.
            #
            #  3. Same as 1, but for later stopping.
            #
            # NOTE: The reason why we, before calculating these advantages, speculatively
            # update the goal inferrer params to get (marginally) better at goal
            # prediction at earlier and later stop-levels is because, presumably, the
            # stopped-at stop level is (likely) the optimal stop level for the CURRENT
            # goal inferrer. However, that does not mean that it is the the optimal stop
            # level for the OPTIMAL goal inferrer. Therefore, by performing speculative
            # updates that marginally unbias the goal inferrer away from its currently
            # optimized-for stopping levels, we are hopefully discovering when slight
            # changes in the stop-level biasing of the current goal-inferring circuits
            # could lead to performanace improvements.
            #
            # ...hopefully, allowing use to, for all situations, hill climb towards the
            # best recursion-stop level.

            #####################################################################################
            ## For all steps: (1) forward pass speculative models, forcing earlier/later stops ##
            ## & (2) compute/compare speculative advs to determine what to use for final loss  ##
            #####################################################################################

            def compute_speculative_advantage_for_step(
                i: int,
                recursive_goal_inferrer: RecursiveGoalInferrer,
                stop_at_recursion_level: int,
            ) -> float:
                goal_hierarchy, _ = recursive_goal_inferrer(
                    env_state=state_latent,
                    goal=self.cur_goal_latent,
                    fixed_num_recursions=stop_at_recursion_level,
                )
                lowest_order_goal = goal_hierarchy[0]
                action_dist_output = self.action_decoder(lowest_order_goal.unsqueeze(0))
                action = self.action_decoder.sample_from_output(action_dist_output)
                next_state_latent = self.latent_dynamics_model(
                    state=state.unsqueeze(0),
                    action=action,
                )
                value = values[i]
                next_value = self.value_function(
                    state=next_state_latent, goal=self.cur_goal_latent.unsqueeze(0)
                )
                return compute_advantage(
                    method="bootstrap",
                    values=[value],
                    next_values=[next_value],
                    dones=[experience.dones[i]],
                    gamma=discount_gamma,
                )[0][0]

            # TODO: Make this more efficient by batching (instead of looping through steps)
            target_recursion_stop_levels_of_batch = torch.zeros(
                size=num_steps_per_update, dtype=torch.int64
            )
            log_probs_for_final_loss = []
            advantages_for_final_loss = []
            for i in range(num_steps_per_update):
                state = experience.states[i]
                state_latent = experience.state_latents[i]
                stopped_at_recursion_level = experience.recursion_stop_levels[i]

                # Compute speculative advantages for comparison
                late_advantage = compute_speculative_advantage_for_step(
                    i,
                    recursive_goal_inferrer=recursive_goal_inferrer_later_stop_bias,
                    stop_at_recursion_level=stopped_at_recursion_level + 1,
                )
                if stopped_at_recursion_level == 1:  # Can't stop earlier than first level
                    early_advantage = float("-inf")
                else:
                    early_advantage = compute_speculative_advantage_for_step(
                        i,
                        recursive_goal_inferrer=recursive_goal_inferrer_earlier_stop_bias,
                        stop_at_recursion_level=stopped_at_recursion_level - 1,
                    )
                highest_advantage = max(early_advantage, advantages[i], late_advantage)
                # TODO: Pretty sure I don't want to aggregate advantages here...
                # Heuristic: I want the model to go towards a different stop level in these
                # situations, therefore, optimizing for that, should result in less bias for
                # current stop level.
                if highest_advantage == early_advantage:
                    log_probs_for_final_loss.append(experience.log_probs_before_stop[i])
                    advantages_for_final_loss.append(advantages_early_stop[i])
                    target_recursion_stop_levels_of_batch[i] = stopped_at_recursion_level - 1
                elif highest_advantage == late_advantage:
                    log_probs_for_final_loss.append(experience.log_probs_after_stop[i])
                    advantages_for_final_loss.append(advantages_late_stop[i])
                    target_recursion_stop_levels_of_batch[i] = stopped_at_recursion_level + 1
                else:  # highest_advantage == advantages[i]
                    log_probs_for_final_loss.append(experience.log_probs[i])
                    advantages_for_final_loss.append(advantages[i])
                    target_recursion_stop_levels_of_batch[i] = stopped_at_recursion_level

            ################################################################
            ## Finally, perform updates bucket-by-bucket, in random order ##
            ################################################################

            # Compute final loss
            loss = compute_clipped_surrogate_loss(
                # TODO: Old log probs?
                # E.g., from forward pass of prev update batch's policy?
                # ...or from multiple update iterations w/in update batch?
                old_log_probs=log_probs_for_final_loss,
                log_probs=log_probs_for_final_loss,
                advantages=advantages_for_final_loss,
                clip_epsilon=clip_epsilon,
            )

            # Set RecursiveGoalInferrer state to inject CE-loss gradient for stop-level
            self.recursive_goal_inferrer.should_assign_credit_for_stop_probs = True
            self.recursive_goal_inferrer.target_recursion_stop_levels_of_batch = (
                target_recursion_stop_levels_of_batch
            )
            # Do update
            loss.backward()
            optimizer_all_params.step()
