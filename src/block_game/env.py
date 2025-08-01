from typing import Generator, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces

from src.block_game.state import BlockGameState
from src.block_game.renderer import BlockGameRenderer


class BlockGameEnv(gym.Env):
    """
    Gymnasium environment for the block stacking game with a 2D action space.
    Modified to include both current state and goal state in a single concatenated observation.

    Environment Description:
    - Goal: Rearrange numbered blocks between stacks to match a target configuration
    - Only the top block of each stack can be moved
    - Game ends when current state matches the target state

    Unconventional Design Choice - 2D Integer Action Space:
    - Action spaces are 2D arrays of shape (n_blocks, n_stacks)
    - Each element is an integer in {0, 1, ..., n_blocks}, where:
        * 0 represents an empty slot
        * 1 to n_blocks represent block IDs
    - Actions specify the target state as a grid of block IDs, reachable in one valid move
    - Invalid actions (unreachable states or malformed configurations) receive -1 reward

    Observation Space (2D Array):
    - Shape: (2 * n_blocks, n_stacks) - concatenated current and goal states
    - First n_blocks rows: current stack configuration
    - Next n_blocks rows: goal stack configuration
    - Each element is the block ID (1 to n_blocks) or 0 (empty) at that position

    Action Space (2D Array):
    - Shape: (n_blocks, n_stacks)
    - Specifies the target state to achieve in one move
    - Must be reachable from the current state in exactly one valid move

    Rewards:
    - +1: Valid move that reduces minimum moves to solution
    - 0.25: Valid move that doesn't improve distance to solution
    - -1: Invalid action (unreachable state or malformed configuration)
    """

    def __init__(
        self,
        n_blocks: int = 4,
        n_stacks: int = 4,
        max_episode_length: int = 20,
        render_mode: Literal["human"] | None = None,
    ):
        """
        Initialize the block stacking environment.

        Args:
            n_blocks: Number of blocks in the game (positive integer)
            n_stacks: Number of stacks (must be >= 3 for full connectivity)
            max_episode_length: Maximum number of steps per episode (None for unlimited)
            render_mode: Optional rendering mode
                ("human" for visual rendering, None for no rendering)

        Raises:
            ValueError: If n_stacks < 3 (insufficient for guaranteed state connectivity)
        """
        if n_stacks < 3:
            msg = "n_stacks must be >= 3 to ensure any state can become any state."
            raise ValueError(msg)

        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode

        self.game_state: None | BlockGameState = None  # Current game state
        self.solution_state: None | BlockGameState = None  # Target state to reach
        self.last_returned_obs: None | NDArray[np.int32] = None  # Last returned obs
        self.min_moves_to_solution: None | int = None  # Min moves to solution
        self.total_step_count = 0  # Total steps across all episodes
        self.episode_step_count = 0  # Steps in the current episode
        self.episode_total_reward = 0  # Total reward in the current episode
        self.last_move_was_valid: None | bool = None  # Whether the last move was valid

        # Observation space: concatenated current and goal states
        # Shape: (2 * n_blocks, n_stacks) where first n_blocks rows are current, next n_blocks are goal
        self.observation_space = spaces.MultiDiscrete(
            [[n_blocks + 1] * n_stacks] * (2 * n_blocks), dtype=np.int32
        )

        # Action space: target state configuration
        self.action_space = spaces.MultiDiscrete(
            [[n_blocks + 1] * n_stacks] * n_blocks, dtype=np.int32
        )

        # Initialize game renderer if needed
        if render_mode == "human":
            self.renderer = BlockGameRenderer(
                n_blocks=self.n_blocks,
                n_stacks=self.n_stacks,
                window_width=1000,
                window_height=600,
            )
        else:
            self.renderer = None

    ###############
    ### Helpers ###
    ###############

    def _state_to_obs(self, game_state: BlockGameState) -> NDArray[np.int32]:
        """
        Convert a BlockGameState to the environment's observation format.

        Args:
            game_state: The game state to convert

        Returns:
            ObservationArray: 2D array of shape (n_blocks, n_stacks) with block IDs
                             (0 for empty, 1 to n_blocks), dtype=int32
        """
        obs = np.zeros((self.n_blocks, self.n_stacks), dtype=np.int32)
        for stack_idx, stack in enumerate(game_state.stacks):
            for height in range(self.n_blocks):
                if height < len(stack):
                    obs[height, stack_idx] = stack[height]
                else:
                    obs[height, stack_idx] = 0  # Empty slot
        return obs

    def _get_full_observation(self) -> NDArray[np.int32]:
        """
        Get the full observation including both current and goal states concatenated.

        Returns:
            NDArray: Observation of shape (2 * n_blocks, n_stacks) where:
                    - First n_blocks rows: current state
                    - Next n_blocks rows: goal state
        """
        current_state_obs = self._state_to_obs(self.game_state)
        goal_state_obs = self._state_to_obs(self.solution_state)

        # Concatenate along the first dimension
        full_obs = np.concatenate([current_state_obs, goal_state_obs], axis=0)
        return full_obs

    def _obs_to_state(self, obs: NDArray[np.int32]) -> BlockGameState | None:
        """
        Convert an observation/action array to a BlockGameState.

        Args:
            obs: 2D array of shape (n_blocks, n_stacks), dtype=int32
                 representing block IDs (0 for empty, 1 to n_blocks)

        Returns:
            BlockGameState: Converted game state, or None if invalid configuration

        Validation Checks:
            - Blocks cannot float (no empty slots below blocks in a stack)
            - Each block ID appears exactly once
            - All expected blocks are present
            - Values are in valid range (0 to n_blocks)
        """
        if not np.all((0 <= obs) & (obs <= self.n_blocks)):
            return None  # Invalid: out-of-range values

        stacks = [[] for _ in range(self.n_stacks)]
        already_seen_block_ids = set()

        for stack_idx in range(self.n_stacks):
            already_saw_empty_slot = False
            for height in range(self.n_blocks):
                block_id = obs[height, stack_idx]
                if block_id == 0:
                    already_saw_empty_slot = True
                else:
                    if already_saw_empty_slot:
                        return None  # Invalid: blocks can't float
                    if block_id in already_seen_block_ids:
                        return None  # Invalid: duplicate block
                    already_seen_block_ids.add(block_id)
                    stacks[stack_idx].append(block_id)

        if len(already_seen_block_ids) != self.n_blocks:
            return None  # Invalid: not all blocks present

        return BlockGameState(self.n_blocks, self.n_stacks, stacks)

    def get_valid_actions(self) -> Generator[NDArray[np.int32], None, None]:
        """
        Generate all valid actions from the current state.

        Yields:
            ActionArray: 2D arrays of shape (n_blocks, n_stacks), dtype=int32
                        representing states reachable in one move
        """
        for valid_move in self.game_state.get_valid_moves():
            next_state = self.game_state.copy()
            next_state.apply_action(*valid_move)
            yield self._state_to_obs(next_state)

    #######################
    ### Environment API ###
    #######################

    def step(
        self, action: NDArray[np.int32]
    ) -> Tuple[NDArray[np.int32], float, bool, bool, dict]:
        """
        Execute one environment step with the given action.

        Args:
            action: 2D array of shape (n_blocks, n_stacks), dtype=int32
                specifying target state to achieve in one move

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
            - observation: Concatenated current and goal states of shape (2*n_blocks, n_stacks)
            - reward: 1 (improvement), 0.25 (valid move, no improvement), or -1 (invalid)
            - terminated: True if target state reached
            - truncated: True if max episode length reached
            - info: Empty dict
        """
        self.episode_step_count += 1
        self.total_step_count += 1

        target_state = self._obs_to_state(action)
        # Invalid action: malformed state or not reachable in one move
        action_was_valid = target_state is not None and BlockGameState.are_one_move_away(
            self.game_state, target_state
        )

        if action_was_valid:
            # Calculate reward based on progress toward solution
            min_moves_before_action = self.min_moves_to_solution
            min_moves_after_action = BlockGameState.get_min_moves_between(
                target_state, self.solution_state
            )

            if min_moves_after_action < min_moves_before_action:
                reward = 1  # Progress toward solution
            else:
                reward = 0.25  # No progress, but still some reward for valid action

            # Update environment state
            self.last_move_was_valid = True
            self.game_state = target_state
            self.min_moves_to_solution = min_moves_after_action

        else:
            reward = -1  # Penalize invalid action
            self.last_move_was_valid = False

        # Always return current observation (whether state changed or not)
        obs = self._get_full_observation()
        self.episode_total_reward += reward
        terminated = self.min_moves_to_solution == 0

        truncated = (  # Check if max episode length reached
            self.max_episode_length is not None
            and self.episode_step_count >= self.max_episode_length
        )

        self.last_returned_obs = obs

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None) -> Tuple[NDArray[np.int32], dict]:
        """
        Reset the environment to a new random episode.

        Args:
            seed: Random seed for reproducible episodes

        Returns:
            tuple: (initial_observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        self.episode_step_count = 0
        self.episode_total_reward = 0
        self.last_move_was_valid = None

        # Generate random start state
        self.game_state = BlockGameState.get_random(
            n_blocks=self.n_blocks, n_stacks=self.n_stacks
        )

        # Generate different random target state (try 30 times)
        for _ in range(30):
            self.solution_state = BlockGameState.get_random(
                n_blocks=self.n_blocks, n_stacks=self.n_stacks
            )
            if self.solution_state != self.game_state:
                self.min_moves_to_solution = BlockGameState.get_min_moves_between(
                    self.game_state, self.solution_state
                )
                break
        else:
            # Fallback: create target by making one move
            self.solution_state = self.game_state.copy()
            first_valid_move = next(self.solution_state.get_valid_moves())
            self.solution_state.apply_action(*first_valid_move)
            self.min_moves_to_solution = 1

        # Return initial observation with both current and goal states
        obs = self._get_full_observation()
        self.last_returned_obs = obs
        return obs, {}

    def render(self) -> None:
        """Render the current game state according to the render mode."""
        if self.render_mode == "human":
            self.renderer.render(
                current_state=self.game_state,
                target_state=self.solution_state,
                episode_step_count=self.episode_step_count,
                episode_total_reward=self.episode_total_reward,
                min_moves_to_solution=self.min_moves_to_solution,
                last_move_was_valid=self.last_move_was_valid,
            )
        elif self.render_mode is not None:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self):
        """Clean up environment resources."""
        if self.render_mode == "human":
            self.renderer.close()
