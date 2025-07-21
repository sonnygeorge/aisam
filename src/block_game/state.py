import random
from collections import deque
from typing import Generator

from joblib import Memory

memory = Memory("./cache", verbose=0)


class BlockGameState:
    """
    Represents the state of a blocks stacking game where numbered blocks can be moved between stacks.

    Game Rules:
    - Blocks are numbered from 1 to n_blocks
    - Blocks are arranged in n_stacks vertical stacks (minimum 3 stacks required)
    - Only the top block of each stack can be moved
    - A block can be moved from any non-empty stack to any other stack
    - All blocks must be present exactly once in the game state

    Stack Representation:
    - Each stack is represented as a list where index 0 is the bottom and index -1 is the top
    - Empty stacks are represented as empty lists
    - Moving a block means popping from source stack and appending to target stack

    Coordinate System:
    - Stacks are indexed from 0 to n_stacks-1
    - Block IDs are integers from 1 to n_blocks

    Thread Safety:
    - The class uses joblib.Memory for caching get_min_moves_between results
    - Cache is shared across all instances, so concurrent access may have race conditions

    Examples:
        >>> # Create a simple 3-block, 3-stack game
        >>> state = BlockGameState(3, 3, [[1, 2], [3], []])
        >>> print(state.stacks)  # [[1, 2], [3], []]
        >>>
        >>> # Move top block from stack 0 to stack 2
        >>> print(state.stacks)  # [[1, 2], [3], []]
        >>> state.apply_action(0, 2)  # Returns True
        >>> print(state.stacks)  # [[1], [3], [2]]
        >>>
        >>> # Generate random initial state
        >>> random_state = BlockGameState.get_random(5, 4)
    """

    def __init__(self, n_blocks: int, n_stacks: int, stacks: list[list[int]]):
        """
        Initialize a new game state.

        Args:
            n_blocks: Total number of blocks in the game (positive integer)
            n_stacks: Number of stacks (must be >= 3 for solvability guarantees)
            stacks: List of stacks, where each stack is a list of block IDs.
                   Length must equal n_stacks. Each block ID should appear exactly once
                   across all stacks.

        Raises:
            AssertionError: If n_stacks < 3

        Note:
            The constructor does not validate that blocks are correctly distributed.
            Use is_valid() to check state validity after construction.
        """
        assert n_stacks >= 3
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.stacks = stacks

    @classmethod
    def get_block_ids(cls, n_blocks: int) -> Generator[int, None, None]:
        """
        Generate the valid block IDs for a game with n_blocks.

        Args:
            n_blocks: Number of blocks in the game

        Yields:
            int: Block IDs from 1 to n_blocks (inclusive)

        Example:
            >>> list(BlockGameState.get_block_ids(3))
            [1, 2, 3]
        """
        # Block ids = ints from 1 to n_blocks
        for block_id in range(1, n_blocks + 1):
            yield block_id

    @classmethod
    def get_random(cls, n_blocks: int, n_stacks: int) -> "BlockGameState":
        """
        Generate a random initial game state with blocks distributed across stacks.

        Args:
            n_blocks: Number of blocks to distribute
            n_stacks: Number of stacks to distribute blocks across

        Returns:
            BlockGameState: A new random game state with all blocks distributed

        Note:
            Uses random module, so set random.seed() for reproducible results.
            Distribution is uniform random - each block independently chooses a random stack.
        """
        stacks = [[] for _ in range(n_stacks)]
        block_ids = list(cls.get_block_ids(n_blocks))
        random.shuffle(block_ids)  # Fixed: shuffle modifies in-place, returns None
        for block in block_ids:
            stack_index = random.randint(0, n_stacks - 1)
            stacks[stack_index].append(block)
        return cls(n_blocks, n_stacks, stacks)

    def __hash__(self) -> int:
        """
        Compute hash of the game state based on stack configuration.

        Returns:
            int: Hash value for use in sets/dictionaries

        Note:
            Two states with identical stack configurations will have the same hash.
            Uses tuple-based hashing for better performance than string conversion.
        """
        return hash(tuple(tuple(stack) for stack in self.stacks))

    def __eq__(self, other: "BlockGameState") -> bool:
        """
        Check if two game states are identical.

        Args:
            other: Another BlockGameState to compare with

        Returns:
            bool: True if both states have identical stack configurations
        """
        return self.stacks == other.stacks

    def _assert_same_dimension_params(self, other: "BlockGameState") -> None:
        """
        Assert that two states have compatible dimensions for comparison/operations.

        Args:
            other: Another BlockGameState to check compatibility with

        Raises:
            AssertionError: If n_blocks or n_stacks differ between states
        """
        assert (
            self.n_blocks == other.n_blocks and self.n_stacks == other.n_stacks
        ), "Compared states must have the same dimension parameters"

    def copy(self) -> "BlockGameState":
        """
        Create a deep copy of the current game state.

        Returns:
            BlockGameState: A new instance with identical but independent stack configuration

        Note:
            Modifications to the copy will not affect the original state.
        """
        return BlockGameState(self.n_blocks, self.n_stacks, [s.copy() for s in self.stacks])

    def apply_action(self, source_stack: int, target_stack: int) -> bool:
        """
        Apply a move action by moving the top block from source to target stack.

        Args:
            source_stack: Index of stack to move block from (0 to n_stacks-1)
            target_stack: Index of stack to move block to (0 to n_stacks-1)

        Returns:
            bool: True if the move was valid and applied, False if invalid

        Side Effects:
            Modifies self.stacks in-place if the move is valid

        Invalid Move Conditions:
            - source_stack == target_stack (no-op move)
            - source_stack is empty (nothing to move)
            - Invalid stack indices (will raise IndexError)

        Example:
            >>> state = BlockGameState(2, 3, [[1], [2], []])
            >>> state.apply_action(0, 2)  # Returns True
            >>> print(state.stacks)  # [[], [2], [1]]
        """
        if source_stack == target_stack:
            return False
        try:
            block = self.stacks[source_stack].pop()
        except IndexError:
            return False  # Cannot pop from an empty stack
        self.stacks[target_stack].append(block)
        return True

    def get_valid_moves(self) -> Generator[tuple[int, int], None, None]:
        """
        Generate all valid moves from the current state.

        Yields:
            tuple[int, int]: (source_stack, target_stack) pairs representing valid moves

        Note:
            A move is valid if the source stack is non-empty and source != target.
            Does not check if the move leads to a useful or optimal state.

        Example:
            >>> state = BlockGameState(2, 3, [[1], [2], []])
            >>> list(state.get_valid_moves())
            [(0, 1), (0, 2), (1, 0), (1, 2)]
        """
        for source_stack in range(self.n_stacks):
            if len(self.stacks[source_stack]) == 0:
                continue  # Cannot move a block from empty stack
            for target_stack in range(self.n_stacks):
                # Can move source stack's top block to any target stack that != source stack
                if source_stack == target_stack:
                    continue
                yield (source_stack, target_stack)

    def is_valid(self) -> bool:
        """
        Check if the current state is valid according to game rules.

        Returns:
            bool: True if state is valid, False otherwise

        Validation Checks:
            - No block ID appears more than once across all stacks
            - All expected block IDs (1 to n_blocks) are present exactly once
            - No unexpected block IDs are present

        Example:
            >>> state = BlockGameState(2, 3, [[1], [2], []])
            >>> state.is_valid()  # True
            >>>
            >>> invalid_state = BlockGameState(2, 3, [[1], [1], []])  # Duplicate block
            >>> invalid_state.is_valid()  # False
        """
        already_seen_block_ids = set()
        for stack in self.stacks:
            for block_id in stack:
                if block_id in already_seen_block_ids:
                    return False  # Duplicate block found
                already_seen_block_ids.add(block_id)
        if already_seen_block_ids != set(self.get_block_ids(self.n_blocks)):
            return False  # Not all blocks are present or bad block IDs are present
        return True

    @staticmethod
    @memory.cache
    def get_min_moves_between(
        start_state: "BlockGameState",
        end_state: "BlockGameState",
        max_search_depth: int = None,
    ) -> int | None:
        """
        Find the minimum number of moves required to transform start_state into end_state.

        Uses breadth-first search to guarantee the minimum number of moves is found.
        Results are cached using joblib.Memory for performance.

        Args:
            start_state: Initial game state
            end_state: Target game state
            max_search_depth: Optional limit on search depth. If provided and no path
                            is found within this depth, returns None instead of continuing.

        Returns:
            int: Minimum number of moves required (0 if states are identical)
            None: If max_search_depth is provided and no path found within limit

        Raises:
            ValueError: If no path exists between states (shouldn't happen with n_stacks >= 3)
            AssertionError: If states have different dimensions (n_blocks, n_stacks)

        Performance Notes:
            - Time complexity: O(b^d) where b is branching factor, d is solution depth
            - Space complexity: O(b^d) for visited states storage
            - Results are cached based on state representations

        Example:
            >>> start = BlockGameState(2, 3, [[1, 2], [], []])
            >>> end = BlockGameState(2, 3, [[], [], [1, 2]])
            >>> BlockGameState.get_min_moves_between(start, end)
            2
        """
        start_state._assert_same_dimension_params(end_state)
        if start_state == end_state:
            return 0

        queue = deque([(start_state.copy(), 0)])
        visited = {start_state}
        while queue:
            current_state, moves = queue.popleft()
            if max_search_depth is not None and moves >= max_search_depth:
                continue
            for source_stack, target_stack in current_state.get_valid_moves():
                next_state = current_state.copy()
                next_state.apply_action(source_stack, target_stack)
                if next_state == end_state:
                    return moves + 1
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, moves + 1))

        if max_search_depth is not None:
            return None

        raise ValueError("No path b/w states, which should not happen if n_stacks >= 3.")

    @staticmethod
    def are_one_move_away(state1: "BlockGameState", state2: "BlockGameState") -> bool:
        """
        Check if two states are exactly one valid move away from each other.

        Args:
            state1: First game state
            state2: Second game state

        Returns:
            bool: True if states are one move apart, False otherwise

        Note:
            This is more efficient than checking if min_moves == 1 because it stops
            searching after depth 1.

        Example:
            >>> state1 = BlockGameState(2, 3, [[1], [2], []])
            >>> state2 = BlockGameState(2, 3, [[], [2], [1]])
            >>> BlockGameState.are_one_move_away(state1, state2)
            True
        """
        moves_away = BlockGameState.get_min_moves_between(state1, state2, max_search_depth=1)
        return moves_away == 1
