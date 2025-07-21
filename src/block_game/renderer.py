import pygame
from typing import Dict, List, Tuple, Union, Optional
from src.block_game.state import BlockGameState


class BlockGameRenderer:
    """
    Handles pygame rendering for the BlockGame environment.

    This renderer displays a visual representation of the block stacking game,
    showing both the current state and target state side by side. It includes
    game statistics and uses color-coded blocks for easy identification.

    The renderer creates a pygame window with two main sections:
    - Left side: Current state of the blocks
    - Right side: Target state to achieve
    - Top: Title and game statistics (steps, reward, moves to goal)

    Attributes:
        n_blocks (int): Number of blocks in the game
        n_stacks (int): Number of stacks available for placing blocks
        window_width (int): Width of the pygame window in pixels
        window_height (int): Height of the pygame window in pixels
        screen (pygame.Surface): Main pygame display surface
        clock (pygame.time.Clock): Pygame clock for controlling frame rate
        font (pygame.font.Font): Font for main text (36pt)
        small_font (pygame.font.Font): Font for smaller text (24pt)
        colors (Dict[str, Tuple[int, int, int]]): Color scheme for rendering
        block_width (int): Width of individual blocks in pixels
        block_height (int): Height of individual blocks in pixels
        stack_spacing (int): Horizontal spacing between stacks in pixels
        stack_base_height (int): Height of stack base indicators in pixels
        margin (int): Border margin around the display area in pixels
    """

    def __init__(
        self,
        n_blocks: int = 4,
        n_stacks: int = 4,
        window_width: int = 1000,
        window_height: int = 600,
    ) -> None:
        """
        Initialize the BlockGame renderer with pygame components.

        Args:
            n_blocks: Number of blocks in the game. Must be positive.
            n_stacks: Number of stacks for placing blocks. Must be positive.
            window_width: Width of the display window in pixels.
            window_height: Height of the display window in pixels.

        Raises:
            pygame.error: If pygame initialization fails

        Note:
            This constructor initializes pygame, creates the display window,
            and sets up fonts and color schemes. The color scheme supports
            up to 8 different colored blocks.
        """
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.window_width = window_width
        self.window_height = window_height

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Blocks World RL Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Color scheme for the renderer
        self.colors: Dict[str, Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {
            "background": (240, 240, 240),  # Light gray background
            "text": (0, 0, 0),  # Black text
            "stack_base": (100, 100, 100),  # Dark gray stack bases
            "current_bg": (220, 220, 255),  # Light blue background for current state
            "target_bg": (255, 220, 220),  # Light red background for target state
            "valid_move": (0, 150, 0),  # Green for valid moves
            "invalid_move": (200, 0, 0),  # Red for invalid moves
            "block_colors": [
                (255, 100, 100),  # Red
                (100, 255, 100),  # Green
                (100, 100, 255),  # Blue
                (255, 255, 100),  # Yellow
                (255, 100, 255),  # Magenta
                (100, 255, 255),  # Cyan
                (255, 150, 100),  # Orange
                (150, 255, 150),  # Light Green
            ],
        }

        # Layout parameters for positioning elements
        self.block_width = 60  # Width of each block in pixels
        self.block_height = 40  # Height of each block in pixels
        self.stack_spacing = 80  # Horizontal distance between stack centers
        self.stack_base_height = 10  # Height of the stack base indicator
        self.margin = 50  # Border margin around display area

    def handle_events(self) -> bool:
        """
        Handle pygame events for window interaction.

        Returns:
            bool: True if the window should close (QUIT event received), False otherwise

        Note:
            Currently only handles the QUIT event (window close button).
            This method should be called regularly to keep the window responsive.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def render(
        self,
        current_state: BlockGameState,
        target_state: BlockGameState,
        episode_step_count: int = 0,
        episode_total_reward: float = 0,
        min_moves_to_solution: int = 0,
        last_move_was_valid: Optional[bool] = None,
    ) -> None:
        """
        Render the complete game state display.

        This method draws the entire game interface including:
        - Title and statistics at the top
        - Current state on the left side
        - Target state on the right side
        - Color-coded blocks with numbers for identification
        - Move validity indicator (green checkmark or red X)

        Args:
            current_state: The current configuration of blocks in stacks
            target_state: The goal configuration to achieve
            episode_step_count: Number of steps taken in current episode
            episode_total_reward: Cumulative reward for current episode
            min_moves_to_solution: Minimum number of moves needed to reach target
            last_move_was_valid: Whether the last move was valid (None for initial state)

        Note:
            This method handles pygame events internally to keep the window
            responsive but does not return event status. Call at 60 FPS for
            smooth display updates.
        """
        # Consume pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            pass  # Just consume events to prevent window from becoming unresponsive

        # Clear screen with background color
        self.screen.fill(self.colors["background"])

        # Draw title and statistics at the top
        self._draw_title_and_stats(
            episode_step_count,
            episode_total_reward,
            min_moves_to_solution,
            last_move_was_valid,
        )

        # Calculate layout positions for current and target state sections
        actual_stacks_width = (self.n_stacks - 1) * self.stack_spacing
        current_x = self.window_width // 4 - actual_stacks_width // 2
        target_x = 3 * self.window_width // 4 - actual_stacks_width // 2
        base_y = self.window_height - self.margin - self.stack_base_height

        # Draw current state section (left side)
        self._draw_state_section(
            "Current State",
            current_state,
            current_x,
            base_y,
            self.colors["current_bg"],
            actual_stacks_width,
        )

        # Draw target state section (right side)
        self._draw_state_section(
            "Target State",
            target_state,
            target_x,
            base_y,
            self.colors["target_bg"],
            actual_stacks_width,
        )

        # Update display and maintain frame rate
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    def _draw_title_and_stats(
        self,
        episode_step_count: int,
        total_reward: float,
        min_moves_to_solution: int,
        was_valid_move: Optional[bool] = None,
    ) -> None:
        """
        Draw the title and game statistics at the top of the window.

        Args:
            episode_step_count: Current step count in the episode
            total_reward: Cumulative reward earned in current episode
            min_moves_to_solution: Minimum moves required to reach target state
            was_valid_move: Whether the last move was valid (None for initial state)

        Note:
            The title is centered at the top, with statistics displayed below it.
            Statistics show step count, total reward, and optimal moves remaining.
            Move validity is shown with a green checkmark (✓) or red X (✗).
        """
        # Draw main title
        title_text = self.font.render(
            "Blocks World RL Environment", True, self.colors["text"]
        )
        title_x = self.window_width // 2 - title_text.get_width() // 2
        self.screen.blit(title_text, (title_x, 20))

        # Draw statistics line
        stats_text = f"Step: {episode_step_count} | Reward: {total_reward} | Moves to Goal: {min_moves_to_solution}"
        stats_surface = self.small_font.render(stats_text, True, self.colors["text"])
        stats_x = self.window_width // 2 - stats_surface.get_width() // 2
        self.screen.blit(stats_surface, (stats_x, 60))

        # Draw move validity indicator if applicable
        if was_valid_move is not None:
            self._draw_move_validity_indicator(was_valid_move)

    def _draw_move_validity_indicator(self, was_valid_move: bool) -> None:
        """
        Draw a visual indicator showing whether the last move was valid.

        Displays "Valid Move:" followed by either a green checkmark for valid moves
        or a red X for invalid moves. Uses drawn shapes instead of Unicode characters
        for better compatibility. Positioned in the upper right corner.

        Args:
            was_valid_move: True if the last move was valid, False if invalid

        Note:
            Uses drawn shapes for checkmark and X. Colors are green for valid,
            red for invalid to provide clear visual feedback.
        """
        import pygame

        # Choose color based on move validity
        if was_valid_move:
            color = self.colors["valid_move"]
        else:
            color = self.colors["invalid_move"]

        # Create the label text surface
        label_text = "Valid Move: "
        label_surface = self.small_font.render(label_text, True, self.colors["text"])

        # Define the symbol area dimensions
        symbol_size = 16  # Size of the symbol area
        symbol_margin = 5  # Space between label and symbol

        # Position in upper right corner with some margin
        label_x = (
            self.window_width
            - self.margin
            - label_surface.get_width()
            - symbol_size
            - symbol_margin
        )
        label_y = 20
        symbol_x = label_x + label_surface.get_width() + symbol_margin
        symbol_y = (
            label_y + (label_surface.get_height() - symbol_size) // 2
        )  # Center vertically with text

        # Draw the label
        self.screen.blit(label_surface, (label_x, label_y))

        # Draw the appropriate symbol
        if was_valid_move:
            self._draw_checkmark(symbol_x, symbol_y, symbol_size, color)
        else:
            self._draw_x_mark(symbol_x, symbol_y, symbol_size, color)

    def _draw_checkmark(self, x: int, y: int, size: int, color: tuple) -> None:
        """
        Draw a checkmark using lines.

        Args:
            x: Left position of the checkmark area
            y: Top position of the checkmark area
            size: Size of the checkmark area
            color: RGB color tuple for the checkmark
        """
        import pygame

        # Calculate checkmark points
        # The checkmark is drawn as two connected lines forming a "✓" shape
        line_width = max(2, size // 8)  # Scale line width with size

        # First line: bottom-left to middle-bottom (short line)
        start1 = (x + size // 4, y + size // 2)
        end1 = (x + size // 2, y + size * 3 // 4)

        # Second line: middle-bottom to top-right (long line)
        start2 = (x + size // 2, y + size * 3 // 4)
        end2 = (x + size * 3 // 4, y + size // 4)

        # Draw the checkmark lines
        pygame.draw.line(self.screen, color, start1, end1, line_width)
        pygame.draw.line(self.screen, color, start2, end2, line_width)

    def _draw_x_mark(self, x: int, y: int, size: int, color: tuple) -> None:
        """
        Draw an X mark using two crossing lines.

        Args:
            x: Left position of the X area
            y: Top position of the X area
            size: Size of the X area
            color: RGB color tuple for the X mark
        """
        import pygame

        # Calculate X mark points with some padding
        line_width = max(2, size // 8)  # Scale line width with size
        padding = size // 6  # Small padding from edges

        # First diagonal line: top-left to bottom-right
        start1 = (x + padding, y + padding)
        end1 = (x + size - padding, y + size - padding)

        # Second diagonal line: top-right to bottom-left
        start2 = (x + size - padding, y + padding)
        end2 = (x + padding, y + size - padding)

        # Draw the X lines
        pygame.draw.line(self.screen, color, start1, end1, line_width)
        pygame.draw.line(self.screen, color, start2, end2, line_width)

    def _draw_state_section(
        self,
        label: str,
        state: BlockGameState,
        offset_x: int,
        base_y: int,
        bg_color: Tuple[int, int, int],
        section_width: int,
    ) -> None:
        """
        Draw a labeled section containing a game state with background.

        This method draws a complete state section including:
        - Section label (e.g., "Current State", "Target State")
        - Colored background panel
        - The actual block configuration

        Args:
            label: Text label for this state section
            state: BlockGameState object containing the block configuration
            offset_x: X coordinate for the left edge of this section
            base_y: Y coordinate for the bottom of the stacks
            bg_color: RGB color tuple for the background panel
            section_width: Width of the section for centering elements

        Note:
            The background panel uses rounded corners for a polished appearance.
        """
        # Draw section label centered above the state
        label_surface = self.font.render(label, True, self.colors["text"])
        label_x = offset_x + section_width // 2 - label_surface.get_width() // 2
        self.screen.blit(label_surface, (label_x, 100))

        # Draw rounded background panel
        bg_rect = pygame.Rect(offset_x - 20, 140, section_width + 40, base_y - 120)
        pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=10)

        # Draw the actual game state
        self._draw_state(state, offset_x, base_y)

    def _draw_state(self, state: BlockGameState, offset_x: int, base_y: int) -> None:
        """
        Draw a complete blocks world state at the specified position.

        This method renders all stacks in the given state, including:
        - Stack base indicators
        - Stack numbers (0, 1, 2, ...)
        - All blocks in each stack with proper vertical positioning

        Args:
            state: BlockGameState object containing stack configurations
            offset_x: X coordinate offset for positioning this state
            base_y: Y coordinate for the bottom of all stacks

        Note:
            Stacks are drawn from left to right with consistent spacing.
            Each stack shows its index number below the base.
        """
        for stack_idx, stack in enumerate(state.stacks):
            stack_x = offset_x + stack_idx * self.stack_spacing

            # Draw stack base indicator
            base_rect = pygame.Rect(
                stack_x - self.block_width // 2,
                base_y,
                self.block_width,
                self.stack_base_height,
            )
            pygame.draw.rect(self.screen, self.colors["stack_base"], base_rect)

            # Draw stack index number below the base
            stack_num_text = self.small_font.render(
                str(stack_idx), True, self.colors["text"]
            )
            text_x = stack_x - stack_num_text.get_width() // 2
            text_y = base_y + self.stack_base_height + 5
            self.screen.blit(stack_num_text, (text_x, text_y))

            # Draw all blocks in this stack from bottom to top
            for height, block_id in enumerate(stack):
                self._draw_block(stack_x, base_y, height, block_id)

    def _draw_block(self, stack_x: int, base_y: int, height: int, block_id: int) -> None:
        """
        Draw a single block with color coding and ID number.

        Each block is rendered as a colored rectangle with:
        - Color determined by block ID (cycling through available colors)
        - Black border for definition
        - Block ID number centered inside
        - Rounded corners for visual appeal

        Args:
            stack_x: X coordinate of the center of the stack
            base_y: Y coordinate of the stack base
            height: Vertical position in stack (0=bottom, 1=second from bottom, etc.)
            block_id: Numeric identifier of the block (determines color)

        Note:
            Blocks are positioned vertically based on their height in the stack.
            Color cycling allows for up to 8 distinct block colors before repeating.
        """
        # Calculate block position
        block_y = base_y - (height + 1) * self.block_height
        block_rect = pygame.Rect(
            stack_x - self.block_width // 2,
            block_y,
            self.block_width,
            self.block_height,
        )

        # Select color based on block ID (with cycling for >8 blocks)
        color_idx = (block_id - 1) % len(self.colors["block_colors"])
        block_color = self.colors["block_colors"][color_idx]

        # Draw filled block with rounded corners
        pygame.draw.rect(self.screen, block_color, block_rect, border_radius=5)
        # Draw black border
        pygame.draw.rect(self.screen, (0, 0, 0), block_rect, width=2, border_radius=5)

        # Draw block ID number centered in the block
        block_text = self.small_font.render(str(block_id), True, (0, 0, 0))
        text_x = stack_x - block_text.get_width() // 2
        text_y = block_y + self.block_height // 2 - block_text.get_height() // 2
        self.screen.blit(block_text, (text_x, text_y))

    def close(self) -> None:
        """
        Clean up pygame resources and quit pygame.

        This method should be called when the renderer is no longer needed
        to properly release pygame resources and close the display window.

        Note:
            After calling this method, the renderer cannot be used again
            without recreating the object.
        """
        pygame.quit()
