from collections import deque
import random
from typing import Dict, List, Tuple, Optional, TypeVar
import networkx as nx
import plotly.graph_objects as go
import plotly.offline as pyo

# Generic type for state representation
StateType = TypeVar("StateType")


def compute_block_game_state_graph(
    n_blocks: int, n_stacks: int
) -> Dict[Tuple[Tuple[int, ...], ...], List[Tuple[Tuple[int, ...], ...]]]:
    """
    Compute the graph of all possible states for a block stacking game.

    Game Rules:
    - Blocks are numbered 1 to n_blocks
    - There are n_stacks vertical stacks
    - Only the top block of each stack can be moved
    - A block can be moved from any non-empty stack to any other stack
    - All blocks must be present exactly once

    Args:
        n_blocks: Number of blocks (positive integer)
        n_stacks: Number of stacks (must be >= 3 for guaranteed connectivity)

    Returns:
        Dictionary mapping each state to list of directly reachable states.
        Each state is represented as a tuple of tuples, where each inner tuple
        represents a stack from bottom to top.

    Example:
        For n_blocks=2, n_stacks=3:
        State ((1, 2), (), ()) means stack 0 has blocks [1,2] (1 at bottom),
        stacks 1 and 2 are empty.

    Raises:
        AssertionError: If n_stacks < 3 or n_blocks <= 0
    """
    assert n_stacks >= 3, "Need at least 3 stacks for guaranteed connectivity"
    assert n_blocks > 0, "Need at least 1 block"

    def stacks_to_tuple(stacks: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Convert list of stacks to immutable tuple representation."""
        return tuple(tuple(stack) for stack in stacks)

    def tuple_to_stacks(state_tuple: Tuple[Tuple[int, ...], ...]) -> List[List[int]]:
        """Convert tuple representation back to list of stacks."""
        return [list(stack) for stack in state_tuple]

    # Start with initial state: all blocks stacked in first stack (1 at bottom, n_blocks at top)
    initial_stacks = [list(range(1, n_blocks + 1))] + [[] for _ in range(n_stacks - 1)]
    initial_state = stacks_to_tuple(initial_stacks)

    # Use BFS to explore all reachable states
    # Since n_stacks >= 3, the graph is guaranteed to be connected
    graph: Dict[Tuple[Tuple[int, ...], ...], List[Tuple[Tuple[int, ...], ...]]] = {}
    queue = deque([initial_state])
    visited = {initial_state}

    while queue:
        current_state_tuple = queue.popleft()
        current_stacks = tuple_to_stacks(current_state_tuple)

        # Find all states reachable in exactly one move
        reachable_states = []

        for source_stack in range(n_stacks):
            if len(current_stacks[source_stack]) == 0:
                continue  # Cannot move from empty stack

            for target_stack in range(n_stacks):
                if source_stack == target_stack:
                    continue  # Cannot move to same stack (no-op)

                # Apply the move: take top block from source, put on target
                next_stacks = [stack.copy() for stack in current_stacks]
                block = next_stacks[source_stack].pop()  # Remove from top (end of list)
                next_stacks[target_stack].append(block)  # Add to top (end of list)
                next_state_tuple = stacks_to_tuple(next_stacks)

                reachable_states.append(next_state_tuple)

                # Add to exploration queue if not seen before
                if next_state_tuple not in visited:
                    visited.add(next_state_tuple)
                    queue.append(next_state_tuple)

        graph[current_state_tuple] = reachable_states

    return graph


def extract_optimal_path_subgraph(
    full_graph: Dict[StateType, List[StateType]],
    start_state: Optional[StateType] = None,
    end_state: Optional[StateType] = None,
) -> Dict[StateType, List[StateType]]:
    """
    Extract the subgraph containing only nodes and edges that participate in
    shortest paths between two given states.

    Args:
        full_graph: Complete state graph as adjacency list (state -> list of neighbors)
        start_state: Starting state (if None, randomly selected from graph)
        end_state: Target state (if None, randomly selected from graph, different from start)

    Returns:
        Subgraph containing only nodes/edges on optimal paths between start and end states.
        Returns empty dict if no path exists between the states.

    Algorithm:
        1. Use BFS from start_state to compute distances to all nodes
        2. Use BFS from end_state to compute distances to all nodes
        3. A node is on an optimal path iff:
           distance_from_start[node] + distance_from_end[node] == shortest_path_distance
        4. An edge (u,v) is on an optimal path iff:
           - Both u and v are on optimal paths AND
           - distance_from_start[u] + 1 + distance_from_end[v] == shortest_path_distance

    Example:
        If shortest path is 3 moves, this returns the subgraph containing all
        nodes and edges that participate in any 3-move path between the states.
    """
    if not full_graph:
        return {}

    # Select random states if not provided
    all_states = list(full_graph.keys())

    if start_state is None:
        start_state = random.choice(all_states)

    if end_state is None:
        available_states = [state for state in all_states if state != start_state]
        if not available_states:
            return {}  # Only one state in graph
        end_state = random.choice(available_states)

    if start_state == end_state:
        # Trivial case: same state
        return {start_state: []}

    # BFS from start_state to get distances from start to all nodes
    distances_from_start = _bfs_distances(full_graph, start_state)

    # BFS from end_state to get distances from end to all nodes
    distances_from_end = _bfs_distances(full_graph, end_state)

    # Check if path exists
    if end_state not in distances_from_start:
        return {}  # No path exists

    shortest_path_distance = distances_from_start[end_state]

    # Find all nodes that lie on at least one optimal path
    optimal_nodes = set()
    for node in full_graph:
        if (
            node in distances_from_start
            and node in distances_from_end
            and distances_from_start[node] + distances_from_end[node]
            == shortest_path_distance
        ):
            optimal_nodes.add(node)

    # Build subgraph with only optimal edges
    optimal_subgraph: Dict[StateType, List[StateType]] = {}

    for node in optimal_nodes:
        optimal_neighbors = []

        for neighbor in full_graph[node]:
            # Edge (node -> neighbor) is optimal if:
            # 1. Both nodes are on optimal paths
            # 2. distance_from_start[node] + 1 + distance_from_end[neighbor] == shortest_path_distance
            if (
                neighbor in optimal_nodes
                and neighbor in distances_from_end
                and distances_from_start[node] + 1 + distances_from_end[neighbor]
                == shortest_path_distance
            ):
                optimal_neighbors.append(neighbor)

        optimal_subgraph[node] = optimal_neighbors

    return optimal_subgraph


def _bfs_distances(
    graph: Dict[StateType, List[StateType]], start_state: StateType
) -> Dict[StateType, int]:
    """
    Compute shortest distances from start_state to all reachable nodes using BFS.

    Args:
        graph: Graph as adjacency list
        start_state: Starting node for BFS

    Returns:
        Dictionary mapping each reachable node to its distance from start_state
    """
    distances = {start_state: 0}
    queue = deque([start_state])

    while queue:
        current_state = queue.popleft()
        current_distance = distances[current_state]

        for neighbor in graph.get(current_state, []):
            if neighbor not in distances:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    return distances


# Helper functions for analysis and debugging


def get_graph_stats(graph: Dict) -> Dict[str, any]:
    """Get basic statistics about the state graph."""
    total_states = len(graph)
    total_edges = sum(len(neighbors) for neighbors in graph.values())
    return {
        "total_states": total_states,
        "total_edges": total_edges,
        "avg_degree": total_edges / total_states if total_states > 0 else 0,
    }


def analyze_optimal_subgraph(
    original_graph: Dict[StateType, List[StateType]],
    optimal_subgraph: Dict[StateType, List[StateType]],
    start_state: StateType,
    end_state: StateType,
) -> Dict[str, any]:
    """
    Analyze the optimal path subgraph and provide statistics.

    Returns:
        Dictionary with analysis results including node counts, path info, etc.
    """
    if not optimal_subgraph:
        return {
            "path_exists": False,
            "original_nodes": len(original_graph),
            "optimal_nodes": 0,
            "compression_ratio": 0.0,
        }

    # Calculate shortest path distance by BFS in optimal subgraph
    distances = _bfs_distances(optimal_subgraph, start_state)
    shortest_distance = distances.get(end_state, float("inf"))

    # Count paths (approximate - exact counting is expensive for large graphs)
    original_edges = sum(len(neighbors) for neighbors in original_graph.values())
    optimal_edges = sum(len(neighbors) for neighbors in optimal_subgraph.values())

    return {
        "path_exists": True,
        "shortest_distance": shortest_distance,
        "original_nodes": len(original_graph),
        "optimal_nodes": len(optimal_subgraph),
        "original_edges": original_edges,
        "optimal_edges": optimal_edges,
        "start_state": start_state,
        "end_state": end_state,
    }


def print_state(state_tuple: Tuple[Tuple[int, ...], ...]) -> None:
    """Pretty print a state for debugging."""
    stacks = [list(stack) for stack in state_tuple]
    print("Stacks:", stacks)


def state_to_visual_text(state_tuple: Tuple[Tuple[int, ...], ...]) -> str:
    """
    Convert a game state to a simple text representation for display.

    Args:
        state_tuple: Game state as tuple of tuples

    Returns:
        Multi-line string showing each stack as a list
    """
    stacks = [list(stack) for stack in state_tuple]

    # Show each stack as a simple list, one per line
    lines = []
    for i, stack in enumerate(stacks):
        lines.append(f"Stack {i}: {stack}")

    return "\n".join(lines)


def create_state_summary(state_tuple: Tuple[Tuple[int, ...], ...]) -> str:
    """Create a brief summary of the state for node labels."""
    stacks = [list(stack) for stack in state_tuple]

    # Count non-empty stacks
    non_empty_count = sum(1 for stack in stacks if stack)
    total_blocks = sum(len(stack) for stack in stacks)

    if non_empty_count == 0:
        return "Empty"
    elif non_empty_count == 1:
        stack_idx = next(i for i, stack in enumerate(stacks) if stack)
        return f"All→{stack_idx}"
    else:
        return f"{non_empty_count}stks"


def visualize_game_graph(
    graph: Dict[Tuple[Tuple[int, ...], ...], List[Tuple[Tuple[int, ...], ...]]],
    start_state: Optional[Tuple[Tuple[int, ...], ...]] = None,
    end_state: Optional[Tuple[Tuple[int, ...], ...]] = None,
    title: str = "Block Game State Graph",
    max_nodes: int = 1000,
) -> go.Figure:
    """
    Create an interactive plotly visualization of the game state graph with
    layered layout showing progression from start to end state.

    Args:
        graph: Game state graph as adjacency list
        start_state: Optional start state to highlight
        end_state: Optional end state to highlight
        title: Title for the plot
        max_nodes: Maximum number of nodes to visualize (for performance)

    Returns:
        Plotly figure object that can be displayed or saved
    """
    if not graph:
        return go.Figure().add_annotation(
            text="No graph to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    # For optimal path visualization, we need start and end states
    if not start_state or not end_state:
        print(
            "Warning: For optimal layered layout, both start and end states should be provided"
        )
        # Fallback to spring layout if we don't have start/end
        return _create_spring_layout_graph(graph, start_state, end_state, title, max_nodes)

    # Calculate distances from start state
    distances_from_start = _bfs_distances(graph, start_state)
    max_distance = distances_from_start.get(end_state, 0)

    if max_distance == 0:
        return go.Figure().add_annotation(
            text="Start and end states are the same",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    # Group nodes by their distance from start (this creates our columns)
    layers = {}
    for node, distance in distances_from_start.items():
        if node in graph:  # Only include nodes that are in our graph
            if distance not in layers:
                layers[distance] = []
            layers[distance].append(node)

    # Create positions: x = distance, y = spread vertically within each layer
    pos = {}
    layer_width = 2.0  # Width of each column

    for distance, nodes in layers.items():
        x = distance * layer_width

        # Spread nodes vertically within the layer
        if len(nodes) == 1:
            y_positions = [0]
        else:
            # Create vertical spread from -1 to +1
            y_range = min(2.0, len(nodes) * 0.3)  # Limit vertical spread
            y_positions = []
            for i in range(len(nodes)):
                if len(nodes) == 1:
                    y = 0
                else:
                    y = -y_range / 2 + (i * y_range / (len(nodes) - 1))
                y_positions.append(y)

        # Assign positions
        for node, y in zip(nodes, y_positions):
            pos[node] = (x, y)

    # Create edge traces
    edge_x = []
    edge_y = []
    for state, neighbors in graph.items():
        if state in pos:
            x0, y0 = pos[state]
            for neighbor in neighbors:
                if neighbor in pos:
                    x1, y1 = pos[neighbor]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="lightblue"),
        hoverinfo="none",
        showlegend=False,
    )

    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    hover_text = []

    for state in graph:
        if state in pos:
            x, y = pos[state]
            node_x.append(x)
            node_y.append(y)

            # Color based on distance from start
            distance = distances_from_start[state]
            node_colors.append(distance)

            # Size based on special states
            if state == start_state:
                node_sizes.append(15)
            elif state == end_state:
                node_sizes.append(15)
            else:
                node_sizes.append(8)

            # Create hover text with number of possible moves
            summary = create_state_summary(state)
            visual = state_to_visual_text(state)

            # Calculate number of possible moves
            n_stacks = len(state)
            non_empty_stacks = sum(1 for stack in state if len(stack) > 0)
            num_possible_moves = non_empty_stacks * (n_stacks - 1)

            node_text.append("")  # No text labels on nodes for cleaner look
            hover_text.append(
                f"{visual.replace(chr(10), '<br>')}<br><b>Num possible moves: {num_possible_moves}</b>"
            )

    # Create the main node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Plasma",
            colorbar=dict(title="Moves from Start", thickness=15, len=0.7),
            line=dict(width=1, color="black"),
        ),
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover_text,
        showlegend=False,
    )

    # Create special traces for start and end states
    traces = [edge_trace, node_trace]

    if start_state and start_state in pos:
        x, y = pos[start_state]
        start_visual = state_to_visual_text(start_state)
        # Calculate moves for start state
        n_stacks = len(start_state)
        non_empty_stacks = sum(1 for stack in start_state if len(stack) > 0)
        start_moves = non_empty_stacks * (n_stacks - 1)

        start_trace = go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(
                size=20, color="green", symbol="star", line=dict(width=2, color="darkgreen")
            ),
            name="Start State",
            hovertemplate=f'<b>START STATE</b><br>{start_visual.replace(chr(10), "<br>")}<br><b>Num possible moves: {start_moves}</b><extra></extra>',
        )
        traces.append(start_trace)

    if end_state and end_state in pos:
        x, y = pos[end_state]
        end_visual = state_to_visual_text(end_state)
        # Calculate moves for end state
        n_stacks = len(end_state)
        non_empty_stacks = sum(1 for stack in end_state if len(stack) > 0)
        end_moves = non_empty_stacks * (n_stacks - 1)

        end_trace = go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(
                size=20, color="red", symbol="diamond", line=dict(width=2, color="darkred")
            ),
            name="End State",
            hovertemplate=f'<b>END STATE</b><br>{end_visual.replace(chr(10), "<br>")}<br><b>Num possible moves: {end_moves}</b><extra></extra>',
        )
        traces.append(end_trace)

    # Add column labels at the bottom
    label_y = min(node_y) - 0.5 if node_y else -1
    for distance in range(max_distance + 1):
        x = distance * layer_width
        count = len(layers.get(distance, []))

        label_trace = go.Scatter(
            x=[x],
            y=[label_y],
            mode="text",
            text=[f"Move {distance}<br>({count} states)"],
            textfont=dict(size=10, color="black"),
            showlegend=False,
            hoverinfo="none",
        )
        traces.append(label_trace)

    # Create the figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Layered view: {max_distance} moves from start to end</sub>",
            x=0.5,
            font=dict(size=16),
        ),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=60, l=40, r=40, t=80),
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=False,
            title="Moves from Start State",
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, title="Alternative Paths"
        ),
        plot_bgcolor="white",
    )

    return fig


def _create_spring_layout_graph(graph, start_state, end_state, title, max_nodes):
    """Fallback spring layout for when we don't have proper start/end states."""
    # This is the old implementation for non-optimal graphs
    if len(graph) > max_nodes:
        sampled_nodes = set(random.sample(list(graph.keys()), max_nodes))
        if start_state and start_state in graph:
            sampled_nodes.add(start_state)
        if end_state and end_state in graph:
            sampled_nodes.add(end_state)

        sampled_graph = {}
        for node in sampled_nodes:
            if node in graph:
                sampled_neighbors = [
                    neighbor for neighbor in graph[node] if neighbor in sampled_nodes
                ]
                sampled_graph[node] = sampled_neighbors
        graph = sampled_graph

    # Create NetworkX graph for layout computation
    G = nx.DiGraph()
    for state, neighbors in graph.items():
        G.add_node(state)
        for neighbor in neighbors:
            if neighbor in graph:
                G.add_edge(state, neighbor)

    # Compute spring layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Create traces (simplified version of the original)
    edge_x, edge_y = [], []
    for state, neighbors in graph.items():
        x0, y0 = pos[state]
        for neighbor in neighbors:
            if neighbor in pos:
                x1, y1 = pos[neighbor]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    node_x = [pos[state][0] for state in graph if state in pos]
    node_y = [pos[state][1] for state in graph if state in pos]
    hover_text = [
        f"{create_state_summary(state)}<br>{state_to_visual_text(state).replace(chr(10), '<br>')}"
        for state in graph
        if state in pos
    ]

    traces = [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="lightgray"),
            hoverinfo="none",
            showlegend=False,
        ),
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=8),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text,
            showlegend=False,
        ),
    ]

    return go.Figure(data=traces).update_layout(title=title, showlegend=False)


# Example usage and main execution:
if __name__ == "__main__":
    complete_graph = compute_block_game_state_graph(n_blocks=6, n_stacks=3)

    # Show basic statistics
    stats = get_graph_stats(complete_graph)
    print(f"Complete graph statistics:")
    print(f"  Total states: {stats['total_states']:,}")
    print(f"  Total edges: {stats['total_edges']:,}")
    print(f"  Average degree: {stats['avg_degree']:.2f}")
    print()

    # Extract optimal path subgraph with random states
    print("Extracting optimal path subgraph between two random states...")
    optimal_subgraph = extract_optimal_path_subgraph(complete_graph, None, None)

    if optimal_subgraph:
        # Get the start and end states that were randomly selected
        # We need to find them by analyzing the subgraph
        distances_from_nodes = {}
        for node in optimal_subgraph:
            distances_from_nodes[node] = _bfs_distances(optimal_subgraph, node)

        # Find the start state (distance 0 from itself, max distance to some other node)
        start_state = None
        end_state = None
        max_distance = 0

        for node, distances in distances_from_nodes.items():
            max_dist_from_node = max(distances.values()) if distances else 0
            if max_dist_from_node > max_distance:
                max_distance = max_dist_from_node
                start_state = node
                # Find the end state (node with maximum distance from start)
                end_state = max(distances.items(), key=lambda x: x[1])[0]

        # Analyze the results
        analysis = analyze_optimal_subgraph(
            complete_graph, optimal_subgraph, start_state, end_state
        )

        fig = visualize_game_graph(
            optimal_subgraph,
            start_state=start_state,
            end_state=end_state,
            title=f"Optimal Path Subgraph (Distance: {analysis['shortest_distance']} moves)",
        )
        pyo.plot(fig, filename="optimal_subgraph.html", auto_open=True)

    else:
        print("No optimal subgraph found (this shouldn't happen with connected graph)")
