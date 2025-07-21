"""Experimental code for breadth first graph search for min distance calculation with caching."""


class MinDistanceCache:
    def __init__(self):
        self._cache = {}

    def _get_key(self, node1, node2):
        return (hash(node1), hash(node2))

    def get(self, node1, node2):
        key = self._get_key(node1, node2)
        key_reverse = self._get_key(node2, node1)
        if key in self._cache:
            return self._cache[key]
        elif key_reverse in self._cache:
            return self._cache[key_reverse]
        return None

    def set(self, node1, node2, value):
        self._cache[self._get_key(node1, node2)] = value


def bfs_min_dist_search_with_caching(
    graph, start, goal, min_distance_cache: MinDistanceCache
) -> int:
    if start == goal:
        return 0
    if min_distance_cache.get(start, goal) is not None:
        return min_distance_cache.get(start, goal)

    visited = set()
    search_nodes_at_cur_depth = graph[start]
    distance_covered = 0
    we_will_for_sure_reach_in: None | int = None
    goal_reached = False

    while not goal_reached:
        # Increments/decrements
        distance_covered += 1
        if we_will_for_sure_reach_in is not None:
            we_will_for_sure_reach_in -= 1

        if we_will_for_sure_reach_in == 0:
            return distance_covered  # Reached the goal through a cached min distance

        # Search the nodes at this depth and append their neighbors to the next depth
        next_depth = []
        for neighbor in search_nodes_at_cur_depth:
            if neighbor == goal:
                min_distance_cache.set(start, neighbor, distance_covered)
                return distance_covered  # Reached the goal through BFS

            if neighbor in visited:
                continue

            # Mark the neighbor as visited
            visited.add(neighbor)

            if we_will_for_sure_reach_in is None:
                # This have been skipping search paths that could have reached this neighbor earlier
                min_distance_cache.set(start, neighbor, distance_covered)

            cached_min_dist_from_nghbr_to_goal = min_distance_cache.get(neighbor, goal)
            if cached_min_dist_from_nghbr_to_goal is None:
                # Keep searching down this path
                next_depth.extend(graph[neighbor])
            elif (
                we_will_for_sure_reach_in is None
                or cached_min_dist_from_nghbr_to_goal <= we_will_for_sure_reach_in
            ):
                # We found a cached min distance that is less than our distance value we
                # know we will reach the goal in.
                we_will_for_sure_reach_in = cached_min_dist_from_nghbr_to_goal

        search_nodes_at_cur_depth = next_depth


# Example usage
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}
graph_min_distance_cache = MinDistanceCache()
print(bfs_min_dist_search_with_caching(graph, "A", "F", graph_min_distance_cache))
