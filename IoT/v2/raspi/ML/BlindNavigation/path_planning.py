import numpy as np
from heapq import heappop, heappush

class PathPlanner:
    def __init__(self, grid_size, grid_resolution, target_pos):
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.target_pos = target_pos
        self.occupancy_grid = np.zeros(grid_size, dtype=np.float32)

    def update_occupancy_grid(self, objects, depth_map, frame_shape):
        """Update occupancy grid with detected objects."""
        self.occupancy_grid.fill(0)

        for label, center_x, depth_meters in objects:
            grid_x = int(depth_meters / self.grid_resolution)
            grid_y = int((center_x - 0.5) * 4)
            grid_x = min(max(grid_x, 0), self.grid_size[0] - 1)
            grid_y = min(max(grid_y + self.grid_size[1] // 2, 0), self.grid_size[1] - 1)
            self.occupancy_grid[grid_x, grid_y] = 1.0

    def plan_path(self, start_pos):
        """Plan path using A*."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        start = (int(start_pos[0]), int(start_pos[1]))
        goal = (int(self.target_pos[0]), int(self.target_pos[1]))

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current_f, current = heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_size[0] and 0 <= neighbor[1] < self.grid_size[1]:
                    if self.occupancy_grid[neighbor[0], neighbor[1]] > 0.5:
                        continue
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def get_navigation_instruction(self, path, current_pos, yaw=0):
        """Generate navigation instruction based on path and user orientation."""
        if not path:
            return "No clear path to target. Stop and wait."
        
        next_pos = path[0]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        # Calculate required heading
        target_angle = math.degrees(math.atan2(dy, dx))
        relative_angle = (target_angle - yaw) % 360
        if relative_angle > 180:
            relative_angle -= 360

        if abs(relative_angle) > 15:
            direction = "right" if relative_angle > 0 else "left"
            return f"Turn {direction} {int(abs(relative_angle))} degrees."
        
        if dx > 0:
            return "Move forward 1 meter."
        elif dx < 0:
            return "Move backward 1 meter."
        elif dy > 0:
            return "Move right 1 meter."
        elif dy < 0:
            return "Move left 1 meter."
        return "Continue forward."