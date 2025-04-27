import numpy as np
import math

class PathPlanner:
    def __init__(self, grid_size, grid_resolution, target_pos):
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.target_pos = target_pos
        self.grid = np.zeros(grid_size)

    def update_occupancy_grid(self, objects, depth_map, frame_shape):
        self.grid = np.zeros(self.grid_size)
        for label, center_x, depth in objects:
            if depth < 5.0:
                grid_x = int(self.target_pos[0] / self.grid_resolution)
                grid_y = int((center_x - 0.5) * 5 / self.grid_resolution + self.grid_size[1] // 2)
                if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
                    self.grid[grid_x, grid_y] = 1

    def plan_path(self, user_pos):
        path = [user_pos]
        current_pos = user_pos
        while np.linalg.norm(np.array(current_pos) - np.array(self.target_pos)) > self.grid_resolution:
            next_pos = [
                current_pos[0] + self.grid_resolution * np.sign(self.target_pos[0] - current_pos[0]),
                current_pos[1] + self.grid_resolution * np.sign(self.target_pos[1] - current_pos[1])
            ]
            grid_x = int(next_pos[0] / self.grid_resolution)
            grid_y = int(next_pos[1] / self.grid_resolution + self.grid_size[1] // 2)
            if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1] and self.grid[grid_x, grid_y] == 0:
                path.append(next_pos)
                current_pos = next_pos
            else:
                break
        return path

    def get_navigation_instruction(self, path, user_pos):
        if not path or len(path) < 2:
            return "No clear path. Stop."
        next_pos = path[1]
        dx = next_pos[0] - user_pos[0]
        dy = next_pos[1] - user_pos[1]
        angle = math.atan2(dy, dx) * 180 / math.pi
        angle = (angle + 180) % 360 - 180
        distance = np.linalg.norm(np.array(next_pos) - np.array(user_pos))
        if abs(angle) > 15:
            return f"Turn {'left' if angle > 0 else 'right'} {int(abs(angle))} degrees."
        return f"Move forward {distance:.1f} meters."