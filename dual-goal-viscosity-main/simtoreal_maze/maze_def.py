"""
maze_def.py — Real-room maze specification.

Mirrors OGBench's `ogbench.locomaze.maze.MazeEnv` data layout (a 2-D grid where
0 = free cell and 1 = wall) but lets you describe an actual room in real-world
metres instead of the simulator's `maze_unit=4.0` blocks. Everything below is
just bookkeeping; no MuJoCo, no gym.

A maze JSON looks like this:

    {
        "name": "lab_north",
        "cell_size_m": 1.0,            // metres per grid cell
        "origin_xy_m": [0.0, 0.0],     // world (x, y) of the *centre* of cell (0,0)
        "wall_thickness_m": 0.05,      // visualisation only
        "maze_map": [                  // row 0 is the top of the map
            [1,1,1,1,1,1,1,1],
            [1,0,0,0,0,1,0,1],
            [1,0,1,1,0,1,0,1],
            [1,0,1,0,0,0,0,1],
            [1,0,0,0,1,1,0,1],
            [1,1,1,1,1,1,1,1]
        ],
        "start_cell": [1, 1],          // [row, col]
        "goal_cell":  [3, 3]
    }

Coordinate convention (matches OGBench point.py):
    world_x = (col - origin_col) * cell_size_m + origin_xy_m[0]
    world_y = (row - origin_row) * cell_size_m + origin_xy_m[1]
where (origin_row, origin_col) is the cell at `origin_xy_m`. We default it to
(0, 0) so cell (0,0)'s centre is exactly origin_xy_m.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class MazeSpec:
    name: str
    cell_size_m: float
    origin_xy_m: Tuple[float, float]
    maze_map: np.ndarray  # int array, shape (H, W); 0 = free, 1 = wall
    start_cell: Tuple[int, int]
    goal_cell: Tuple[int, int]
    wall_thickness_m: float = 0.05

    @classmethod
    def from_json(cls, path: str | Path) -> "MazeSpec":
        with open(path) as f:
            d = json.load(f)
        m = np.asarray(d["maze_map"], dtype=np.int32)
        if m.ndim != 2:
            raise ValueError(f"maze_map must be 2-D, got shape {m.shape}")
        return cls(
            name=d.get("name", Path(path).stem),
            cell_size_m=float(d["cell_size_m"]),
            origin_xy_m=tuple(float(v) for v in d["origin_xy_m"]),
            maze_map=m,
            start_cell=tuple(int(v) for v in d["start_cell"]),
            goal_cell=tuple(int(v) for v in d["goal_cell"]),
            wall_thickness_m=float(d.get("wall_thickness_m", 0.05)),
        )

    # ------- coordinate conversions ----------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.maze_map.shape)

    def cell_center_xy(self, cell: Tuple[int, int]) -> np.ndarray:
        """Return the (x, y) metres of a cell's centre."""
        r, c = cell
        return np.array(
            [c * self.cell_size_m + self.origin_xy_m[0],
             r * self.cell_size_m + self.origin_xy_m[1]],
            dtype=np.float32,
        )

    def xy_to_cell(self, xy: np.ndarray) -> Tuple[int, int]:
        """Inverse of cell_center_xy; rounds to the nearest cell."""
        c = int(round((xy[0] - self.origin_xy_m[0]) / self.cell_size_m))
        r = int(round((xy[1] - self.origin_xy_m[1]) / self.cell_size_m))
        return (r, c)

    def is_free(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        H, W = self.shape
        if not (0 <= r < H and 0 <= c < W):
            return False
        return int(self.maze_map[r, c]) == 0

    # ------- cell sets ------------------------------------------------------

    def free_cells(self) -> List[Tuple[int, int]]:
        H, W = self.shape
        return [(r, c) for r in range(H) for c in range(W)
                if self.maze_map[r, c] == 0]

    def vertex_cells(self) -> List[Tuple[int, int]]:
        """OGBench-style vertex cells: free cells that are *not* hallways
        (i.e. corners and intersections). Used as candidate goal cells so
        we never sample a goal in the middle of a corridor.
        """
        H, W = self.shape
        out = []
        for r, c in self.free_cells():
            up = self.maze_map[r - 1, c] if r > 0 else 1
            down = self.maze_map[r + 1, c] if r < H - 1 else 1
            left = self.maze_map[r, c - 1] if c > 0 else 1
            right = self.maze_map[r, c + 1] if c < W - 1 else 1
            # Pure horizontal hallway → skip.
            if up == 0 and down == 0 and left == 1 and right == 1:
                continue
            # Pure vertical hallway → skip.
            if left == 0 and right == 0 and up == 1 and down == 1:
                continue
            out.append((r, c))
        return out

    # ------- planning -------------------------------------------------------

    def bfs_path(self, start: Tuple[int, int],
                 goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Shortest grid path from start to goal (4-connected). Empty if none."""
        if start == goal:
            return [start]
        if not (self.is_free(start) and self.is_free(goal)):
            return []
        H, W = self.shape
        prev = {start: None}
        queue: List[Tuple[int, int]] = [start]
        while queue:
            cur = queue.pop(0)
            if cur == goal:
                break
            r, c = cur
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                nb = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and self.maze_map[nr, nc] == 0 and nb not in prev:
                    prev[nb] = cur
                    queue.append(nb)
        if goal not in prev:
            return []
        path = [goal]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        path.reverse()
        return path
