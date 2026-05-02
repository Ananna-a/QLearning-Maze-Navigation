from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, Optional, Set, Tuple

State = Tuple[int, int]
Action = int

ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    0: (0, -1),  # up
    1: (0, 1),  # down
    2: (-1, 0),  # left
    3: (1, 0),  # right
}


class MazeEnv:
    """Grid maze environment with a Gym-like API (reset/step)."""

    def __init__(
        self,
        width: int,
        height: int,
        start: State,
        goal: State,
        walls: Optional[Iterable[State]] = None,
        traps: Optional[Iterable[State]] = None,
        step_penalty: float = -0.01,
        wall_penalty: float = -1.0,
        trap_penalty: float = -1.0,
        goal_reward: float = 1.0,
        max_steps: int = 100,
        dynamic_wall_min: int = 0,
        dynamic_wall_max: int = 0,
        dynamic_wall_refresh_max_tries: int = 200,
        dynamic_wall_candidates: Optional[Iterable[State]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal

        self.static_walls = set(walls or [])
        self.dynamic_walls: Set[State] = set()
        self.walls = set(self.static_walls)
        self.traps = set(traps or [])

        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.trap_penalty = trap_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.dynamic_wall_min = dynamic_wall_min
        self.dynamic_wall_max = dynamic_wall_max
        self.dynamic_wall_refresh_max_tries = dynamic_wall_refresh_max_tries
        self.dynamic_wall_candidates = set(dynamic_wall_candidates or [])

        self._validate_layout()
        self.action_space = tuple(ACTION_DELTAS.keys())

        self._refresh_dynamic_walls()

        self.state: State = self.start
        self.steps_taken = 0
        self.done = False

    def _validate_layout(self) -> None:
        if self.width <= 1 or self.height <= 1:
            raise ValueError("Maze width/height must be larger than 1.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")
        if self.dynamic_wall_min < 0 or self.dynamic_wall_max < 0:
            raise ValueError("dynamic wall counts cannot be negative.")
        if self.dynamic_wall_min > self.dynamic_wall_max:
            raise ValueError("dynamic_wall_min must be <= dynamic_wall_max.")
        if self.dynamic_wall_refresh_max_tries <= 0:
            raise ValueError("dynamic_wall_refresh_max_tries must be greater than 0.")

        for cell_name, cell in (("start", self.start), ("goal", self.goal)):
            if not self._in_bounds(cell):
                raise ValueError(f"{cell_name} cell {cell} is outside the maze.")

        for wall in self.static_walls:
            if not self._in_bounds(wall):
                raise ValueError(f"Wall cell {wall} is outside the maze.")
        for trap in self.traps:
            if not self._in_bounds(trap):
                raise ValueError(f"Trap cell {trap} is outside the maze.")

        forbidden = self.static_walls | self.traps
        if self.start in forbidden:
            raise ValueError("Start cell cannot be a wall or trap.")
        if self.goal in forbidden:
            raise ValueError("Goal cell cannot be a wall or trap.")

        for candidate in self.dynamic_wall_candidates:
            if not self._in_bounds(candidate):
                raise ValueError(
                    f"Dynamic wall candidate {candidate} is outside the maze."
                )
            if candidate in forbidden:
                raise ValueError(
                    f"Dynamic wall candidate {candidate} conflicts with wall/trap."
                )
            if candidate == self.start or candidate == self.goal:
                raise ValueError(
                    f"Dynamic wall candidate {candidate} cannot be start/goal."
                )

        if self.dynamic_wall_candidates and self.dynamic_wall_max > len(
            self.dynamic_wall_candidates
        ):
            raise ValueError(
                "dynamic_wall_max exceeds number of dynamic wall candidate cells."
            )

        if not self._has_safe_path(self.static_walls | self.traps):
            raise ValueError("Maze layout has no safe path from start to goal.")

    def _has_safe_path(self, blocked: Set[State]) -> bool:
        queue = deque([self.start])
        visited = {self.start}

        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True

            for dx, dy in ACTION_DELTAS.values():
                next_state = (current[0] + dx, current[1] + dy)
                if not self._in_bounds(next_state):
                    continue
                if next_state in blocked or next_state in visited:
                    continue
                visited.add(next_state)
                queue.append(next_state)

        return False

    def _refresh_dynamic_walls(self) -> None:
        self.dynamic_walls.clear()
        self.walls = set(self.static_walls)

        if self.dynamic_wall_max == 0:
            return

        dynamic_count = random.randint(self.dynamic_wall_min, self.dynamic_wall_max)
        if dynamic_count == 0:
            return

        blocked_base = self.static_walls | self.traps
        if self.dynamic_wall_candidates:
            candidates = [
                cell
                for cell in self.dynamic_wall_candidates
                if cell not in blocked_base and cell != self.start and cell != self.goal
            ]
        else:
            candidates = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) not in blocked_base
                and (x, y) != self.start
                and (x, y) != self.goal
            ]

        if dynamic_count > len(candidates):
            raise ValueError("Not enough free cells to place dynamic walls.")

        for _ in range(self.dynamic_wall_refresh_max_tries):
            sampled = set(random.sample(candidates, dynamic_count))
            if self._has_safe_path(blocked_base | sampled):
                self.dynamic_walls = sampled
                self.walls = self.static_walls | self.dynamic_walls
                return

        raise RuntimeError("Failed to generate solvable dynamic walls after retries.")

    def _in_bounds(self, state: State) -> bool:
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def reset(self) -> State:
        self._refresh_dynamic_walls()
        self.state = self.start
        self.steps_taken = 0
        self.done = False
        return self.state

    def step(self, action: Action) -> Tuple[State, float, bool, dict]:
        if action not in ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action}")
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() before step().")

        self.steps_taken += 1

        dx, dy = ACTION_DELTAS[action]
        nx = self.state[0] + dx
        ny = self.state[1] + dy
        next_state = (nx, ny)

        reward = self.step_penalty
        event = "move"

        if (not self._in_bounds(next_state)) or (next_state in self.walls):
            # Keep the episode alive and penalize collision; this improves
            # learnability on larger maps with dynamic obstacles.
            reward = self.wall_penalty
            event = "wall"
        elif next_state in self.traps:
            self.state = next_state
            reward = self.trap_penalty
            self.done = True
            event = "trap"
        elif next_state == self.goal:
            self.state = next_state
            reward = self.goal_reward
            self.done = True
            event = "goal"
        else:
            self.state = next_state

        if not self.done and self.steps_taken >= self.max_steps:
            self.done = True
            event = "timeout"

        info = {"event": event, "steps": self.steps_taken}
        return self.state, reward, self.done, info

    def render_ascii(self) -> str:
        symbols = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = (x, y)
                if cell == self.state:
                    row.append("A")
                elif cell == self.start:
                    row.append("S")
                elif cell == self.goal:
                    row.append("G")
                elif cell in self.walls:
                    row.append("#")
                elif cell in self.traps:
                    row.append("T")
                else:
                    row.append(".")
            symbols.append(" ".join(row))
        return "\n".join(symbols)


def make_default_env(max_steps: int = 120) -> MazeEnv:
    """Create a 10x10 maze with per-episode refreshed dynamic walls."""
    return MazeEnv(
        width=10,
        height=10,
        start=(0, 0),
        goal=(9, 9),
        walls={
            (3, 2),
            (3, 3),
            (3, 6),
            (3, 7),
            (6, 2),
            (6, 3),
            (6, 6),
            (6, 7),
            (2, 4),
            (5, 4),
            (7, 4),
            (2, 8),
            (5, 8),
            (7, 8),
        },
        traps={(5, 6), (7, 1)},
        max_steps=max_steps,
        dynamic_wall_min=2,
        dynamic_wall_max=3,
        dynamic_wall_candidates={
            (1, 6),
            (1, 7),
            (2, 6),
            (2, 7),
            (4, 1),
            (4, 2),
            (5, 1),
            (5, 2),
            (4, 6),
            (4, 7),
            (5, 7),
            (7, 2),
            (8, 2),
            (7, 6),
            (8, 6),
            (7, 7),
            (8, 7),
        },
    )
