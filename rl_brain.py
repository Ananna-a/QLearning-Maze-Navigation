from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

State = Tuple[int, int]
Action = int


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        actions: Sequence[Action],
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        if not actions:
            raise ValueError("Action space cannot be empty.")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1].")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if not (0.0 <= epsilon_min <= 1.0):
            raise ValueError("epsilon_min must be in [0, 1].")
        if not (0.0 < epsilon_decay <= 1.0):
            raise ValueError("epsilon_decay must be in (0, 1].")

        self.actions = list(actions)
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: Dict[State, List[float]] = {}

    def _ensure_state(self, state: State) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def choose_action(self, state: State, training: bool = True) -> Action:
        q_values = self._ensure_state(state)

        if training and random.random() < self.epsilon:
            return random.choice(self.actions)

        max_q = max(q_values)
        best_actions = [
            action for action, value in zip(self.actions, q_values) if value == max_q
        ]
        return random.choice(best_actions)

    def learn(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        if action not in self.action_to_index:
            raise ValueError(f"Action {action} is not in the action space.")

        current_values = self._ensure_state(state)
        next_values = self._ensure_state(next_state)

        action_index = self.action_to_index[action]
        predict = current_values[action_index]
        target = reward if done else reward + self.gamma * max(next_values)

        current_values[action_index] += self.learning_rate * (target - predict)
        return current_values[action_index]

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)
