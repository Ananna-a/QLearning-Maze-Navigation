from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import tkinter as tk
except Exception:  # pragma: no cover - depends on local GUI installation
    tk = None

from maze_env import MazeEnv, State, make_default_env
from rl_brain import QLearningAgent


DEFAULT_REPLAY_RECORD_FILE = Path(__file__).with_name("last_training_record.json")


@dataclass
class EpisodeTrace:
    episode: int
    path: List[State]
    total_reward: float
    success: bool
    epsilon: float
    dynamic_walls: Tuple[State, ...]


@dataclass
class TrainStats:
    rewards: List[float]
    successes: List[int]
    best_path: List[State]
    episode_traces: List[EpisodeTrace]


class TkMazeRenderer:
    """Simple Tkinter renderer for real-time training visualization."""

    def __init__(self, env: MazeEnv, cell_size: int = 80) -> None:
        if tk is None:
            raise RuntimeError("Tkinter is unavailable in this Python installation.")

        self.env = env
        self.cell_size = cell_size
        self.alive = True

        self.root = tk.Tk()
        self.root.title("Q-Learning Maze")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value="Waiting to start training...")

        canvas_width = env.width * cell_size
        canvas_height = env.height * cell_size
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_width,
            height=canvas_height,
            bg="#f7f4ea",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=(12, 8))

        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            font=("Consolas", 11),
        )
        self.status_label.pack(fill="x", padx=12, pady=(0, 12))

        self.path_items: List[int] = []
        self.agent_item: Optional[int] = None

        self._draw_static_cells()
        self.draw_agent(env.start)
        self._refresh_window()

    def _on_close(self) -> None:
        self.alive = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def _cell_bounds(self, state: State) -> Tuple[int, int, int, int]:
        x, y = state
        x1 = x * self.cell_size
        y1 = y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return x1, y1, x2, y2

    def _cell_center(self, state: State) -> Tuple[int, int]:
        x1, y1, x2, y2 = self._cell_bounds(state)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _draw_static_cells(self) -> None:
        self.canvas.delete("all")
        self.path_items.clear()
        self.agent_item = None

        dynamic_walls = getattr(self.env, "dynamic_walls", set())

        for y in range(self.env.height):
            for x in range(self.env.width):
                cell = (x, y)
                x1, y1, x2, y2 = self._cell_bounds(cell)

                fill = "#f3eee2"
                text = ""
                text_color = "#222222"

                if cell in self.env.walls:
                    if cell in dynamic_walls:
                        fill = "#5c6770"
                        text = "D"
                        text_color = "#ffffff"
                    else:
                        fill = "#2f3e46"
                elif cell in self.env.traps:
                    fill = "#bc4749"
                    text = "T"
                    text_color = "#ffffff"
                elif cell == self.env.goal:
                    fill = "#2a9d8f"
                    text = "G"
                    text_color = "#ffffff"
                elif cell == self.env.start:
                    fill = "#457b9d"
                    text = "S"
                    text_color = "#ffffff"

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=fill,
                    outline="#d5c9a2",
                    width=2,
                )
                if text:
                    self.canvas.create_text(
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                        text=text,
                        fill=text_color,
                        font=("Consolas", 14, "bold"),
                    )

    def redraw_maze(self, state: State) -> None:
        self._draw_static_cells()
        self.draw_agent(state)
        self._refresh_window()

    def _clear_path(self) -> None:
        for item in self.path_items:
            self.canvas.delete(item)
        self.path_items.clear()

    def draw_path(self, path: List[State], color: str = "#f4a261") -> None:
        self._clear_path()
        if len(path) < 2:
            return

        for idx in range(len(path) - 1):
            x1, y1 = self._cell_center(path[idx])
            x2, y2 = self._cell_center(path[idx + 1])
            line = self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                width=4,
                capstyle=tk.ROUND,
            )
            self.path_items.append(line)

    def draw_agent(self, state: State) -> None:
        x1, y1, x2, y2 = self._cell_bounds(state)
        pad = max(6, self.cell_size // 6)
        coords = (x1 + pad, y1 + pad, x2 - pad, y2 - pad)

        if self.agent_item is None:
            self.agent_item = self.canvas.create_oval(
                *coords,
                fill="#ffd166",
                outline="#6a4c0c",
                width=2,
            )
        else:
            self.canvas.coords(self.agent_item, *coords)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _refresh_window(self) -> None:
        if not self.alive:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.alive = False

    def render_step(
        self,
        phase: str,
        episode: int,
        step: int,
        reward: float,
        epsilon: float,
        state: State,
        path: List[State],
    ) -> None:
        if not self.alive:
            return

        self.draw_path(path)
        self.draw_agent(state)
        self.set_status(
            f"Phase: {phase} | Episode: {episode} | Step: {step} | "
            f"Reward: {reward:.2f} | Epsilon: {epsilon:.3f}"
        )
        self._refresh_window()

    def keep_open(self) -> None:
        if self.alive:
            self.root.mainloop()


def train_agent(
    env: MazeEnv,
    agent: QLearningAgent,
    episodes: int,
    max_steps: int,
    renderer: Optional[TkMazeRenderer] = None,
    train_delay: float = 0.02,
    render_every: int = 10,
    log_interval: int = 20,
) -> TrainStats:
    rewards: List[float] = []
    successes: List[int] = []
    best_path: List[State] = []
    episode_traces: List[EpisodeTrace] = []
    best_length = 10**9

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        path = [state]
        dynamic_walls_snapshot = tuple(sorted(getattr(env, "dynamic_walls", set())))

        if renderer and renderer.alive:
            renderer.redraw_maze(state)

        for step in range(1, max_steps + 1):
            action = agent.choose_action(state, training=True)
            next_state, reward, done, _ = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            path.append(state)

            should_render = (
                renderer
                and renderer.alive
                and (episode <= 120 or episode % max(1, render_every) == 0)
            )
            if should_render:
                renderer.render_step(
                    phase="Training",
                    episode=episode,
                    step=step,
                    reward=episode_reward,
                    epsilon=agent.epsilon,
                    state=state,
                    path=path,
                )
                if train_delay > 0:
                    time.sleep(train_delay)

            if done:
                break

        success = state == env.goal
        rewards.append(episode_reward)
        successes.append(1 if success else 0)
        episode_traces.append(
            EpisodeTrace(
                episode=episode,
                path=path.copy(),
                total_reward=episode_reward,
                success=success,
                epsilon=agent.epsilon,
                dynamic_walls=dynamic_walls_snapshot,
            )
        )

        if success and len(path) < best_length:
            best_length = len(path)
            best_path = path.copy()

        agent.decay_epsilon()

        if episode % log_interval == 0 or episode == 1 or episode == episodes:
            window = min(log_interval, len(rewards))
            avg_reward = sum(rewards[-window:]) / window
            success_rate = sum(successes[-window:]) / window
            print(
                f"[Episode {episode:4d}] avg_reward={avg_reward:7.3f} "
                f"success_rate={success_rate:6.1%} epsilon={agent.epsilon:.3f} "
                f"q_states={agent.q_table_size}"
            )

    return TrainStats(
        rewards=rewards,
        successes=successes,
        best_path=best_path,
        episode_traces=episode_traces,
    )


def _apply_dynamic_wall_snapshot(
    env: MazeEnv, dynamic_walls: Tuple[State, ...]
) -> None:
    if not hasattr(env, "static_walls"):
        return
    dynamic_set = set(dynamic_walls)
    env.dynamic_walls = dynamic_set
    env.walls = set(env.static_walls) | dynamic_set


def replay_training_history(
    env: MazeEnv,
    stats: TrainStats,
    renderer: Optional[TkMazeRenderer],
    fast_stride: int = 1,
    slow_tail_episodes: int = 8,
    fast_delay: float = 0.001,
    slow_delay: float = 0.05,
    target_seconds: float = 18.0,
) -> None:
    if not renderer or not renderer.alive or not stats.episode_traces:
        return

    fast_stride = max(1, fast_stride)
    slow_tail_episodes = max(1, slow_tail_episodes)

    traces = stats.episode_traces
    tail_start = max(1, len(traces) - slow_tail_episodes + 1)

    replay_plan: List[Tuple[EpisodeTrace, str]] = []
    for trace in traces:
        if trace.episode >= tail_start:
            replay_plan.append((trace, "slow"))
        elif trace.episode == 1 or trace.episode % fast_stride == 0:
            replay_plan.append((trace, "fast"))

    fast_episode_count = sum(1 for _, mode in replay_plan if mode == "fast")
    slow_step_count = sum(
        max(1, len(trace.path) - 1) for trace, mode in replay_plan if mode == "slow"
    )

    effective_fast_delay = max(0.0, fast_delay)
    effective_slow_delay = max(0.0, slow_delay)

    if target_seconds > 0:
        if fast_episode_count > 0 and slow_step_count > 0:
            fast_budget = target_seconds * 0.35
            slow_budget = max(0.0, target_seconds - fast_budget)
            effective_fast_delay = fast_budget / fast_episode_count
            effective_slow_delay = slow_budget / slow_step_count
            effective_slow_delay = max(
                effective_slow_delay,
                effective_fast_delay * 2.0,
            )
        elif fast_episode_count > 0:
            effective_fast_delay = target_seconds / fast_episode_count
            effective_slow_delay = 0.0
        elif slow_step_count > 0:
            effective_fast_delay = 0.0
            effective_slow_delay = target_seconds / slow_step_count

    effective_fast_delay = min(effective_fast_delay, 0.08)
    effective_slow_delay = min(effective_slow_delay, 0.2)

    total_replays = len(replay_plan)
    for replay_index, (trace, mode) in enumerate(replay_plan, start=1):
        if not renderer.alive or not trace.path:
            break

        phase = "Replay-Slow" if mode == "slow" else "Replay-Fast"
        _apply_dynamic_wall_snapshot(env, trace.dynamic_walls)
        renderer.redraw_maze(trace.path[0])

        step_count = len(trace.path) - 1
        if mode == "fast":
            renderer.render_step(
                phase=f"{phase} {replay_index}/{total_replays}",
                episode=trace.episode,
                step=max(0, step_count),
                reward=trace.total_reward,
                epsilon=trace.epsilon,
                state=trace.path[-1],
                path=trace.path,
            )
            if effective_fast_delay > 0:
                time.sleep(effective_fast_delay)
            continue

        if step_count <= 0:
            renderer.render_step(
                phase=f"{phase} {replay_index}/{total_replays}",
                episode=trace.episode,
                step=0,
                reward=trace.total_reward,
                epsilon=trace.epsilon,
                state=trace.path[0],
                path=trace.path,
            )
            if effective_slow_delay > 0:
                time.sleep(effective_slow_delay)
            continue

        for step in range(1, step_count + 1):
            if not renderer.alive:
                break
            partial_path = trace.path[: step + 1]
            renderer.render_step(
                phase=f"{phase} {replay_index}/{total_replays}",
                episode=trace.episode,
                step=step,
                reward=trace.total_reward,
                epsilon=trace.epsilon,
                state=partial_path[-1],
                path=partial_path,
            )
            if effective_slow_delay > 0:
                time.sleep(effective_slow_delay)


def save_training_record(stats: TrainStats, record_file: Path) -> None:
    payload = {
        "version": 1,
        "rewards": stats.rewards,
        "successes": stats.successes,
        "best_path": [[x, y] for x, y in stats.best_path],
        "episode_traces": [
            {
                "episode": trace.episode,
                "path": [[x, y] for x, y in trace.path],
                "total_reward": trace.total_reward,
                "success": trace.success,
                "epsilon": trace.epsilon,
                "dynamic_walls": [[x, y] for x, y in trace.dynamic_walls],
            }
            for trace in stats.episode_traces
        ],
    }
    record_file.write_text(json.dumps(payload), encoding="utf-8")


def load_training_record(record_file: Path) -> Optional[TrainStats]:
    if not record_file.exists():
        return None

    try:
        payload = json.loads(record_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to load replay record: {exc}")
        return None

    try:
        raw_traces = payload.get("episode_traces", [])
        episode_traces = [
            EpisodeTrace(
                episode=int(raw_trace.get("episode", idx + 1)),
                path=[
                    (int(cell[0]), int(cell[1])) for cell in raw_trace.get("path", [])
                ],
                total_reward=float(raw_trace.get("total_reward", 0.0)),
                success=bool(raw_trace.get("success", False)),
                epsilon=float(raw_trace.get("epsilon", 0.0)),
                dynamic_walls=tuple(
                    (int(cell[0]), int(cell[1]))
                    for cell in raw_trace.get("dynamic_walls", [])
                ),
            )
            for idx, raw_trace in enumerate(raw_traces)
        ]

        rewards = [trace.total_reward for trace in episode_traces]
        successes = [1 if trace.success else 0 for trace in episode_traces]
        best_path = [
            (int(cell[0]), int(cell[1])) for cell in payload.get("best_path", [])
        ]

        return TrainStats(
            rewards=rewards,
            successes=successes,
            best_path=best_path,
            episode_traces=episode_traces,
        )
    except Exception as exc:
        print(f"Replay record format is invalid: {exc}")
        return None


def run_greedy_episode(
    env: MazeEnv,
    agent: QLearningAgent,
    max_steps: int,
    renderer: Optional[TkMazeRenderer] = None,
    delay: float = 0.15,
) -> Tuple[List[State], bool, float]:
    state = env.reset()
    path = [state]
    total_reward = 0.0

    if renderer and renderer.alive:
        renderer.redraw_maze(state)

    for step in range(1, max_steps + 1):
        action = agent.choose_action(state, training=False)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_reward += reward
        path.append(state)

        if renderer and renderer.alive:
            renderer.render_step(
                phase="Greedy Demo",
                episode=0,
                step=step,
                reward=total_reward,
                epsilon=agent.epsilon,
                state=state,
                path=path,
            )
            if delay > 0:
                time.sleep(delay)

        if done:
            break

    return path, state == env.goal, total_reward


def evaluate_policy(
    env: MazeEnv,
    agent: QLearningAgent,
    episodes: int,
    max_steps: int,
) -> Tuple[float, float]:
    successes = 0
    successful_steps: List[int] = []

    for _ in range(episodes):
        state = env.reset()

        for step in range(1, max_steps + 1):
            action = agent.choose_action(state, training=False)
            state, _, done, _ = env.step(action)
            if done:
                break

        if state == env.goal:
            successes += 1
            successful_steps.append(step)

    success_rate = successes / episodes
    avg_steps = (
        sum(successful_steps) / len(successful_steps)
        if successful_steps
        else float("inf")
    )
    return success_rate, avg_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabular Q-learning maze assignment")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.08)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)

    parser.add_argument(
        "--no-gui", action="store_true", help="Run without Tkinter window"
    )
    parser.add_argument("--train-delay", type=float, default=0.02)
    parser.add_argument("--demo-delay", type=float, default=0.15)
    parser.add_argument("--render-every", type=int, default=8)
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Skip post-training replay animation",
    )
    parser.add_argument(
        "--replay-last",
        action="store_true",
        help="Replay the last saved training record and skip new training",
    )
    parser.add_argument(
        "--prompt-replay",
        action="store_true",
        help="Prompt at startup: Enter to replay last record, n to train new",
    )
    parser.add_argument(
        "--replay-record-file",
        type=str,
        default=str(DEFAULT_REPLAY_RECORD_FILE),
        help="Path to replay record JSON file",
    )
    parser.add_argument(
        "--replay-fast-stride",
        type=int,
        default=1,
        help="Render every Nth early episode during replay",
    )
    parser.add_argument(
        "--replay-tail-episodes",
        type=int,
        default=8,
        help="Number of final episodes to replay slowly",
    )
    parser.add_argument(
        "--replay-fast-delay",
        type=float,
        default=0.001,
        help="Per-episode delay for early fast replay",
    )
    parser.add_argument(
        "--replay-slow-delay",
        type=float,
        default=0.05,
        help="Per-step delay for final slow replay",
    )
    parser.add_argument(
        "--replay-target-seconds",
        type=float,
        default=18.0,
        help="Approximate replay duration target in seconds (<=0 disables auto timing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    replay_record_file = Path(args.replay_record_file)
    replay_only = args.replay_last

    if args.prompt_replay and not replay_only and replay_record_file.exists():
        user_choice = (
            input(
                "Press Enter to replay last record, or type n then Enter for new training: "
            )
            .strip()
            .lower()
        )
        replay_only = user_choice == ""

    env = make_default_env(max_steps=args.max_steps)

    renderer: Optional[TkMazeRenderer] = None
    if not args.no_gui:
        if tk is None:
            print("Tkinter is unavailable. Falling back to no-GUI mode.")
        else:
            renderer = TkMazeRenderer(env)

    if replay_only:
        loaded_stats = load_training_record(replay_record_file)
        if loaded_stats is None:
            print(
                f"No replay record found at {replay_record_file}. "
                "Falling back to new training."
            )
        else:
            if renderer and renderer.alive:
                replay_training_history(
                    env=env,
                    stats=loaded_stats,
                    renderer=renderer,
                    fast_stride=args.replay_fast_stride,
                    slow_tail_episodes=args.replay_tail_episodes,
                    fast_delay=args.replay_fast_delay,
                    slow_delay=args.replay_slow_delay,
                    target_seconds=args.replay_target_seconds,
                )

                if loaded_stats.episode_traces:
                    final_trace = loaded_stats.episode_traces[-1]
                    if final_trace.path:
                        renderer.draw_path(final_trace.path, color="#2a9d8f")
                        renderer.draw_agent(final_trace.path[-1])
                renderer.set_status("Replay complete. Close the window to exit.")
                renderer.keep_open()
            else:
                print("Replay mode requires GUI. Run without --no-gui to watch replay.")
            return

    agent = QLearningAgent(
        actions=env.action_space,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    stats = train_agent(
        env=env,
        agent=agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        renderer=renderer,
        train_delay=args.train_delay,
        render_every=args.render_every,
    )

    try:
        save_training_record(stats, replay_record_file)
        print(f"Replay record updated: {replay_record_file}")
    except Exception as exc:
        print(f"Failed to save replay record: {exc}")

    if renderer and renderer.alive and not args.no_replay:
        replay_training_history(
            env=env,
            stats=stats,
            renderer=renderer,
            fast_stride=args.replay_fast_stride,
            slow_tail_episodes=args.replay_tail_episodes,
            fast_delay=args.replay_fast_delay,
            slow_delay=args.replay_slow_delay,
            target_seconds=args.replay_target_seconds,
        )

    eval_success_rate, eval_avg_steps = evaluate_policy(
        env=env,
        agent=agent,
        episodes=40,
        max_steps=args.max_steps,
    )

    path, demo_success, demo_reward = run_greedy_episode(
        env=env,
        agent=agent,
        max_steps=args.max_steps,
        renderer=renderer,
        delay=args.demo_delay,
    )

    print("\nTraining finished.")
    print(f"Final evaluation success rate (greedy): {eval_success_rate:.1%}")
    if eval_avg_steps == float("inf"):
        print("Average steps (successful episodes): N/A")
    else:
        print(f"Average steps (successful episodes): {eval_avg_steps:.2f}")
    print(f"Greedy demo success: {demo_success}, reward: {demo_reward:.2f}")
    print(
        f"Best discovered path length: {len(stats.best_path) if stats.best_path else 'N/A'}"
    )

    if renderer and renderer.alive:
        final_path = path if demo_success else stats.best_path
        if final_path:
            renderer.draw_path(final_path, color="#2a9d8f")
            renderer.draw_agent(final_path[-1])
        renderer.set_status(
            "Training complete. Green path is the learned greedy route. "
            "Close the window to exit."
        )
        renderer.keep_open()


if __name__ == "__main__":
    main()
