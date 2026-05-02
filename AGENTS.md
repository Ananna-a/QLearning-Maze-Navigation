# AGENTS.md — Python 强化学习迷宫程序设计

## 项目概要

Q-learning 迷宫作业，智能体在 10×10 网格迷宫中通过试错学习从起点 (0,0) 走到终点 (9,9)。

## 架构

| 文件 | 职责 |
|------|------|
| `maze_env.py` | Gym 风格迷宫环境：状态 `(x,y)`、动作 `0/1/2/3`（上下左右）、奖励（终点 +1，撞墙/陷阱 -1，每步 -0.01） |
| `rl_brain.py` | 表格型 Q-learning 智能体，epsilon-greedy 策略 |
| `main.py` | 训练循环 + Tkinter 可视化 + 训练记录回放 + 策略评估 |

## 运行

```bash
python main.py                          # 默认训练 1000 回合 + GUI
python main.py --no-gui                 # 无界面训练（适合服务器）
python main.py --replay-last            # 回放上次训练记录（不重新训练）
python main.py --prompt-replay          # 启动时询问：回放还是重新训练
python main.py --episodes 500 --seed 0  # 自定义回合数和随机种子
```

## 关键参数（CLI）

`--episodes` `--max-steps` `--seed` `--lr` `--gamma` `--epsilon-decay`——均在 `parse_args()` 中定义，见 `main.py:592`。

## 环境注意事项

- **撞墙不终止回合**——智能体原地不动、得 `wall_penalty`（-1），但 `done=False`，继续在同一状态选下一个动作。这是故意的，有助于在较大地图 + 动态障碍下学习。
- **每回合动态障碍**：`Dynamic Wall` 每 `reset()` 随机刷新（2-3 个），从预定义候选集中采样，保证存在可行路径。
- 验证布局时会 BFS 检查 `static_walls | traps` 下是否存在可达路径，布局非法会抛 `ValueError`。

## 无 GUI 环境

- Tkinter 不可用时自动 fallback 到无 GUI 模式（`--no-gui` 等效行为）。
- 训练记录保存到 `last_training_record.json`，可在有 GUI 的机器上用 `--replay-last` 回放。

## 依赖

- 仅 Python 标准库，无第三方包。
- Python 3.13（实际版本见 `.venv/pyvenv.cfg`，向下兼容 3.9+）。
- 虚拟环境 `.venv/`。

## 测试 / 校验

- 项目无测试框架和 CI 配置。
- 运行 `python main.py --no-gui` 并观察日志（`avg_reward` 和 `success_rate` 是否上升）是最基本的验证方式。
- 教学指南见 `Q_learning_maze_guide.md`。

## 报告

- LaTeX 报告源码在 `report/report.tex`。
- `report/report.tex` 为当前版本，`report/参考/` 为参考版本。
