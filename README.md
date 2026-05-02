# QLearning-Maze-Navigation

基于 Q-Learning 的强化学习迷宫导航程序。智能体在 10×10 网格迷宫中通过试错学习从起点走到终点。

## 文件结构

- `maze_env.py` — 迷宫环境（Gym 风格接口）
- `rl_brain.py` — Q-Learning 智能体（表格型 Q-table + epsilon-greedy）
- `main.py` — 训练循环 + 可视化 + 记录回放
- `report/` — LaTeX 实验报告

## 运行

```bash
python main.py                           # 训练 300 回合 + GUI 可视化
python main.py --no-gui                  # 无界面训练
python main.py --replay-last             # 回放上次训练记录
python main.py --episodes 500 --seed 0   # 自定义参数
```

## 依赖

仅 Python 标准库，无第三方包。
