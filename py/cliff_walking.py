#!/usr/bin/env python3

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 环境设置：4x12 的网格
WORLD_HEIGHT = 4
WORLD_WIDTH = 12

# 动作定义
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# 起点和终点
START = [3, 0]
GOAL = [3, 11]

# 强化学习参数
EPSILON = 0.1  # 探索率
# ALPHA = 0.1      # 学习率
ALPHA_START = 0.5
ALPHA_MIN = 0.01
GAMMA = 1  # 折现因子（由于是有限步任务，设为1）


def step(state, action):
    """
    环境交互函数
    返回：(新状态, 奖励)
    """
    i, j = state
    if action == UP:
        next_state = [max(i - 1, 0), j]
    elif action == DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    elif action == LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]

    # 默认步数奖励
    reward = -1

    # 悬崖区域判定：最后一行除起点和终点外的区域
    if (next_state[0] == 3) and (1 <= next_state[1] <= 10):
        reward = -100
        next_state = START  # 掉下悬崖，回到起点

    return next_state, reward


def choose_action(state, q_table):
    """
    Epsilon-greedy 策略选择动作
    """
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values = q_table[state[0], state[1], :]
        # 如果有多个最大值，随机选一个
        return np.random.choice(
            [action for action, value in enumerate(values) if value == np.max(values)]
        )


def sarsa(q_table, alpha):
    """
    Sarsa 算法实现：On-policy
    """
    state = START
    action = choose_action(state, q_table)
    total_reward = 0
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_table)
        total_reward += reward
        # Sarsa 核心更新公式：使用下一步实际执行的动作 next_action
        target = reward + GAMMA * q_table[next_state[0], next_state[1], next_action]
        q_table[state[0], state[1], action] += alpha * (
            target - q_table[state[0], state[1], action]
        )
        state = next_state
        action = next_action
    return total_reward


def q_learning(q_table, alpha):
    """
    Q-learning 算法实现：Off-policy
    """
    state = START
    total_reward = 0
    while state != GOAL:
        action = choose_action(state, q_table)
        next_state, reward = step(state, action)
        total_reward += reward
        # Q-learning 核心更新公式：使用下一步 Q 值中最大的那个，不考虑实际动作
        best_next_q = np.max(q_table[next_state[0], next_state[1], :])
        target = reward + GAMMA * best_next_q
        q_table[state[0], state[1], action] += alpha * (
            target - q_table[state[0], state[1], action]
        )
        state = next_state
    return total_reward


# ----------------- 运行与对比 -----------------


def run_experiment(episodes=500):
    q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))

    rewards_sarsa = []
    rewards_q_learning = []

    for i in range(episodes):
        current_alpha = max(ALPHA_MIN, ALPHA_START * (0.9**i))

        rewards_sarsa.append(sarsa(q_sarsa, alpha=current_alpha))
        rewards_q_learning.append(q_learning(q_q_learning, alpha=current_alpha))

    # 转换为 Series 对象
    s_sarsa = pd.Series(rewards_sarsa)
    s_q = pd.Series(rewards_q_learning)

    # 计算滑动平均，window=20 表示每20个点取一次平均
    # min_periods=1 的作用是即使开头不足20个点也计算平均，避免出现空值
    smoothed_sarsa = s_sarsa.rolling(window=20, min_periods=1).mean()
    smoothed_q = s_q.rolling(window=20, min_periods=1).mean()

    # 绘制平滑后的奖励曲线
    plt.plot(smoothed_sarsa, label="Sarsa")
    plt.plot(smoothed_q, label="Q-Learning")
    # plt.plot(rewards_sarsa, label="Sarsa")
    # plt.plot(rewards_q_learning, label="Q-Learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.ylim([-200, 0])
    plt.title("Sarsa vs Q-Learning on Cliff Walking")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_experiment()
