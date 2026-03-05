import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# ==================== Cliff Walking 环境 ====================
class CliffWalkingEnv:
    """同前，略（保持原样）"""
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start_state = (0, 0)
        self.goal_state = (0, 11)
        self.cliff_row = 1
        self.cliff_col_start = 1  # 悬崖起始列（包含）
        self.cliff_col_end = 11  # 悬崖结束列（不包含）
        self.current_state = None

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        r, c = self.current_state
        if action == 0:   # 上
            r = max(r - 1, 0)
        elif action == 1: # 下
            r = min(r + 1, self.rows - 1)
        elif action == 2: # 左
            c = max(c - 1, 0)
        elif action == 3: # 右
            c = min(c + 1, self.cols - 1)

        next_state = (r, c)
        if r == self.cliff_row and self.cliff_col_start <= c < self.cliff_col_end:
            reward = -100
            done = False
            next_state = self.start_state
        elif next_state == self.goal_state:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        self.current_state = next_state
        return next_state, reward, done

# ==================== ε-贪婪策略 ====================
def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(Q[state[0], state[1], :])

# ==================== Sarsa ====================
def sarsa(env, alpha, gamma, epsilon, episodes=500):
    Q = np.zeros((env.rows, env.cols, 4))
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            if done:
                Q[state[0], state[1], action] += alpha * (reward - Q[state[0], state[1], action])
                total_reward += reward
                break
            next_action = epsilon_greedy(Q, next_state, epsilon)
            td_target = reward + gamma * Q[next_state[0], next_state[1], next_action]
            Q[state[0], state[1], action] += alpha * (td_target - Q[state[0], state[1], action])
            state, action = next_state, next_action
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# ==================== Q-learning ====================
def q_learning(env, alpha, gamma, epsilon, episodes=500):
    Q = np.zeros((env.rows, env.cols, 4))
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            if done:
                Q[state[0], state[1], action] += alpha * (reward - Q[state[0], state[1], action])
                total_reward += reward
                break
            td_target = reward + gamma * np.max(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (td_target - Q[state[0], state[1], action])
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# ==================== 多次运行取平均 ====================
def run_experiment(algo, env, alpha, gamma, epsilon, runs=10, episodes=500):
    all_rewards = np.zeros((runs, episodes))
    for r in range(runs):
        rewards = algo(env, alpha, gamma, epsilon, episodes)
        all_rewards[r, :] = rewards
    return np.mean(all_rewards, axis=0)

# ==================== 参数扫描与绘图（每组参数一张子图，包含Sarsa和Q-learning）====================
def parameter_sweep():
    # 参数选择（可自行调整）
    alphas = [0.1, 0.5]
    gammas = [1.0]  # 也可以加入更多 gamma，但注意子图数量
    epsilons = [0.1, 0.5]
    episodes = 500
    runs = 10

    # 生成所有参数组合
    param_combos = list(product(alphas, gammas, epsilons))
    n_combos = len(param_combos)

    # 计算子图网格的行列数
    cols = min(2, n_combos)
    rows = (n_combos + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    env = CliffWalkingEnv()

    for idx, (alpha, gamma, eps) in enumerate(param_combos):
        print(f"Running α={alpha}, γ={gamma}, ε={eps}")
        s_curve = run_experiment(sarsa, env, alpha, gamma, eps, runs, episodes)
        q_curve = run_experiment(q_learning, env, alpha, gamma, eps, runs, episodes)

        ax = axes_flat[idx]
        ax.plot(s_curve, label="Sarsa", color="blue", alpha=0.8, linewidth=1.5)
        ax.plot(q_curve, label="Q-learning", color="red", alpha=0.8, linewidth=1.5)
        ax.set_title(f'α={alpha}, γ={gamma}, ε={eps}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(True)

    # 隐藏多余的子图
    for idx in range(n_combos, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig('cliff_walking_comparison_per_param.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    parameter_sweep()
