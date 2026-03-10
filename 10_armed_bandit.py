# %%

import numpy as np
import matplotlib.pyplot as plt


class BanditEnvironment:
    """10臂老虎机环境"""

    def __init__(self, k=10):
        self.k = k
        self.reset()

    def reset(self):
        # 每个动作的真实期望奖励 q*(a) 服从标准正态分布
        self.q_true = np.random.randn(self.k)
        self.optimal_action = np.argmax(self.q_true)

    def get_reward(self, action):
        # 实际奖励 = 期望奖励 + 噪声
        return self.q_true[action] + np.random.randn()


def run_experiment(k=10, steps=1000, runs=2000):
    # 初始化统计数据
    rewards_eps = np.zeros(steps)
    rewards_ucb = np.zeros(steps)
    rewards_grad = np.zeros(steps)

    for r in range(runs):
        env = BanditEnvironment(k)

        # --- 1. Epsilon-Greedy (eps=0.1) ---
        q_est = np.zeros(k)
        n_count = np.zeros(k)
        epsilon = 0.1
        for t in range(steps):
            action = (
                np.argmax(q_est) if np.random.rand() > epsilon else np.random.randint(k)
            )
            reward = env.get_reward(action)
            n_count[action] += 1
            q_est[action] += (reward - q_est[action]) / n_count[action]
            rewards_eps[t] += reward

        # --- 2. UCB (c=2) ---
        q_est = np.zeros(k)
        n_count = np.zeros(k)
        for t in range(steps):
            if 0 in n_count:
                action = np.where(n_count == 0)[0][0]
            else:
                ucb_values = q_est + 2 * np.sqrt(np.log(t + 1) / n_count)
                action = np.argmax(ucb_values)
            reward = env.get_reward(action)
            n_count[action] += 1
            q_est[action] += (reward - q_est[action]) / n_count[action]
            rewards_ucb[t] += reward

        # --- 3. Gradient Bandit (alpha=0.1, with baseline) ---
        h_pref = np.zeros(k)
        avg_reward = 0
        for t in range(steps):
            # Softmax 选择动作
            exp_h = np.exp(h_pref - np.max(h_pref))  # 防止数值溢出
            probs = exp_h / np.sum(exp_h)
            action = np.random.choice(range(k), p=probs)

            reward = env.get_reward(action)
            # 更新基准线 (增量平均)
            avg_reward += (reward - avg_reward) / (t + 1)

            # 更新偏好度 H
            one_hot = np.zeros(k)
            one_hot[action] = 1
            h_pref += 0.1 * (reward - avg_reward) * (one_hot - probs)
            rewards_grad[t] += reward

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_eps / runs, label="$\epsilon$-greedy ($\epsilon=0.1$)")
    plt.plot(rewards_ucb / runs, label="UCB ($c=2$)")
    plt.plot(rewards_grad / runs, label="Gradient Bandit ($\\alpha=0.1$)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("10-Armed Bandit: Algorithm Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_experiment()
