# %%

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 1. 环境初始化
env = gym.make("MountainCar-v0")

# 2. 超参数设置
EPISODES = 5000  # 训练轮次
LEARNING_RATE = 0.1  # 学习率
DISCOUNT = 0.99  # 折现因子
EPSILON = 0.1  # 探索率

# 3. 状态空间离散化
# 将连续的位置和速度划分为 20x20 的网格
DISCRETE_OS_SIZE = [20, 20]
win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# 初始化 Q 表：20x20x3 (位置, 速度, 3个动作: 左, 不动, 右)
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)


def get_discrete_state(state):
    """将连续状态转换为离散索引"""
    discrete_state = (state - env.observation_space.low) / win_size
    return tuple(discrete_state.astype(int))


# 用于记录奖励以供绘图
all_rewards = []
aggr_rewards = {"ep": [], "avg": [], "min": [], "max": []}

# 4. 训练循环
for episode in range(EPISODES):
    episode_reward = 0
    # 获取初始离散状态
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    while not done:
        # epsilon-greedy 策略
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # 执行动作
        new_state_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state_raw)

        episode_reward += reward

        if not done:
            # Q-learning 更新公式
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            # 方括号内的 TD Error 部分
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            q_table[discrete_state + (action,)] = new_q

        elif new_state_raw[0] >= env.unwrapped.goal_position:
            # 如果到达终点，给予奖励奖励（原本环境每步都是-1）
            q_table[discrete_state + (action,)] = 0
            print(f"Episode {episode} reached the goal!")

        discrete_state = new_discrete_state

    all_rewards.append(episode_reward)

    # 每 200 轮统计一次数据
    if episode % 200 == 0:
        average_reward = sum(all_rewards[-200:]) / 200
        aggr_rewards["ep"].append(episode)
        aggr_rewards["avg"].append(average_reward)
        aggr_rewards["min"].append(min(all_rewards[-200:]))
        aggr_rewards["max"].append(max(all_rewards[-200:]))

env.close()

# 5. 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(aggr_rewards["ep"], aggr_rewards["avg"], label="Average Reward")
plt.plot(aggr_rewards["ep"], aggr_rewards["max"], label="Max Reward", alpha=0.3)
plt.plot(aggr_rewards["ep"], aggr_rewards["min"], label="Min Reward", alpha=0.3)
plt.title("Mountain Car Q-Learning Training")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(loc=4)
plt.grid(True)
plt.show()
