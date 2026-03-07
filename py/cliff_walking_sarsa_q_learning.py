# %%

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 1. 初始化环境
# CliffWalking-v0 环境：一个 4x12 的网格，中间是悬崖，到达终点奖励为 0，掉落悬崖惩罚 -100
env = gym.make('CliffWalking-v0')

# 超参数设置
EPISODES = 500      # 训练回合数
ALPHA = 0.1         # 学习率
GAMMA = 0.95        # 折扣因子
EPSILON = 0.1       # epsilon-greedy 策略中的探索率

# 初始化 Q 表 (状态空间 48, 动作空间 4)
def init_q_table():
    return np.zeros((48, 4))

# Epsilon-Greedy 策略：选择动作
def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # 探索：随机选择
    else:
        return np.argmax(q_table[state])  # 利用：选择当前状态下Q值最大的动作

# 2. SARSA 算法实现
def run_sarsa():
    q_table = init_q_table()
    rewards_history = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        action = choose_action(state, q_table, EPSILON) # SARSA: 先选出第一个动作
        
        episode_reward = 0
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = choose_action(next_state, q_table, EPSILON) # SARSA: 选出下一个动作
            
            # SARSA 更新公式: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
            predict = q_table[state, action]
            target = reward + GAMMA * q_table[next_state, next_action]
            q_table[state, action] += ALPHA * (target - predict)
            
            state = next_state
            action = next_action
            episode_reward += reward
            
            if terminated or truncated:
                break
        rewards_history.append(episode_reward)
    return q_table, rewards_history

# 3. Q-Learning 算法实现
def run_q_learning():
    q_table = init_q_table()
    rewards_history = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        
        episode_reward = 0
        while True:
            action = choose_action(state, q_table, EPSILON) # Q-Learning: 直接根据当前Q表选
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Q-Learning 更新公式: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',:)) - Q(s,a))
            # 注意：这里直接取下一个状态的最大Q值，不考虑下一个具体动作
            predict = q_table[state, action]
            target = reward + GAMMA * np.max(q_table[next_state])
            q_table[state, action] += ALPHA * (target - predict)
            
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        rewards_history.append(episode_reward)
    return q_table, rewards_history

# 4. 运行与对比
print("正在训练 SARSA...")
sarsa_q, sarsa_rewards = run_sarsa()

print("正在训练 Q-Learning...")
ql_q, ql_rewards = run_q_learning()

# 绘图展示结果
plt.plot(sarsa_rewards, label='SARSA')
plt.plot(ql_rewards, label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('SARSA vs Q-Learning on Cliff Walking')
plt.legend()
plt.show()

print("训练完成！")