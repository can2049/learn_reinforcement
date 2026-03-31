# %%

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# --- 超参数 ---
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 20000
BATCH_SIZE = 64
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # 每 10 个 Episode 更新一次目标网络

# --- 1. 定义 Q 网络结构 ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 简单的三层全连接网络，ReLU 激活
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- 2. 经验回放池 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # 随机采样，打破状态之间的时序相关性
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

# --- 3. DQN 训练逻辑 ---
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化两个网络：行为网络和目标网络
policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict()) # 初始时参数一致
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_MAX
history = []

for ep in range(300):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-Greedy 探索
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_t).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # --- 奖励塑造 (Reward Shaping) ---
        # 原始奖励每步都是 -1。
        # 这里给一个小 trick：如果位置更高，给一点额外奖励，诱导 Agent 往山上爬。
        shaped_reward = reward + 10 * abs(next_state[1]) # 奖励速度感
        if next_state[0] >= 0.5: shaped_reward += 50    # 成功大奖
        
        memory.push(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += reward # 绘图依然记录原始奖励
        
        # 开始训练
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions).unsqueeze(1)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)
            
            # 计算当前 Q 值
            current_q = policy_net(states_t).gather(1, actions_t).squeeze()
            
            # 计算目标 Q 值 (使用 Target Network)
            with torch.no_grad():
                max_next_q = target_net(next_states_t).max(1)[0]
                target_q = rewards_t + (1 - dones_t) * GAMMA * max_next_q
            
            # 计算均方误差损失
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 更新 Epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # 定期同步目标网络
    if ep % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    history.append(total_reward)
    if ep % 20 == 0:
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# 绘制结果
plt.plot(history)
plt.title("DQN Learning Curve (Mountain Car)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
