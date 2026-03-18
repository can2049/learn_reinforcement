# %%

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Hyperparameters ---
EPISODES = 1000
ALPHA = 0.1 / 8  # Step size divided by number of tilings (Standard practice)
GAMMA = 0.99
EPSILON = 0.0  # Sarsa with tile coding often works well even with 0 epsilon due to initial exploration
NUM_TILINGS = 8  # Number of overlapping layers
TILES_PER_DIM = 8  # Grid size per layer (8x8)
SHOW_PROGRESS_EVERY = 100  # 每隔多少回合打印一次统计信息


once_debug_flag = True


class TileCoder:
    def __init__(self, num_tilings, tiles_per_dim):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        # Total features = layers * (grid_w * grid_h)
        self.total_tiles = num_tilings * tiles_per_dim * tiles_per_dim

        # Define environment bounds
        self.pos_low, self.vel_low = -1.2, -0.07
        self.pos_high, self.vel_high = 0.6, 0.07

        # Calculate scale factor to map physical values to tile indices
        self.pos_scale = tiles_per_dim / (self.pos_high - self.pos_low)
        self.vel_scale = tiles_per_dim / (self.vel_high - self.vel_low)

        # Offsets for each tiling layer (Standard asymmetric offsets)
        self.offsets = [
            [i / num_tilings, j / num_tilings]
            for i, j in zip(range(num_tilings), range(num_tilings))
        ]

        if once_debug_flag:
            print(
                f"TileCoder initialized with {num_tilings} tilings and {tiles_per_dim} tiles per dimension."
            )
            print(f"Total features: {self.total_tiles}")
            print(
                f"Position scale: {self.pos_scale:.2f}, Velocity scale: {self.vel_scale:.2f}"
            )
            print(f"Offsets: {self.offsets}")

    def get_features(self, state):
        pos, vel = state
        active_tiles = []

        for i in range(self.num_tilings):
            # 计算索引后，使用 np.clip 限制范围，防止溢出
            p_idx = int((pos - self.pos_low) * self.pos_scale + self.offsets[i][0])
            v_idx = int((vel - self.vel_low) * self.vel_scale + self.offsets[i][1])

            # --- 关键修复：确保索引不越界 ---
            p_idx = max(0, min(p_idx, self.tiles_per_dim - 1))
            v_idx = max(0, min(v_idx, self.tiles_per_dim - 1))

            tile_idx = (i * self.tiles_per_dim**2) + (
                p_idx * self.tiles_per_dim + v_idx
            )
            active_tiles.append(tile_idx)

        if once_debug_flag:
            print(f"get_features.  state: {state} | active_tiles: {active_tiles}")
        return active_tiles


# --- Training Setup ---
env = gym.make("MountainCar-v0")
tc = TileCoder(NUM_TILINGS, TILES_PER_DIM)
# Weights for 3 actions, each having 'total_tiles' features
weights = np.zeros((env.action_space.n, tc.total_tiles))


def get_q(features, action):
    """Q(s,a) is the sum of weights of active tiles for that action."""
    return np.sum(weights[action][features])


def select_action(features):
    """Epsilon-greedy selection."""
    if np.random.rand() < EPSILON:
        return env.action_space.sample()
    # Calculate Q for all 3 actions and pick max
    q_values = [get_q(features, a) for a in range(env.action_space.n)]
    return np.argmax(q_values)


# --- Main Sarsa Loop ---
history = []
aggr_ep_rewards = {"ep": [], "avg": [], "max": [], "min": []}
for ep in range(EPISODES):
    state, _ = env.reset()
    features = tc.get_features(state)
    action = select_action(features)
    total_reward = 0

    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        current_q = get_q(features, action)

        if done:
            # Terminal update: Target is just the reward
            error = reward - current_q
            weights[action][features] += ALPHA * error
            break

        next_features = tc.get_features(next_state)
        next_action = select_action(next_features)

        # Semi-gradient Sarsa Update Rule:
        # w = w + alpha * [R + gamma * Q(s',a') - Q(s,a)] * grad(Q)
        # Note: grad(Q) for active tiles is 1, others 0.
        next_q = get_q(next_features, next_action)
        error = reward + GAMMA * next_q - current_q
        weights[action][features] += ALPHA * error

        features, action = next_features, next_action

        once_debug_flag = False

    history.append(total_reward)

    # 日志输出：每 SHOW_PROGRESS_EVERY 回合统计近期平均/最大/最小奖励
    if not ep % SHOW_PROGRESS_EVERY:
        avg_reward = sum(history[-SHOW_PROGRESS_EVERY:]) / SHOW_PROGRESS_EVERY
        aggr_ep_rewards["ep"].append(ep)
        aggr_ep_rewards["avg"].append(avg_reward)
        aggr_ep_rewards["max"].append(max(history[-SHOW_PROGRESS_EVERY:]))
        aggr_ep_rewards["min"].append(min(history[-SHOW_PROGRESS_EVERY:]))
        print(f"Episode: {ep:>5d} | Avg Reward: {avg_reward:>5.1f}")

# --- Plotting ---
# 观察 avg reward 是否随训练上升（趋向 -110 左右说明已学会快速到达目标）
plt.figure(figsize=(10, 5))
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="Average Reward")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="Max Reward", alpha=0.3)
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="Min Reward", alpha=0.3)
plt.title("Stage 2: Sarsa with Tile Coding")
plt.xlabel("Episodes")
plt.ylabel("Reward (Goal = -140 approx)")
plt.legend(loc=4)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
