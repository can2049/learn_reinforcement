# %%

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Hyperparameters ---
EPISODES = 200     # With Eligibility Traces, we need MUCH fewer episodes!
ALPHA = 0.1 / 8    # Step size
GAMMA = 0.99
LAMBDA = 0.9       # Trace decay rate (The "Magic" parameter)
NUM_TILINGS = 8
TILES_PER_DIM = 8

class TileCoder:
    def __init__(self, num_tilings, tiles_per_dim):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.total_tiles = num_tilings * tiles_per_dim * tiles_per_dim
        self.pos_low, self.vel_low = -1.2, -0.07
        self.pos_high, self.vel_high = 0.6, 0.07
        self.pos_scale = tiles_per_dim / (self.pos_high - self.pos_low)
        self.vel_scale = tiles_per_dim / (self.vel_high - self.vel_low)
        self.offsets = [[i / num_tilings, j / num_tilings] for i, j in zip(range(num_tilings), range(num_tilings))]

    def get_features(self, state):
        pos, vel = state
        active_tiles = []
        for i in range(self.num_tilings):
            p_idx = int((pos - self.pos_low) * self.pos_scale + self.offsets[i][0])
            v_idx = int((vel - self.vel_low) * self.vel_scale + self.offsets[i][1])
            # Clip to stay in bounds
            p_idx = max(0, min(p_idx, self.tiles_per_dim - 1))
            v_idx = max(0, min(v_idx, self.tiles_per_dim - 1))
            tile_idx = (i * self.tiles_per_dim**2) + (p_idx * self.tiles_per_dim + v_idx)
            active_tiles.append(tile_idx)
        return active_tiles

# --- Initialize Environment and Agent ---
env = gym.make("MountainCar-v0")
tc = TileCoder(NUM_TILINGS, TILES_PER_DIM)
weights = np.zeros((env.action_space.n, tc.total_tiles))

def get_q(features, action):
    return np.sum(weights[action][features])

def select_action(features):
    q_values = [get_q(features, a) for a in range(env.action_space.n)]
    return np.argmax(q_values) # Pure Greedy usually works here

history = []

# --- Training Loop ---
for ep in range(EPISODES):
    # Initialize Traces z for each episode
    z = np.zeros_like(weights) 
    
    state, _ = env.reset()
    features = tc.get_features(state)
    action = select_action(features)
    total_reward = 0
    
    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # 1. Calculate TD Error
        current_q = get_q(features, action)
        
        if done:
            target = reward
            delta = target - current_q
        else:
            next_features = tc.get_features(next_state)
            next_action = select_action(next_features)
            next_q = get_q(next_features, next_action)
            delta = reward + GAMMA * next_q - current_q
        
        # 2. Update Traces (Replacing Traces logic)
        # First, decay all traces
        z *= GAMMA * LAMBDA
        # Then, set active features' traces to 1 (Replacing)
        z[action][features] = 1.0 
        
        # 3. Global Update: Weights move according to the broadcasted delta
        weights += ALPHA * delta * z
        
        if done: break
            
        features, action = next_features, next_action

    history.append(total_reward)
    if ep % 20 == 0: print(f"Episode {ep}, Reward: {total_reward}")

# --- Final Reward Visualization ---
plt.plot(history)
plt.axhline(y=-110, color='r', linestyle='--', label='Expert Performance')
plt.title(f"Stage 3: Sarsa(λ={LAMBDA}) with Tile Coding")
plt.xlabel("Episode")
plt.ylabel("Steps (Negative Reward)")
plt.legend()
plt.show()
