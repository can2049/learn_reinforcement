# %%

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
ENV_NAME = "MountainCar-v0"
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
EPISODES = 4000
SHOW_PROGRESS_EVERY = 500

# Exploration settings
epsilon = 0.5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# --- Discretization Setup ---
env = gym.make(ENV_NAME)
# Divide the continuous space into 20 slots for each dimension
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(f"env.action_space.n: {env.action_space.n}")

# Initialize Q-Table with random values (-2 to 0)
# Structure: [Position_Index, Velocity_Index, Action]
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    """Converts continuous state to discrete grid indices."""
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Tracking metrics
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# --- Training Loop ---
for episode in range(EPISODES):
    episode_reward = 0
    raw_state, _ = env.reset()
    discrete_state = get_discrete_state(raw_state)
    done = False

    while not done:
        # 1. Action Selection (Epsilon-Greedy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # 2. Environment Step
        new_state_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state_raw)
        episode_reward += reward

        # 3. Q-Table Update
        if not done:
            # Standard Q-Learning formula: Q = (1-a)Q + a(R + g*maxQ')
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        
        elif new_state_raw[0] >= env.unwrapped.goal_position:
            # Reward for reaching the flag
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decay epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    # Logging
    if not episode % SHOW_PROGRESS_EVERY:
        avg_reward = sum(ep_rewards[-SHOW_PROGRESS_EVERY:]) / SHOW_PROGRESS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_PROGRESS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_PROGRESS_EVERY:]))
        print(f"Episode: {episode:>5d} | Avg Reward: {avg_reward:>5.1f} | Epsilon: {epsilon:>4.2f}")

env.close()

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Average Reward")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Reward", alpha=0.3)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Reward", alpha=0.3)
plt.title("Stage 1: Tabular Q-Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Reward (Goal = -140 approx)")
plt.legend(loc=4)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
