"""
Mountain Car Solution
Algorithm: Sarsa(lambda) with Tile Coding
Environment: Gymnasium (formerly OpenAI Gym) MountainCar-v0
"""
# %%

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")

# ==================== Tile Coding Implementation ====================


class TileCoder:
    """
    Tile Coding: Maps continuous state space to discrete features.
    Uses multiple overlapping grids (tilings) covering the state space, each with a different offset.
    Each state activates one tile per tiling, resulting in a sparse binary feature vector.
    """

    def __init__(
        self,
        n_tilings: int = 8,
        n_tiles: int = 8,
        low: List[float] = [-1.2, -0.07],  # [position, velocity]
        high: List[float] = [0.6, 0.07],  # [position, velocity]
        offset_fraction: float = 0.5,
    ):
        """
        Initialize Tile Coder

        Args:
            n_tilings: Number of overlapping grids (more = higher resolution)
            n_tiles: Number of tiles per dimension (controls generalization width)
            low: Lower bounds of state space
            high: Upper bounds of state space
            offset_fraction: Grid offset fraction (breaks symmetry)
        """
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.low = np.array(low)
        self.high = np.array(high)

        self.tile_width = (self.high - self.low) / n_tiles

        self.offsets = np.zeros((n_tilings, len(low)))
        for i in range(n_tilings):
            for d in range(len(low)):
                self.offsets[i, d] = i * (self.tile_width[d] * offset_fraction)
                self.offsets[i, d] = self.offsets[i, d] % self.tile_width[d]

        self.n_features_per_action = n_tilings * (n_tiles ** len(low))

    def get_features(self, state: np.ndarray, action: int) -> List[int]:
        """
        Convert a (state, action) pair to a list of feature indices.

        Args:
            state: Continuous state [position, velocity]
            action: Action (0: push left, 1: no push, 2: push right)

        Returns:
            List of feature indices (one per tiling)
        """
        features = []

        for tiling in range(self.n_tilings):
            tile_indices = []

            for d in range(len(state)):
                shifted = state[d] + self.offsets[tiling, d]
                idx = int((shifted - self.low[d]) / self.tile_width[d])
                idx = max(0, min(idx, self.n_tiles - 1))
                tile_indices.append(idx)

            # Convert multi-dimensional index to a flat index
            tile_id = 0
            for d, idx in enumerate(tile_indices):
                tile_id += idx * (self.n_tiles**d)

            # Add tiling offset and action offset so each action has its own feature space
            feature_id = tile_id + tiling * (self.n_tiles ** len(state))
            feature_id += action * self.n_features_per_action
            features.append(feature_id)

        return features


# ==================== Sarsa(lambda) Agent ====================


class SarsaLambdaAgent:
    """
    Sarsa(lambda) agent with Tile Coding and linear function approximation.

    Core formulas:
        Q(s,a) = w^T * features(s,a)
        delta = R + gamma*Q(s',a') - Q(s,a)
        e = gamma*lambda*e + grad_Q(s,a)
        w = w + alpha * delta * e
    """

    def __init__(
        self,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lambda_: float = 0.9,
        epsilon: float = 0.0,  # Use optimistic initialization instead
        n_tilings: int = 8,
        n_tiles: int = 8,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon

        self.tile_coder = TileCoder(
            n_tilings=n_tilings, n_tiles=n_tiles, low=[-1.2, -0.07], high=[0.6, 0.07]
        )

        self.n_features = self.tile_coder.n_features_per_action * n_actions
        self.w = np.zeros(self.n_features)
        self.e = np.zeros(self.n_features)  # Eligibility trace vector

        self.episode_lengths = []
        self.episode_rewards = []

    def get_q(self, state: np.ndarray, action: int) -> float:
        """
        Compute Q(s,a) = w^T * features(s,a).
        Since features are binary (0 or 1), Q-value is the sum of corresponding weights.
        """
        features = self.tile_coder.get_features(state, action)
        return np.sum(self.w[features])

    def get_all_q(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        q_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q_values[a] = self.get_q(state, a)
        return q_values

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection (epsilon=0 relies on optimistic initialization)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.get_all_q(state)
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
    ):
        """
        Sarsa(lambda) update step:
            1. Compute current Q-value
            2. Compute TD target
            3. Compute TD error
            4. Update eligibility traces (replacing traces)
            5. Update weights
            6. Handle terminal state
        """
        q_current = self.get_q(state, action)

        if done:
            td_target = reward
        else:
            next_q = self.get_q(next_state, next_action)
            td_target = reward + self.gamma * next_q

        td_error = td_target - q_current

        # Decay all traces, then set current (state, action) traces to 1 (replacing traces)
        self.e *= self.gamma * self.lambda_

        features = self.tile_coder.get_features(state, action)
        for f in features:
            self.e[f] = 1

        # w <- w + alpha * delta * e
        self.w += self.alpha * td_error * self.e

        if done:
            self.e.fill(0)

    def train_episode(self, env) -> Tuple[float, int]:
        """
        Train one episode.

        Returns:
            total_reward: Total reward (typically negative; closer to 0 is better)
            steps: Number of steps
        """
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        action = self.select_action(state)

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            next_action = self.select_action(next_state)

            self.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

        return total_reward, steps


# ==================== Training and Evaluation ====================


def train_mountain_car(
    episodes: int = 1000,
    alpha: float = 0.1,
    lambda_: float = 0.9,
    n_tilings: int = 8,
    render_final: bool = False,
) -> SarsaLambdaAgent:
    """
    Train a Mountain Car solution.

    Args:
        episodes: Number of training episodes
        alpha: Learning rate
        lambda_: Trace decay parameter
        n_tilings: Number of tilings
        render_final: Whether to render the final policy

    Returns:
        Trained agent
    """
    env = gym.make("MountainCar-v0")

    # Rule of thumb: alpha = 0.1 / n_tilings balances update magnitude
    adjusted_alpha = alpha / n_tilings

    agent = SarsaLambdaAgent(
        n_actions=3,
        alpha=adjusted_alpha,
        gamma=1.0,  # No discounting for episodic task
        lambda_=lambda_,
        epsilon=0.0,  # Explore via optimistic initialization
        n_tilings=n_tilings,
        n_tiles=8,
    )

    print(f"Starting training for {episodes} episodes...")
    print(
        f"Parameters: alpha={adjusted_alpha:.4f}, lambda={lambda_}, tilings={n_tilings}"
    )

    for episode in range(episodes):
        total_reward, steps = agent.train_episode(env)
        agent.episode_lengths.append(steps)
        agent.episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_steps = np.mean(agent.episode_lengths[-100:])
            print(
                f"Episode {episode + 1:4d}, "
                f"Avg Steps: {avg_steps:.1f}, "
                f"Best: {np.min(agent.episode_lengths[-100:]):.1f}"
            )

    print("Training complete!")

    if render_final:
        render_policy(agent)

    env.close()
    return agent


def render_policy(agent: SarsaLambdaAgent, episodes: int = 3):
    """
    Render the agent's policy.

    Args:
        agent: Trained agent
        episodes: Number of episodes to render
    """
    env = gym.make("MountainCar-v0", render_mode="human")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        print(f"\nRendering Episode {ep + 1}...")

        while not done and steps < 500:
            q_values = agent.get_all_q(state)
            action = int(np.argmax(q_values))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1

        print(f"Episode {ep + 1} finished in {steps} steps")

    env.close()


def plot_learning_curve(agent: SarsaLambdaAgent, window: int = 50):
    """
    Plot the learning curve.

    Args:
        agent: Trained agent
        window: Smoothing window size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    steps = agent.episode_lengths

    cumsum = np.cumsum(np.insert(steps, 0, 0))
    smooth = (cumsum[window:] - cumsum[:-window]) / window

    ax1.plot(steps, alpha=0.3, color="blue", label="Raw")
    ax1.plot(
        range(window - 1, len(steps)),
        smooth,
        color="red",
        label=f"{window}-episode average",
    )
    ax1.axhline(y=100, color="green", linestyle="--", label="Target (<100 steps)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Steps per Episode")
    ax1.set_title("Learning Curve - Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    rewards = agent.episode_rewards

    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smooth_rewards = (cumsum[window:] - cumsum[:-window]) / window

    ax2.plot(rewards, alpha=0.3, color="blue", label="Raw")
    ax2.plot(
        range(window - 1, len(rewards)),
        smooth_rewards,
        color="red",
        label=f"{window}-episode average",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Learning Curve - Total Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== Parameter Experiments ====================


def compare_lambda_values():
    """
    Compare the effect of different lambda values on learning.

    lambda controls eligibility trace decay rate:
        lambda=0: Only the current state is updated (similar to TD(0))
        lambda=0.5: Moderate memory
        lambda=0.9: Long memory, good for delayed rewards
        lambda=1: Never forgets (similar to Monte Carlo)
    """
    lambda_values = [0.0, 0.5, 0.9, 0.99]
    colors = ["red", "green", "blue", "purple"]

    plt.figure(figsize=(12, 6))

    for lambda_, color in zip(lambda_values, colors):
        print(f"\nTesting lambda={lambda_}...")
        agent = train_mountain_car(
            episodes=500, alpha=0.1, lambda_=lambda_, n_tilings=8, render_final=False
        )

        steps = agent.episode_lengths
        window = 50
        cumsum = np.cumsum(np.insert(steps, 0, 0))
        smooth = (cumsum[window:] - cumsum[:-window]) / window

        plt.plot(
            range(window - 1, len(steps)),
            smooth,
            color=color,
            label=f"lambda={lambda_}",
        )

    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode (smoothed)")
    plt.title("Learning Curve Comparison for Different Lambda Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color="black", linestyle="--", alpha=0.5)
    plt.show()


# ==================== Main Program ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Mountain Car Solution")
    print("=" * 50)

    EPISODES = 1000
    ALPHA = 0.1
    LAMBDA = 0.9  # Best practice value for trace decay
    N_TILINGS = 8

    # 1. Train agent
    agent = train_mountain_car(
        episodes=EPISODES,
        alpha=ALPHA,
        lambda_=LAMBDA,
        n_tilings=N_TILINGS,
        render_final=False,
    )

    # 2. Plot learning curve
    plot_learning_curve(agent, window=50)

    # 3. Render final policy
    print("\nRendering final policy...")
    render_policy(agent, episodes=3)

    # 4. Optional: compare different lambda values
    print("\n" + "=" * 50)
    print("Optional: Compare different lambda values")
    print("=" * 50)
    compare = input("Run lambda comparison experiment? (y/n): ").lower()
    if compare == "y":
        compare_lambda_values()

    print("\nProgram finished!")
