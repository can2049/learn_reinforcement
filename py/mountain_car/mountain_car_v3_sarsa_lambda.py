# %%
#
# ============================================================================
# Sarsa(λ) 求解 MountainCar-v0
# ============================================================================
#
# 【问题背景】
#   一辆小车位于两座山之间的谷底，目标是驶上右侧山顶（position >= 0.5）。
#   小车引擎动力不足以直接爬上山，必须来回借助势能（左右摇摆积累动量）。
#   状态空间: (position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07])  — 连续
#   动作空间: {0: 向左推, 1: 不动, 2: 向右推}                      — 离散
#   奖励: 每个时间步 reward = -1（鼓励尽快到达终点）
#
# 【算法选择 — Sarsa(λ)】
#   Sarsa(λ) 是 Sarsa 的改进版本，引入了资格迹（Eligibility Traces）。
#
#   核心优势：
#     1. 信用分配（Credit Assignment）：能够将奖励适当地分配给过去的状态-动作对
#     2. 多步学习：结合了 n-step 的思想，可以向前/向后看多步
#     3. 收敛速度：比传统 Sarsa 快很多（通常只需 200 个回合 vs 1000 个）
#
#   Sarsa(λ) 更新规则：
#     Q(s,a) ← Q(s,a) + α · δ · e(s,a)
#     其中：
#       δ = R + γ · Q(s',a') - Q(s,a)           — TD 误差
#       e(s,a) — 资格迹（Eligibility Trace）
#
#   资格迹更新（Replacing Traces）：
#     e(s,a) ← γ · λ · e(s,a) + 1              如果 (s,a) 被访问
#     e(s,a) ← γ · λ · e(s,a)                  如果 (s,a) 未被访问
#
#   其中：
#     α (ALPHA)    — 学习率
#     γ (GAMMA)    — 折扣因子
#     λ (LAMBDA)   — 资格迹衰减率，控制向前看多少步
#                    λ=0: 等价于 Sarsa(1)（单步 TD）
#                    λ=1: 等价于 Monte Carlo（全回溯）
#
# 【为什么 λ=0.9？】
#   λ=0.9 意味着奖励的影响会向前传递约 10 步（因为 0.9^10 ≈ 0.35）。
#   这对 MountainCar 问题很合适，因为最佳策略需要约 140 步，
#   但关键决策通常发生在较短的序列中。
#
# 【Tile Coding】
#   与 v2 相同，使用 8 层 8×8 的重叠网格来离散化连续状态空间。
#   每个状态激活 8 个 tile，Q 值通过求和这些 tile 的权重得到。
#   这提供了良好的泛化能力，同时保持稀疏性。
#
# ============================================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Hyperparameters ---
EPISODES = 200     # With Eligibility Traces, we need MUCH fewer episodes!
                   # 传统 Sarsa 需要 1000 回合，Sarsa(λ) 只需 200 回合
ALPHA = 0.1 / 8    # Step size, 除以 8 是因为有 8 个 tilings（标准实践）
GAMMA = 0.99       # 折扣因子，接近 1 表示重视长远奖励
LAMBDA = 0.9       # Trace decay rate (The "Magic" parameter)
                   # λ=0.9: 奖励影响向前传递约 10 步
NUM_TILINGS = 8    # Tile Coding 的层数
TILES_PER_DIM = 8  # 每层的网格大小 (8x8)


class TileCoder:
    """Tile Coding: 用多层偏移的粗粒度网格实现细粒度但稀疏的状态表示。"""
    def __init__(self, num_tilings, tiles_per_dim):
        self.num_tilings = num_tilings          # 网格层数
        self.tiles_per_dim = tiles_per_dim        # 每层的网格维度
        # 总特征数 = 层数 × (网格宽 × 网格高)
        self.total_tiles = num_tilings * tiles_per_dim * tiles_per_dim

        # 定义环境边界
        self.pos_low, self.vel_low = -1.2, -0.07
        self.pos_high, self.vel_high = 0.6, 0.07

        # 计算缩放因子：将物理值映射到 tile 索引
        self.pos_scale = tiles_per_dim / (self.pos_high - self.pos_low)
        self.vel_scale = tiles_per_dim / (self.vel_high - self.vel_low)

        # 每层的偏移量（标准非对称偏移）
        # 通过偏移实现多层网格错开，提高整体分辨率
        self.offsets = [
            [i / num_tilings, j / num_tilings]
            for i, j in zip(range(num_tilings), range(num_tilings))
        ]

    def get_features(self, state):
        """将连续状态映射为激活的 tile 索引列表。"""
        pos, vel = state  # 解包：位置和速度
        active_tiles = []

        for i in range(self.num_tilings):
            # 计算当前层的 tile 索引（加入偏移量）
            p_idx = int((pos - self.pos_low) * self.pos_scale + self.offsets[i][0])
            v_idx = int((vel - self.vel_low) * self.vel_scale + self.offsets[i][1])

            # 边界保护：确保索引不越界
            p_idx = max(0, min(p_idx, self.tiles_per_dim - 1))
            v_idx = max(0, min(v_idx, self.tiles_per_dim - 1))

            # 计算全局 tile 编号
            tile_idx = (i * self.tiles_per_dim**2) + (
                p_idx * self.tiles_per_dim + v_idx
            )
            active_tiles.append(tile_idx)

        return active_tiles  # 返回 8 个激活的 tile 索引


# --- Initialize Environment and Agent ---
env = gym.make("MountainCar-v0")
tc = TileCoder(NUM_TILINGS, TILES_PER_DIM)
# 权重矩阵：3 个动作，每个动作有 512 个 tile 的权重
weights = np.zeros((env.action_space.n, tc.total_tiles))


def get_q(features, action):
    """计算状态-动作对的 Q 值：对激活 tile 的权重求和。"""
    return np.sum(weights[action][features])


def select_action(features):
    """纯贪心策略：选择 Q 值最大的动作（因为资格迹提供了足够的探索）。"""
    q_values = [get_q(features, a) for a in range(env.action_space.n)]
    return np.argmax(q_values)


history = []


# --- Training Loop ---
for ep in range(EPISODES):
    # 初始化资格迹 z（每个 episode 开始时重置）
    # z 形状与 weights 相同：3 × 512
    z = np.zeros_like(weights)

    state, _ = env.reset()
    features = tc.get_features(state)  # 获取初始状态的激活 tiles
    action = select_action(features)
    total_reward = 0

    while True:
        # 执行动作，与环境交互
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Step 1: 计算 TD 误差 ---
        current_q = get_q(features, action)

        if done:
            # 终止状态：目标就是即时奖励
            target = reward
            delta = target - current_q
        else:
            # 非终止状态：目标 = 即时奖励 + 折扣后的下一个 Q 值
            next_features = tc.get_features(next_state)
            next_action = select_action(next_features)
            next_q = get_q(next_features, next_action)
            # TD 误差：R + γ·Q(s',a') - Q(s,a)
            delta = reward + GAMMA * next_q - current_q

        # --- Step 2: 更新资格迹（Replacing Traces）---
        # 2.1 首先衰减所有资格迹
        z *= GAMMA * LAMBDA
        # 2.2 将当前激活特征的资格迹设置为 1（替换而非累加）
        # 这意味着最近访问的状态-动作对有最高的资格
        z[action][features] = 1.0

        # --- Step 3: 全局权重更新 ---
        # 根据 TD 误差和资格迹更新所有权重
        # 公式：w = w + α·δ·e
        # 注意：这是向量化操作，所有权重的更新一步完成
        weights += ALPHA * delta * z

        if done:
            break

        # 移动到下一个状态-动作对
        features, action = next_features, next_action

    history.append(total_reward)
    if ep % 20 == 0:
        print(f"Episode {ep}, Reward: {total_reward}")


# --- Final Reward Visualization ---
plt.plot(history)
# 添加专家性能参考线（约 -110 表示已学会快速到达目标）
plt.axhline(y=-110, color='r', linestyle='--', label='Expert Performance')
plt.title(f"Stage 3: Sarsa(λ={LAMBDA}) with Tile Coding")
plt.xlabel("Episode")
plt.ylabel("Steps (Negative Reward)")
plt.legend()
plt.show()

# %%
