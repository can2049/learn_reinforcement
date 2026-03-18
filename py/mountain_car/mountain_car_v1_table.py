# %%
#
# ============================================================================
# Tabular Q-Learning 求解 MountainCar-v0
# ============================================================================
#
# 【问题背景】
#   一辆小车位于两座山之间的谷底，目标是驶上右侧山顶（position >= 0.5）。
#   小车引擎动力不足以直接爬上山，必须来回借助势能（左右摇摆积累动量）。
#   状态空间: (position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07])  — 连续
#   动作空间: {0: 向左推, 1: 不动, 2: 向右推}                      — 离散
#   奖励: 每个时间步 reward = -1（鼓励尽快到达终点）
#
# 【算法选择 — Q-Learning（off-policy TD control）】
#   Q-Learning 直接学习最优动作价值函数 Q*(s, a)，无需知道环境的转移概率模型。
#   核心思想：用 Bellman 最优方程的采样近似来迭代更新 Q 值表。
#
#   Bellman 最优方程（理论基础）：
#     Q*(s, a) = E[ R + γ · max_a' Q*(s', a') ]
#
#   Q-Learning 更新规则（采样逼近版本）：
#     Q(s, a) ← Q(s, a) + α · [ R + γ · max_a' Q(s', a') − Q(s, a) ]
#
#   展开等价形式：
#     Q(s, a) ← (1 − α) · Q(s, a) + α · [ R + γ · max_a' Q(s', a') ]
#
#   其中:
#     α (LEARNING_RATE)    — 学习率，控制新经验对旧估计的覆盖幅度
#     γ (DISCOUNT_FACTOR)  — 折扣因子，衡量未来奖励的当前价值
#     R                    — 执行动作后获得的即时奖励
#     s'                   — 转移后的新状态
#     max_a' Q(s', a')     — 对新状态取所有动作中最大的 Q 值（贪心）
#
# 【连续状态 → 离散化（Discretization）】
#   Q-Table 要求状态为离散索引。本代码将二维连续状态空间均匀划分为 20×20 网格，
#   每个格子对应一个离散状态 (i, j)，Q-Table 形状为 (20, 20, 3)。
#   离散化公式:
#     index = floor( (state − state_low) / bin_width )
#     bin_width = (state_high − state_low) / num_bins
#
# 【探索策略 — ε-Greedy】
#   以概率 ε 随机选动作（探索），以概率 1−ε 选 Q 值最大的动作（利用）。
#   ε 随训练推进线性衰减至 0，使智能体从「大量探索」过渡到「纯利用」。
#   衰减区间: episode ∈ [START_EPSILON_DECAYING, END_EPSILON_DECAYING]
#   每步衰减量: Δε = ε_initial / (END − START)
#
# ============================================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
ENV_NAME = "MountainCar-v0"
LEARNING_RATE = 0.2  # α: 学习率，较大值使 Q 值更快收敛但可能振荡
DISCOUNT_FACTOR = 0.95  # γ: 折扣因子，接近 1 表示重视长远奖励
EPISODES = 4000  # 总训练回合数
SHOW_PROGRESS_EVERY = 500  # 每隔多少回合打印一次统计信息

# ε-Greedy 探索参数
epsilon = 0.5  # ε 初始值：50% 的概率随机探索
START_EPSILON_DECAYING = 1  # 从第 1 回合开始衰减 ε
END_EPSILON_DECAYING = EPISODES // 2  # 到一半回合时 ε 衰减到 0
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)  # Δε
print(f"epsilon_decay_value: {epsilon_decay_value:.5f}")

# --- 状态空间离散化 ---
env = gym.make(ENV_NAME)

# 将每个维度的连续区间均分为 20 个桶（bin）
# observation_space.high = [0.6, 0.07], observation_space.low = [-1.2, -0.07]
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
print(f"DISCRETE_OS_SIZE: {DISCRETE_OS_SIZE}")

# 每个桶的宽度: bin_width = (high − low) / num_bins
discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE
print(f"discrete_os_win_size: {discrete_os_win_size}")

print(f"env.action_space.n: {env.action_space.n}")

# 初始化 Q-Table，形状 (20, 20, 3)，即 (position_bins, velocity_bins, actions)
# 用 [-2, 0] 之间的随机值初始化，因为 MountainCar 的奖励全为负值（每步 -1），
# 随机初始化可打破对称性，促进早期探索。
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)
print(f"q_table shape: {q_table.shape}")

once_debug_flag = True


def get_discrete_state(state):
    """将连续状态映射到离散网格索引。

    公式: index_i = floor( (state_i − low_i) / bin_width_i )
    返回 tuple，可直接用作 numpy 数组索引。
    """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


# 用于记录每个回合的总奖励，便于后续可视化训练曲线
ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "max": [], "min": []}


# --- 训练主循环 ---
for episode in range(EPISODES):
    episode_reward = 0
    raw_state, _ = env.reset()
    if once_debug_flag:
        print(f"raw_state: {raw_state}")
    discrete_state = get_discrete_state(raw_state)
    if once_debug_flag:
        print(f"raw_state: {raw_state}  discrete_state: {discrete_state}")

    done = False
    while not done:
        # ---- Step 1: ε-Greedy 动作选择 ----
        # 以概率 (1−ε) 选择贪心动作 argmax_a Q(s, a)（利用）
        # 以概率 ε 均匀随机选一个动作（探索）
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        if once_debug_flag:
            print(f"action: {action}")

        # ---- Step 2: 与环境交互 ----
        new_state_raw, reward, terminated, truncated, _ = env.step(action)
        if once_debug_flag:
            print(
                f"new_state_raw: {new_state_raw} | reward: {reward} | terminated: {terminated} | truncated: {truncated}"
            )
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state_raw)
        if once_debug_flag:
            print(f"new_discrete_state: {new_discrete_state}")

        episode_reward += reward

        # ---- Step 3: Q-Table 更新 ----
        if not done:
            # Q-Learning 核心更新:
            #   Q(s,a) ← (1−α)·Q(s,a) + α·[R + γ·max_a' Q(s',a')]
            # 其中 TD target = R + γ·max_a' Q(s',a')
            #      TD error  = TD target − Q(s,a)
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * max_future_q
            )
            q_table[discrete_state + (action,)] = new_q

        elif new_state_raw[0] >= env.unwrapped.goal_position:
            # 到达目标位置：将该 (s, a) 的 Q 值直接设为 0。
            # 因为终止状态没有后续奖励，Q*(s_terminal, a) = 0 是准确的。
            # 这相当于给智能体一个明确信号：到达目标是最佳结局。
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

        once_debug_flag = False

    # ---- ε 线性衰减 ----
    # 在 [START_EPSILON_DECAYING, END_EPSILON_DECAYING] 区间内，
    # 每回合 ε 减少 Δε，使策略逐步从探索过渡到纯利用。
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    # 日志输出：每 SHOW_PROGRESS_EVERY 回合统计近期平均/最大/最小奖励
    if not episode % SHOW_PROGRESS_EVERY:
        avg_reward = sum(ep_rewards[-SHOW_PROGRESS_EVERY:]) / SHOW_PROGRESS_EVERY
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(avg_reward)
        aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_PROGRESS_EVERY:]))
        aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_PROGRESS_EVERY:]))
        print(
            f"Episode: {episode:>5d} | Avg Reward: {avg_reward:>5.1f} | Epsilon: {epsilon:>4.2f}"
        )

env.close()

# --- 训练曲线可视化 ---
# 观察 avg reward 是否随训练上升（趋向 -110 左右说明已学会快速到达目标）
plt.figure(figsize=(10, 5))
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="Average Reward")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="Max Reward", alpha=0.3)
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="Min Reward", alpha=0.3)
plt.title("Stage 1: Tabular Q-Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Reward (Goal = -140 approx)")
plt.legend(loc=4)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# %%
