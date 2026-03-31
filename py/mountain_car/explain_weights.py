"""
可视化 weights[action][features] 的含义
"""
import numpy as np

# 模拟 weights 结构
num_actions = 3
num_tiles = 512
weights = np.random.randn(num_actions, num_tiles)  # 随机初始化

print("=" * 60)
print("weights 的结构")
print("=" * 60)
print(f"形状: {weights.shape} = (动作数, tile数)")
print(f"  - 第1维 (动作): {num_actions} 个 [0=左推, 1=不动, 2=右推]")
print(f"  - 第2维 (tile): {num_tiles} 个 [8层 × 8×8网格]")
print()

# 模拟 features
features = [37, 110, 174, 238, 302, 366, 430, 494]  # 8 个激活的 tile 索引

print("=" * 60)
print("features 的含义")
print("=" * 60)
print(f"features = {features}")
print(f"  - 长度: {len(features)} (等于 tile 层数)")
print(f"  - 每个值: 激活的 tile 全局索引")
print(f"  - 范围: [0, {num_tiles-1}]")
print()

print("=" * 60)
print("不同 action 下的 weights[action][features]")
print("=" * 60)

for action in [0, 1, 2]:
    action_name = ['左推', '不动', '右推'][action]

    # weights[action] - 取出该动作的所有 tile 权重 (512 维向量)
    action_weights = weights[action]
    print(f"\n动作 {action} ({action_name}):")
    print(f"  weights[{action}] 的形状: {action_weights.shape}")

    # weights[action][features] - 取出激活的 8 个 tile 的权重
    active_weights = weights[action][features]
    print(f"  weights[{action}][features] 的形状: {active_weights.shape}")
    print(f"  激活的 tile 权重值:")
    for i, (tile_idx, w) in enumerate(zip(features, active_weights)):
        print(f"    tile[{tile_idx}] = {w:.3f}")

    # Q 值 = 激活 tile 权重的和
    q_value = np.sum(active_weights)
    print(f"  Q(s, {action_name}) = Σ weights = {q_value:.3f}")

print()
print("=" * 60)
print("动作选择：比较 3 个动作的 Q 值")
print("=" * 60)

q_values = [np.sum(weights[a][features]) for a in range(3)]
for a, q in enumerate(q_values):
    print(f"  Q(s, {['左推', '不动', '右推'][a]}) = {q:.3f}")

best_action = np.argmax(q_values)
print(f"\n最佳动作: {best_action} ({['左推', '不动', '右推'][best_action]})")
print(f"  最大 Q 值: {q_values[best_action]:.3f}")
