"""
验证学习率缩放的必要性
"""
import numpy as np

# 模拟 Tile Coding
num_tilings = 8
num_tiles = 512
num_actions = 3

# 初始权重
weights = np.zeros((num_actions, num_tiles))

# 假设某个 (s,a) 激活的 tiles
features = [37, 110, 174, 238, 302, 366, 430, 494]  # 8 个 tile
action = 2

# 初始 Q 值
q_initial = np.sum(weights[action][features])
print(f"初始 Q 值: {q_initial}")

# 假设 TD 误差
delta = 1.0
print(f"TD 误差 (delta): {delta}")

print("\n" + "="*50)
print("情况 1: 不除以 8 (ALPHA = 0.1)")
print("="*50)

ALPHA = 0.1
weights1 = weights.copy()
weights1[action][features] += ALPHA * delta  # 8 个 tile 都增加 0.1
q_new1 = np.sum(weights1[action][features])
print(f"每个 tile 增加: {ALPHA * delta}")
print(f"Q 值实际变化: {q_new1 - q_initial}")
print(f"期望 Q 值变化: {ALPHA * delta}")
print(f"缩放倍数: {(q_new1 - q_initial) / (ALPHA * delta):.1f} 倍")

print("\n" + "="*50)
print("情况 2: 除以 8 (ALPHA = 0.1/8)")
print("="*50)

ALPHA = 0.1 / 8
weights2 = weights.copy()
weights2[action][features] += ALPHA * delta  # 8 个 tile 都增加 0.0125
q_new2 = np.sum(weights2[action][features])
print(f"每个 tile 增加: {ALPHA * delta}")
print(f"Q 值实际变化: {q_new2 - q_initial}")
print(f"期望 Q 值变化: {0.1 * delta}")  # 用原始 ALPHA 0.1 作为期望
print(f"缩放倍数: {(q_new2 - q_initial) / (0.1 * delta):.1f} 倍")

print("\n" + "="*50)
print("结论")
print("="*50)
print(f"不除以 8: Q 值变化被放大 {num_tilings} 倍")
print(f"除以 {num_tilings}: Q 值变化符合预期")
