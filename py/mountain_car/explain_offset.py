"""
详细解释 offset 的计算
"""
import numpy as np

# 参数设置
num_tilings = 8
tiles_per_dim = 8

print("=" * 70)
print("1. offset 的计算公式")
print("=" * 70)
print("offsets = [")
print("    [i / num_tilings, j / num_tilings]")
print("    for i, j in zip(range(num_tilings), range(num_tilings))")
print("]")
print()

print("=" * 70)
print("2. 逐层计算 offset")
print("=" * 70)

offsets = []
for i, j in zip(range(num_tilings), range(num_tilings)):
    offset = [i / num_tilings, j / num_tilings]
    offsets.append(offset)
    print(f"第 {i} 层: i={i}, j={j} → offset = [{offset[0]:.3f}, {offset[1]:.3f}]")

print()
print("=" * 70)
print("3. 完整的 offsets 列表")
print("=" * 70)
print(f"offsets = {offsets}")
print()

print("=" * 70)
print("4. offset 的含义")
print("=" * 70)
print("每个 offset 是一个 2D 向量：[pos_offset, vel_offset]")
print(f"  - pos_offset: 位置维度的偏移，范围 [0, {(num_tilings-1)/num_tilings:.3f}]")
print(f"  - vel_offset: 速度维度的偏移，范围 [0, {(num_tilings-1)/num_tilings:.3f}]")
print()
print("偏移的目的：")
print("  - 第 0 层: 无偏移，网格从 (0, 0) 开始")
print("  - 第 1 层: 偏移 (0.125, 0.125)，网格整体右移")
print("  - 第 2 层: 偏移 (0.25, 0.25)，网格整体右移更多")
print("  - ...")
print(f"  - 第 {num_tilings-1} 层: 偏移 ({(num_tilings-1)/num_tilings:.3f}, {(num_tilings-1)/num_tilings:.3f})")
print()

print("=" * 70)
print("5. offset 如何影响 tile 索引计算")
print("=" * 70)

# 模拟一个状态
pos = -0.3
vel = 0.02
pos_low, vel_low = -1.2, -0.07
pos_high, vel_high = 0.6, 0.07

print(f"示例状态: pos = {pos}, vel = {vel}")
print()

# 计算缩放因子
pos_scale = tiles_per_dim / (pos_high - pos_low)
vel_scale = tiles_per_dim / (vel_high - vel_low)

print(f"缩放因子:")
print(f"  pos_scale = {tiles_per_dim} / ({pos_high} - {pos_low}) = {pos_scale:.3f}")
print(f"  vel_scale = {tiles_per_dim} / ({vel_high} - {vel_low}) = {vel_scale:.3f}")
print()

print("不加偏移的 tile 索引:")
base_p_idx = int((pos - pos_low) * pos_scale)
base_v_idx = int((vel - vel_low) * vel_scale)
print(f"  p_idx = ({pos} - {pos_low}) × {pos_scale:.3f} = {base_p_idx}")
print(f"  v_idx = ({vel} - {vel_low}) × {vel_scale:.3f} = {base_v_idx}")
print()

print("=" * 70)
print("6. 不同层的偏移效果对比")
print("=" * 70)

for layer in range(num_tilings):
    offset = offsets[layer]

    # 计算带偏移的 tile 索引
    p_idx = int((pos - pos_low) * pos_scale + offset[0])
    v_idx = int((vel - vel_low) * vel_scale + offset[1])

    # 边界保护
    p_idx = max(0, min(p_idx, tiles_per_dim - 1))
    v_idx = max(0, min(v_idx, tiles_per_dim - 1))

    pos_off_str = f"{offset[0]:.3f}"
    vel_off_str = f"{offset[1]:.3f}"
    print(f"第 {layer} 层 (offset = [{pos_off_str}, {vel_off_str}]):")
    print(f"  p_idx = {base_p_idx} + {pos_off_str} = {base_p_idx + offset[0]:.3f} → {p_idx}")
    print(f"  v_idx = {base_v_idx} + {vel_off_str} = {base_v_idx + offset[1]:.3f} → {v_idx}")

    if p_idx != base_p_idx or v_idx != base_v_idx:
        print(f"  ⚠️  与基础索引不同！偏移产生了作用")

print()

print("=" * 70)
print("7. 可视化：8 层网格的偏移效果")
print("=" * 70)
print("假设 4×4 网格（简化示例）:")
print()

# 简化示例：4 层 4×4 网格
simple_num_tilings = 4
simple_tiles = 4

print("第 0 层 (无偏移):")
print("网格线位置: |  0  |  1  |  2  |  3  |")
print("             0.0  0.25 0.50 0.75 1.0")
print()

print("第 1 层 (偏移 0.25):")
print("网格线位置:    |  0  |  1  |  2  |  3  |")
print("               0.0  0.25 0.50 0.75 1.0")
print("              ^偏移 0.25")
print()

print("第 2 层 (偏移 0.5):")
print("网格线位置:       |  0  |  1  |  2  |  3  |")
print("                 0.0  0.25 0.50 0.75 1.0")
print("                ^偏移 0.5")
print()

print("第 3 层 (偏移 0.75):")
print("网格线位置:          |  0  |  1  |  2  |  3  |")
print("                    0.0  0.25 0.50 0.75 1.0")
print("                   ^偏移 0.75")
print()

print("=" * 70)
print("8. 偏移的效果：游标卡尺原理")
print("=" * 70)
print("单一粗粒度网格:")
print("  |-----|-----|-----|-----|")
print("  0     1     2     3     4")
print()
print("  状态 2.6 只能定位到第 2 格（分辨率低）")
print()

print("多层偏移网格:")
print("第0层: |-----|-----|-----|-----|  ← 状态 2.6 落在第 2 格")
print("       0     1     2     3     4")
print()
print("第1层:   |-----|-----|-----|-----|  ← 状态 2.6 落在第 2 格")
print("       0     1     2     3     4")
print()
print("第2层:     |-----|-----|-----|-----|  ← 状态 2.6 落在第 3 格！")
print("       0     1     2     3     4")
print()
print("结论：通过不同层的激活位置组合，可以精确定位状态")
print()

print("=" * 70)
print("9. 实际效果：分辨率提升")
print("=" * 70)
print(f"单层 8×8 网格: {tiles_per_dim * tiles_per_dim} 个区域")
print(f"{num_tilings} 层 8×8 网格: 约 {num_tilings * tiles_per_dim * tiles_per_dim} 个有效区域")
print(f"分辨率提升: 约 {num_tilings} 倍")
