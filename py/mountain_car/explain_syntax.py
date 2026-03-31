"""
详细解读 Python 语法
"""

print("=" * 70)
print("原代码:")
print("=" * 70)
print("""
self.offsets = [
    [i / num_tilings, j / num_tilings]
    for i, j in zip(range(num_tilings), range(num_tilings))
]
""")

print("=" * 70)
print("1. range() 函数")
print("=" * 70)
print("range(stop): 生成从 0 到 stop-1 的整数序列")
print()
num_tilings = 8
print(f"num_tilings = {num_tilings}")
print(f"range({num_tilings}) = {list(range(num_tilings))}")
print()
print("作用：")
print("  - range(8) 生成 [0, 1, 2, 3, 4, 5, 6, 7]")
print("  - 对应 8 层 tile 的索引")
print()

print("=" * 70)
print("2. zip() 函数")
print("=" * 70)
print("zip(*iterables): 将多个可迭代对象的元素打包成元组")
print()
list1 = range(num_tilings)
list2 = range(num_tilings)
print(f"zip(range(8), range(8)) 生成:")
for i, j in zip(list1, list2):
    print(f"  元组: ({i}, {j})")
print()
print("作用：")
print("  - 将两个 range(8) 配对")
print("  - 每次迭代得到 (i, j)，其中 i == j")
print("  - 这里两个 range 相同，所以 i 和 j 总是相等")
print()

print("=" * 70)
print("3. 列表推导式 (List Comprehension)")
print("=" * 70)
print("语法：[expression for item in iterable]")
print()
print("当前表达式：")
print("  [expression]  → [i/num_tilings, j/num_tilings]")
print("  for item      → for i, j in zip(...)")
print()
print("逐步展开：")
print()
print("迭代 0: i=0, j=0")
print("  表达式: [0/8, 0/8] = [0.0, 0.0]")
print("  添加到列表")
print()
print("迭代 1: i=1, j=1")
print("  表达式: [1/8, 1/8] = [0.125, 0.125]")
print("  添加到列表")
print()
print("迭代 2: i=2, j=2")
print("  表达式: [2/8, 2/8] = [0.25, 0.25]")
print("  添加到列表")
print("  ...")
print()

print("=" * 70)
print("4. 完整流程拆解")
print("=" * 70)

# 逐步展示
num_tilings = 8
print("Step 1: 创建两个 range")
range1 = range(num_tilings)
range2 = range(num_tilings)
print(f"  range1 = {list(range1)}")
print(f"  range2 = {list(range2)}")
print()

print("Step 2: 用 zip 配对")
zipped = zip(range(num_tilings), range(num_tilings))
print(f"  zip 结果 = {list(zipped)}")
print()

print("Step 3: 列表推导式遍历 zip 结果")
offsets = []
for i, j in zip(range(num_tilings), range(num_tilings)):
    value = [i / num_tilings, j / num_tilings]
    offsets.append(value)
    print(f"  i={i}, j={j} → {value}")
print()

print("Step 4: 最终结果")
print(f"  offsets = {offsets}")
print()

print("=" * 70)
print("5. 等价的循环写法")
print("=" * 70)
print("使用 for 循环实现相同功能：")
print("""
offsets = []
for i in range(num_tilings):
    for j in range(num_tilings):
        # 实际上这里 zip 使得 i == j，所以不需要嵌套循环
        offsets.append([i / num_tilings, j / num_tilings])
""")
print()
print("由于 zip(range(8), range(8)) 保证 i == j，")
print("可以简化为：")
print("""
offsets = []
for i in range(num_tilings):
    offsets.append([i / num_tilings, i / num_tilings])
""")
print()

print("=" * 70)
print("6. 语法糖对比")
print("=" * 70)
print("传统写法:")
print("""
offsets = []
for i in range(num_tilings):
    for j in range(num_tilings):
        offsets.append([i / num_tilings, j / num_tilings])
# 需要 5 行代码
""")
print()
print("列表推导式写法:")
print("""
offsets = [
    [i / num_tilings, j / num_tilings]
    for i, j in zip(range(num_tilings), range(num_tilings))
]
# 只需要 3 行代码，更简洁
""")
print()

print("=" * 70)
print("7. 更复杂的例子")
print("=" * 70)
print("例子：创建一个乘法表")
print()
multiplication_table = [
    [i * j for j in range(1, 4)]
    for i in range(1, 4)
]
print("嵌套列表推导式:")
print("  [[i * j for j in range(1, 4)] for i in range(1, 4)]")
print(f"  结果: {multiplication_table}")
print()
print("展开过程:")
for i in range(1, 4):
    row = [i * j for j in range(1, 4)]
    print(f"  i={i}: row = {row}")
print()

print("=" * 70)
print("8. 关键语法点总结")
print("=" * 70)
print("1. range(n): 生成 0 到 n-1 的整数序列")
print("2. zip(a, b): 将 a 和 b 的元素配对成元组")
print("3. [expr for item in iterable]: 列表推导式，简洁创建列表")
print("4. [expr1, expr2]: 创建包含两个元素的子列表")
print("5. /: 除法运算符，产生浮点数结果")
print()

print("=" * 70)
print("9. 实际运行验证")
print("=" * 70)
num_tilings = 8
offsets = [
    [i / num_tilings, j / num_tilings]
    for i, j in zip(range(num_tilings), range(num_tilings))
]
print(f"offsets 的形状: {len(offsets)} × {len(offsets[0])}")
print(f"offsets = {offsets}")
print()

print("验证：")
for idx, (i, j) in enumerate(zip(range(num_tilings), range(num_tilings))):
    expected = [i / num_tilings, j / num_tilings]
    actual = offsets[idx]
    match = "✓" if expected == actual else "✗"
    print(f"  {match} offsets[{idx}] = {actual} (expected: {expected})")
