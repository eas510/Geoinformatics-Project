import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

# ==========================================
# 0. 配置路径
# ==========================================
input_dir = 'result'
output_dir = 'result' # 图像也保存在 result 文件夹
os.makedirs(output_dir, exist_ok=True)

npy_file = os.path.join(input_dir, 'raw_upper_tri_sim_values.npy')
json_file = os.path.join(input_dir, 'summary_statistics.json')
output_image = os.path.join(output_dir, 'cosine_similarity_histogram.png')

# 检查文件是否存在
if not os.path.exists(npy_file):
    raise FileNotFoundError(f"找不到 .npy 文件: {npy_file}，请先运行之前的计算脚本。")

# ==========================================
# 1. 加载数据 (使用 mmap_mode 节省内存)
# ==========================================
print(f"正在读取二进制大文件 (1.6GB+): {npy_file}...")
start_total = time.time()

# 使用 mmap_mode='r' (内存映射) 是关键。
# 它不会一次性把 1.6GB 全部载入物理内存，而是用到哪读到哪，
# 即使你的电脑只有 8GB 内存也能轻松处理。
sim_values = np.load(npy_file, mmap_mode='r')
total_pairs = sim_values.shape[0]
print(f"成功载入 {total_pairs:,} 个相似度数值。")

# ==========================================
# 2. 高效计算直方图 (聚合数据)
# ==========================================
print("正在高效计算直方图频数 (聚合 4 亿个数据点)...")
start_hist = time.time()

# 设定直方图区间：从 -1 到 1，分 100 个桶
# NumPy 底层 C 语言执行此操作非常快
counts, bin_edges = np.histogram(sim_values, bins=100, range=(-1.0, 1.0))

# 计算桶的中心点，用于绘图 X 轴
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

print(f"聚合完成！耗时: {time.time() - start_hist:.2f} 秒。")

# ==========================================
# 3. 尝试加载之前算好的统计量 (用于图表注释)
# ==========================================
stats_text = ""
if os.path.exists(json_file):
    try:
        with open(json_file, 'r') as f:
            stats_data = json.load(f)
            stats_text = (f"Total Pairs: {stats_data['total_pairs']:,}\n"
                          f"Mean: {stats_data['mean_similarity']:.4f}\n"
                          f"Median: {stats_data['median_similarity']:.4f}\n"
                          f"Std Dev: {stats_data['std_dev']:.4f}")
    except Exception:
        pass # 加载失败也不影响绘图

# ==========================================
# 4. 绘图 (仅绘制 100 根柱子，瞬间完成)
# ==========================================
print("正在生成可视化图像...")
plt.figure(figsize=(12, 7), dpi=150) # 设置高 DPI 确保清晰

# 使用 fill_between 绘制面积图，比常规 bar 图在桶很多时视觉效果更好
plt.fill_between(bin_centers, counts, step="mid", color='#2c7bb6', alpha=0.7, edgecolor='#1a5a8c')

# 或者是标准的柱状图 (两者选其一，面积图看起来更平滑)
# plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), color='#2c7bb6', alpha=0.7, edgecolor='white', linewidth=0.3)

# 基础设置
plt.title("Consine Similarity Distribution (Upper Triangle Matrix)\nWashington DC 100m Embeddings (2022)", fontsize=15, fontweight='bold')
plt.xlabel("Cosine Similarity", fontsize=12)
plt.ylabel("Frequency (Count)", fontsize=12)

# X 轴范围强制为 [-1, 1]
plt.xlim(-1.0, 1.0)

# 格式化 Y 轴，使用科学计数法或千分位 (如果频数太高)
plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加统计信息文本框
if stats_text:
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

# ==========================================
# 5. 保存图像
# ==========================================
plt.tight_layout()
print(f"正在保存图像到: {output_image}...")
plt.savefig(output_image)

# 关闭绘图以释放内存
plt.close()

print(f"可视化完成！总耗时: {time.time() - start_total:.2f} 秒。")
print(f"请检查文件: ./{output_image}")