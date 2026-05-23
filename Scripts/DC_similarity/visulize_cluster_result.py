import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time

# ==========================================
# Render the cluster result
# # ==========================================

input_csv = 'result_1000m/DC_1000m_Clustered_k14_sphere.csv' 
output_img = 'result_1000m/DC_K14_Map_Python_sphere.png'

print(f"正在读取数据: {input_csv} ...")
start_time = time.time()
df = pd.read_csv(input_csv)

# ==========================================
# 2. 配置调色板 (高对比度 8 色)
# ==========================================
# 颜色依次为: 深蓝, 浅蓝, 深绿, 浅绿, 粉红, 大红, 亮橙, 金黄
# hex_colors = ['#1f78b4', '#a6cee3', '#33a02c', '#b2df8a', '#fb9a99', '#e31a1c', '#ff7f00', '#fdbf6f']
# cmap = ListedColormap(hex_colors)

# ==========================================
# 2. 配置调色板 (高对比度 14 色 - 基于 Tab20)
# ==========================================
# 颜色依次为: 
# 深蓝, 浅蓝, 亮橙, 浅橙, 深绿, 浅绿, 大红, 浅红, 深紫, 浅紫, 棕色, 亮粉, 橄榄绿, 青色
hex_colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', 
    '#2ca02c', '#98df8a', '#d62728', '#ff9896', 
    '#9467bd', '#c5b0d5', '#8c564b', '#e377c2', 
    '#bcbd22', '#17becf'
]
cmap = ListedColormap(hex_colors)

# ==========================================
# 3. 开始渲染地图
# ==========================================
print("正在渲染高分辨率地图...")
# 设置画布大小和高 DPI 保证清晰度，背景设为深灰色看起来更高级
fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
fig.patch.set_facecolor('#f0f0f0') 

# 直接将 11.5 万个网格画成散点 (由于密集排列，视觉上就是面)
# s=3 是点的大小，可以根据出图效果微调 (1~5之间)
# marker='s' 代表正方形 (square)
scatter = ax.scatter(
    df['x0'], 
    df['y0'], 
    c=df['cluster'], 
    cmap=cmap, 
    s=4, 
    marker='s', 
    edgecolors='none',
    vmin=0, 
    vmax=7
)

# 强制 X 轴和 Y 轴的比例为 1:1，防止地图被拉伸变形
ax.set_aspect('equal')

# 隐藏坐标轴，让它看起来像一张纯净的地图
ax.axis('off')

# 添加标题
plt.title("Washington DC 50m K-Means Clusters (K=14)\nDirect Python Rendering", 
          fontsize=16, fontweight='bold', pad=20)

# ==========================================
# 4. 保存图像
# ==========================================
plt.tight_layout()
plt.savefig(output_img, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"渲染完成！耗时: {time.time() - start_time:.2f} 秒。")