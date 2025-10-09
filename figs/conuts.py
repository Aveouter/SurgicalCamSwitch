import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===== 方法名称 =====
methods = [
    "Shimizu et al.",
    "Hachiuma et al.",
    "Hachiuma et al. w/o Semantic",
    "Saito et al.",
    "Ours w/o Video",
    "Ours w/o Semantic",
    "Ours"
]

# ===== 示例数据：每行方法、列为 [P, R, F1] =====
# TODO: 替换为真实数值
metrics_seq = np.array([
    [0.61, 0.59, 0.60],   # Shimizu et al.
    [0.74, 0.75, 0.85],   # Hachiuma et al.
    [0.80, 0.76, 0.81],   # Hachiuma et al. w/o Semantic
    [0.59, 0.63, 0.61],   # Saito et al.    
    [0.80, 0.79, 0.81],   # Ours w/o Video
    [0.86, 0.87, 0.88],   # Ours w/o Semantic
    [0.92, 0.87, 0.91],   # Ours
])

metrics_surg = np.array([
    [0.60, 0.58, 0.59],
    [0.79, 0.80, 0.79],
    [0.78, 0.79, 0.79],
    [0.61, 0.60, 0.60],
    [0.67, 0.66, 0.67],
    [0.89, 0.89, 0.89],
    [0.89, 0.90, 0.89],
])

# ===== 风格参数 =====
plt.rcParams.update({
    "font.size": 18,       # 默认字体
    "axes.titlesize": 18   # 标题字体
})
metric_labels = ["Precision", "Recall", "F1"]
colors = ["#1f77b4", "#2ca02c", "#d62728"]  # 蓝 / 绿 / 红

n_methods = len(methods)
n_metrics = 3
bar_group_width = 0.8
bar_width = bar_group_width / (n_metrics * 2)

x = np.arange(n_methods)

fig, ax = plt.subplots(figsize=(16, 7))

# ===== 画柱子 =====
for m in range(n_metrics):  # P/R/F1
    offset = (m - (n_metrics - 1) / 2) * 2 * bar_width
    # Sequence-Out 深色
    ax.bar(x + offset - bar_width/2, metrics_seq[:, m],
           width=bar_width, color=colors[m], alpha=0.9)
    # Surgery-Out 浅色
    ax.bar(x + offset + bar_width/2, metrics_surg[:, m],
           width=bar_width, color=colors[m], alpha=0.45)

# ===== 样式 =====
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=14)  # X轴文字缩小
ax.set_ylabel("Score", weight="bold", fontsize=16)                 # Y轴缩小
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# ===== 图例 =====
legend_elements_metrics = [Patch(facecolor=colors[i], label=metric_labels[i]) for i in range(n_metrics)]
legend_elements_settings = [
    Patch(facecolor="gray", alpha=0.9, label="Sequence-Out"),
    Patch(facecolor="gray", alpha=0.45, label="Surgery-Out")
]

legend_all = legend_elements_metrics + legend_elements_settings
labels_all = metric_labels + ["Sequence-Out", "Surgery-Out"]

fig.legend(legend_all, labels_all,
           loc="lower center", ncol=5, frameon=False, fontsize=16,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out_path = "/baksv/CIGIT/GXN_Liuxy/Time-Series-Library/figs/PRF1_MVOR.pdf"
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()

print(out_path)
