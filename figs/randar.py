import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ================== 数据 ==================
methods = [
    "Shimizu et al.",
    "Hachiuma et al.",
    "Hachiuma et al. w/o Semantic",
    "Saito et al.",
    "Ours w/o Video",
    "Ours w/o Semantic",
    "Ours"
]

# —— 示例 Sequence-Out —— 
values_seq = np.array([
    [0.608, 0.715, 0.758, 0.689, 0.716, 0.701],
    [0.797, 0.821, 0.835, 0.823, 0.826, 0.820],
    [0.807, 0.756, 0.844, 0.826, 0.822, 0.811],
    [0.586, 0.627, 0.620, 0.702, 0.590, 0.625],
    [0.802, 0.786, 0.820, 0.807, 0.832, 0.809],
    [0.863, 0.871, 0.880, 0.891, 0.873, 0.875],
    [0.919, 0.869, 0.923, 0.920, 0.925, 0.911]
])

# —— 示例 Surgery-Out —— 
values_surg = np.array([
    [0.602, 0.572, 0.589, 0.670, 0.656, 0.618],
    [0.798, 0.794, 0.808, 0.783, 0.802, 0.797],
    [0.772, 0.773, 0.785, 0.808, 0.802, 0.788],
    [0.581, 0.604, 0.611, 0.609, 0.591, 0.600],
    [0.659, 0.658, 0.648, 0.694, 0.691, 0.670],
    [0.889, 0.890, 0.924, 0.881, 0.893, 0.895],
    [0.867, 0.891, 0.880, 0.909, 0.893, 0.888],
])

labels = ["S1", "S2", "S3", "S4", "S5"]
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

vals_seq = values_seq[:, :5]
vals_surg = values_surg[:, :5]

# 范围与样式
rmin, rmax = 0.55, 1.0
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 18,
})

# =============== 新布局：左右两个雷达图 + 下方图例栏 ===============
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(
    nrows=2, ncols=2, height_ratios=[1.0, 0.18], width_ratios=[1.0, 1.0],
    figure=fig
)

ax_seq = plt.subplot(gs[0, 0], polar=True)
ax_surg = plt.subplot(gs[0, 1], polar=True)
ax_legend = plt.subplot(gs[1, :])  # 下方横跨两列放图例
ax_legend.axis("off")

def plot_radar(ax, vals5, title):
    ax.set_facecolor("white")
    ax.set_ylim(rmin, rmax)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=18)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 灰色虚线网格
    levels = np.linspace(rmin, rmax, 5)
    for lvl in levels:
        rs = [lvl]*N + [lvl]
        ax.plot(angles, rs, linewidth=1.4, alpha=0.4, linestyle="--", color="gray", zorder=1)
    for th in angles[:-1]:
        ax.plot([th, th], [rmin, rmax], linewidth=1.2, alpha=0.25, linestyle="--", color="gray", zorder=1)

    # 方法曲线
    lines = []
    for i, method in enumerate(methods):
        stats = vals5[i].tolist() + [vals5[i, 0]]
        lw = 5.0 if method == "Ours" else (3.8 if method == "Ours w/o Semantic" else 2.8)
        alpha_fill = 0.30 if method == "Ours" else (0.18 if method == "Ours w/o Semantic" else 0.10)
        ms = 9 if method == "Ours" else 7
        line, = ax.plot(angles, stats, linewidth=lw, marker="o", markersize=ms, zorder=3, label=method)
        ax.fill(angles, stats, alpha=alpha_fill, zorder=2, color=line.get_color())
        lines.append(line)

    ax.set_title(title, fontsize=18, pad=12, weight="bold")
    return lines

# 分别绘制
lines_seq = plot_radar(ax_seq, vals_seq, "Sequence-Out")
lines_surg = plot_radar(ax_surg, vals_surg, "Surgery-Out")

# 统一图例（横排，居中放下方）
legend = ax_legend.legend(
    lines_surg, methods,
    loc="center", frameon=False, ncol=3,
    fontsize=18, handlelength=2.8, columnspacing=1.5
)

for legline in legend.legend_handles:
    try:
        legline.set_linewidth(3.5)
        legline.set_markersize(8)
    except Exception:
        pass

plt.tight_layout()
out_path = "Accuracy_MVOR_radar.pdf"
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()
print(out_path)
