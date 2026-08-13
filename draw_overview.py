# -*- coding: utf-8 -*-
"""整篇论文技术路线图：四问关系（Python/matplotlib）"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import apply_style
apply_style()
os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, text, color="#4C72B0", fs=11, sub=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=color, edgecolor="k", alpha=0.9)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), text,
            ha="center", va="center", fontsize=fs, color="white", fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.26, sub, ha="center", va="center",
                fontsize=9, color="white", alpha=0.95)
    return (x, y, w, h)


def arrow(p1, p2, label=None, color="#333", ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                        lw=1.8, color=color, linestyle=ls)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.15, my + 0.12, label, fontsize=9, color="#333",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15",
                                                     fc="white", ec="#ccc", alpha=0.9))


# ---- 问题框 ----
b1 = box(0.4, 4.6, 2.6, 1.5, "问题1\n抽样检测方案", "#4C72B0",
         sub="判定可靠度读法\nn=29 拒收 / n=22 接收")
b2 = box(3.9, 4.6, 2.6, 1.5, "问题4\n抽样误差下的重解", "#C44E52",
         sub="CP精确置信区间\n三口径重解+蒙特卡洛")
b3 = box(3.9, 1.0, 2.6, 1.5, "问题2\n两配件生产决策", "#55A868",
         sub="期望利润模型（口径B）\n16策略全枚举")
b4 = box(7.1, 1.0, 2.6, 1.5, "问题3\n多级生产决策", "#8172B2",
         sub="级联递推+2^16枚举\n配件全检测+成品检测拆解")

# ---- 箭头 ----
arrow((b3[0] + b3[2], b3[1] + b3[3] / 2), (b4[0], b4[1] + b4[3] / 2),
      "模型推广\n（多级嵌套）")
arrow((b1[0] + b1[2], b1[1] + b1[3] / 2), (b2[0], b2[1] + b2[3] / 2),
      "抽样方法\n（n=22）", ls="--")
arrow((b2[0] + b2[2] / 2, b2[1]), (b4[0] + b4[2] / 2, b4[1] + b4[3]),
      "重解", ls="--")
arrow((b2[0] + b2[2] / 2, b2[1]), (b3[0] + b3[2] / 2, b3[1] + b3[3]),
      "重解", ls="--")

ax.text(5, 6.7, "2024 国赛 B 题：生产过程中的决策问题 —— 技术路线", ha="center",
        fontsize=14, fontweight="bold")
ax.text(5, 0.25, "数据与验证：Python 3.11 (mcm) 精确计算 + 蒙特卡洛模拟 + MATLAB 3D 示意",
        ha="center", fontsize=9.5, color="#666")

fig.tight_layout()
fig.savefig("figures/技术路线图.png", dpi=300)
plt.close(fig)
print("saved: figures/技术路线图.png")
