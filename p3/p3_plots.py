# -*- coding: utf-8 -*-
"""
问题3 图表绘制（Python/matplotlib）
数据来源：p3/data/（p3_solver.py、p3_simulate.py 生成）
输出：p3/figures/ 三个 PNG（300dpi）
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import apply_style
apply_style()

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)


def fig1_baseline_compare():
    """图1：基准策略与最优策略的期望利润对比"""
    df = pd.read_csv(os.path.join(DATA, "p3_results.csv"))
    df = df.iloc[::-1]                       # 最优在最后，反转使最优在前
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52", "#C44E52", "#C44E52", "#C44E52", "#55A868"]
    bars = ax.barh(df["策略"], df["利润"], color=colors, height=0.6)
    for b, v in zip(bars, df["利润"]):
        ax.text(v + (2 if v >= 0 else -2), b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=10, color="#333")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("期望利润（元/件合格品）")
    ax.set_title("问题3：基准策略与最优策略对比（2道工序8零配件）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_baseline_compare.png"), dpi=300)
    plt.close(fig)


def fig2_quality_funnel():
    """图2：质量级联漏斗 —— 最优策略下各级合格率传递"""
    # 最优策略：配件全检测(100%合格入装) → 半成品不检测(90%合格) → 成品检测(65.61%卖出)
    stages = [
        ("零配件（全部检测）", 1.00, "检测拦截10%次品"),
        ("半成品装配（不检测）", 0.90, "半成品次品率10%"),
        ("成品（检测+拆解兜底）", 0.6561, "成品实际不合格率34.39%"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    widths = [s[1] for s in stages]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, ((name, w, note), c) in enumerate(zip(stages, colors)):
        ax.barh(i, w, height=0.55, color=c, alpha=0.85)
        ax.text(w / 2, i, f"{w*100:.2f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(1.02, i, note, va="center", fontsize=9, color="#555")
        ax.text(-0.02, i + 0.32, name, va="bottom", ha="right", fontsize=10.5)
    ax.set_xlim(-0.35, 1.55)
    ax.set_ylim(-0.6, 2.7)
    ax.axis("off")
    ax.set_title("质量级联：最优策略下从零配件到成品合格率的传递")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_quality_funnel.png"), dpi=300)
    plt.close(fig)


def fig3_cost_structure():
    """图3：最优策略的期望成本构成（半成品环节 vs 成品环节 vs 收入）"""
    # 从 solver 结果（66.0869, income 200, cost 133.91, cost_semi=107.33, cost_final=26.58）
    income = 200.0
    profit = 66.0869
    cost_semi = 107.3333
    cost_final = 133.9131 - 107.3333
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["期望收入", "半成品环节成本\n(配件采购+装配)", "成品环节成本\n(装配+检测+拆解)", "期望利润"]
    vals = [income, -cost_semi, -cost_final, profit]
    colors = ["#4C72B0", "#C44E52", "#C44E52", "#55A868"]
    bars = ax.bar(cats, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.5 if v >= 0 else -1.5),
                f"{v:.2f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("金额（元/件合格品）")
    ax.set_title("最优策略的期望收入-成本结构（利润 66.09 元/件）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_cost_structure.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_baseline_compare()
    fig2_quality_funnel()
    fig3_cost_structure()
    print("figures:", os.listdir(FIG))
