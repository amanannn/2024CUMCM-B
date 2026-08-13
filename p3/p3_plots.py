# -*- coding: utf-8 -*-
"""Publication-ready figures for Question 3, generated from saved results."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plot_style import OKABE_ITO, apply_style

apply_style()
BASE = Path(__file__).resolve().parent
DATA, FIG = BASE / "data", BASE / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, VERMILION, SKY, PURPLE = OKABE_ITO[:6]
INK, MUTED, GRID = "#253238", "#6B7479", "#D9DEE1"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=400, facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def polish(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)


def fig1_baseline_compare():
    df = pd.read_csv(DATA / "p3_results.csv").iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    colors = [GREEN if label == "★最优" else (VERMILION if value < 0 else "#AAB4B8")
              for label, value in zip(df["策略"], df["利润"])]

    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    bars = ax.barh(y, df["利润"], color=colors, height=0.58)
    for bar, value in zip(bars, df["利润"]):
        offset = 3 if value >= 0 else -3
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", ha="left" if value >= 0 else "right", color=INK, fontsize=7.5,
                weight="bold" if value == df["利润"].max() else "normal")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, df["策略"])
    ax.set_xlabel("期望利润（元/件合格品）")
    ax.set_xlim(df["利润"].min() - 28, df["利润"].max() + 24)
    polish(ax)
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.16, top=0.97)
    save(fig, "fig1_baseline_compare")


def fig2_quality_funnel():
    stages = [
        ("合格零配件进入装配", 1.0000, "全检拦截 10% 次品"),
        ("合格半成品", 0.9000, "半成品不检测"),
        ("合格成品", 0.6561, "成品检测并拆解兜底"),
    ]
    y = np.array([2, 1, 0])
    widths = np.array([s[1] for s in stages])
    colors = [BLUE, SKY, GREEN]

    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    for i, ((label, value, note), ypos, color) in enumerate(zip(stages, y, colors)):
        left = (1 - value) / 2
        ax.barh(ypos, value, left=left, height=0.52, color=color)
        ax.text(0.5, ypos, f"{value:.1%}", ha="center", va="center", color="white",
                fontsize=9, weight="bold")
        ax.text(0.0, ypos, label, ha="right", va="center", color=INK, fontsize=8)
        ax.text(1.02, ypos, note, ha="left", va="center", color=MUTED, fontsize=7.2)
        if i < 2:
            next_value = widths[i + 1]
            loss = 1 - next_value / value
            ax.annotate(f"损失 {loss:.1%}", xy=(0.5, ypos - 0.29), xytext=(0.5, ypos - 0.63),
                        ha="center", va="center", color=VERMILION, fontsize=7.2,
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.7})
    ax.set_xlim(-0.04, 1.37)
    ax.set_ylim(-0.52, 2.5)
    ax.axis("off")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.05, top=0.98)
    save(fig, "fig2_quality_funnel")


def fig3_cost_structure():
    income = 200.0
    cost_semi = 107.3333
    cost_final = 133.9131 - cost_semi
    profit = 66.0869
    starts = [0, income, income - cost_semi, 0]
    values = [income, -cost_semi, -cost_final, profit]
    labels = ["期望收入", "半成品环节", "成品环节", "期望利润"]
    colors = [BLUE, VERMILION, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    x = np.arange(4)
    for i, (start, value, color) in enumerate(zip(starts, values, colors)):
        bottom = start if value >= 0 else start + value
        ax.bar(i, abs(value), bottom=bottom, width=0.58, color=color)
        edge = start + value
        ax.text(i, edge + (6 if value >= 0 else -6), f"{abs(value):.2f}", ha="center",
                va="bottom" if value >= 0 else "top", color=INK, fontsize=7.5, weight="bold")
    levels = [income, income - cost_semi, profit]
    for i, level in enumerate(levels):
        ax.plot([i + 0.30, i + 0.70], [level, level], color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("金额（元/件合格品）")
    ax.set_ylim(0, 225)
    polish(ax, "y")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.97)
    save(fig, "fig3_cost_structure")


def fig4_flow_network():
    part_y = np.linspace(0.92, 0.08, 8)
    semi_y = [0.76, 0.46, 0.17]
    final_y = 0.46
    semi_of = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
    x_part, x_semi, x_final = 0.12, 0.54, 0.88

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, js in semi_of.items():
        for j in js:
            ax.add_patch(FancyArrowPatch((x_part + 0.065, part_y[j]), (x_semi - 0.08, semi_y[i]),
                                         arrowstyle="-|>", mutation_scale=7, lw=0.75,
                                         color="#AAB4B8", connectionstyle="arc3,rad=0"))
    for y in semi_y:
        ax.add_patch(FancyArrowPatch((x_semi + 0.08, y), (x_final - 0.085, final_y),
                                     arrowstyle="-|>", mutation_scale=8, lw=1.0, color=MUTED))

    for i, y in enumerate(part_y, 1):
        rect = Rectangle((x_part - 0.065, y - 0.034), 0.13, 0.068, facecolor="#E6F0F7",
                         edgecolor=BLUE, lw=0.8)
        ax.add_patch(rect)
        ax.text(x_part, y, f"零配件 {i}", ha="center", va="center", color=INK, fontsize=7)
    for i, y in enumerate(semi_y, 1):
        rect = Rectangle((x_semi - 0.08, y - 0.045), 0.16, 0.09, facecolor="#E4F2ED",
                         edgecolor=GREEN, lw=0.9)
        ax.add_patch(rect)
        ax.text(x_semi, y, f"半成品 {i}", ha="center", va="center", color=INK, fontsize=7.4)
    rect = Rectangle((x_final - 0.085, final_y - 0.052), 0.17, 0.104, facecolor="#FBEDE8",
                     edgecolor=VERMILION, lw=1.0)
    ax.add_patch(rect)
    ax.text(x_final, final_y, "成品", ha="center", va="center", color=INK, fontsize=8, weight="bold")
    ax.text(x_part, 1.01, "零配件层", ha="center", color=MUTED, fontsize=7.5)
    ax.text(x_semi, 1.01, "半成品层", ha="center", color=MUTED, fontsize=7.5)
    ax.text(x_final, 1.01, "成品层", ha="center", color=MUTED, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.01, 1.06)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
    save(fig, "fig4_flow_3d")


def fig5_sensitivity():
    df = pd.read_csv(DATA / "p3_sensitivity_results.csv")
    order = ["S", "F_F", "D_H", "F_H"]
    symbols = {"S": r"$S$", "F_F": r"$F_F$", "D_H": r"$D_H$", "F_H": r"$F_H$"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for panel, (ax, param) in enumerate(zip(axes.ravel(), order)):
        sub = df[df["参数"] == param].sort_values("参数值")
        ax.step(sub["参数值"], sub["Σx(检测配件数)"], where="mid", color=BLUE, lw=1.5,
                label="检测配件数")
        ax.step(sub["参数值"], sub["Σy(检测半成品数)"], where="mid", color=GREEN, lw=1.5,
                ls="--", label="检测半成品数")
        ax.step(sub["参数值"], sub["Σz(拆解半成品数)"], where="mid", color=ORANGE, lw=1.25,
                ls=(0, (2, 2)), label="拆解半成品数")
        decision_cols = ["Σx(检测配件数)", "Σy(检测半成品数)", "Σz(拆解半成品数)",
                         "yf(成品检测)", "zf(成品拆解)"]
        change = sub[decision_cols].ne(sub[decision_cols].shift()).any(axis=1)
        for value in sub.loc[change, "参数值"].iloc[1:]:
            ax.axvline(value, color=MUTED, lw=0.7, ls=(0, (2, 2)))
        ax.set_ylim(-0.25, 8.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_xlabel(f"{sub['说明'].iloc[0]}  {symbols[param]}")
        ax.set_ylabel("决策数量")
        ax.text(-0.15, 1.02, chr(97 + panel), transform=ax.transAxes, weight="bold")
        polish(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.00))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.90, wspace=0.32, hspace=0.40)
    save(fig, "fig5_sensitivity")


def fig6_rule_tree():
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    df = pd.read_csv(DATA / "p3_rules.csv")
    features = ["c_j", "d_j", "p_j", "同半成品配件数", "半成品平均单价", "S", "F_F"]
    display = [r"$c_j$", r"$d_j$", r"$p_j$", "同组配件数", "同组均价", r"$S$", r"$F_F$"]
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(df))
    n_train = int(0.8 * len(df))
    train, valid = df.iloc[idx[:n_train]], df.iloc[idx[n_train:]]
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.02, random_state=7)
    clf.fit(train[features], train["检测y"])
    accuracy = clf.score(valid[features], valid["检测y"])

    fig, ax = plt.subplots(figsize=(9.2, 4.25))
    plot_tree(clf, feature_names=display, class_names=["不检测", "检测"], filled=True,
              impurity=False, proportion=True, rounded=False, fontsize=6, ax=ax)
    for text in ax.texts:
        patch = text.get_bbox_patch()
        if patch:
            patch.set_facecolor("#E4F2ED" if "class = 检测" in text.get_text() else "#EEF2F3")
            patch.set_edgecolor("#AAB4B8")
            patch.set_linewidth(0.55)
    ax.set_title(f"配件检测决策树   |   留出集准确率 {accuracy:.1%}", loc="left", color=INK, pad=8)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92)
    save(fig, "fig6_rule_tree_part")


def main():
    fig1_baseline_compare()
    fig2_quality_funnel()
    fig3_cost_structure()
    fig4_flow_network()
    fig5_sensitivity()
    fig6_rule_tree()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
