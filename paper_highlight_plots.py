# -*- coding: utf-8 -*-
"""High-value synthesis figures derived from the saved model outputs."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_style import OKABE_ITO, apply_style

apply_style()
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, VERMILION, SKY = OKABE_ITO[:5]
INK, MUTED, GRID, LIGHT = "#253238", "#6B7479", "#D9DEE1", "#EEF2F3"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=400, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def strategy_code(df):
    return df[["x1", "x2", "x3", "x4"]].astype(int).astype(str).agg("".join, axis=1)


def fig_strategy_regret():
    """Minimax-regret view of all 16 strategies across the six cases."""
    df = pd.read_csv(ROOT / "p2" / "data" / "p2_all_strategies.csv")
    df["策略"] = strategy_code(df)
    profit = df.pivot(index="策略", columns="情形", values="profit").sort_index()
    regret = profit.max(axis=0) - profit
    summary = pd.DataFrame({
        "平均遗憾": regret.mean(axis=1),
        "最大遗憾": regret.max(axis=1),
        "最差利润": profit.min(axis=1),
        "平均利润": profit.mean(axis=1),
    })
    robust = summary.sort_values(["最大遗憾", "平均遗憾"]).index[0]
    case_optima = set(profit.idxmax(axis=0))

    fig = plt.figure(figsize=(7.2, 3.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])

    for code, row in summary.iterrows():
        if code == robust:
            color, marker, size, zorder = GREEN, "*", 115, 5
        elif code in case_optima:
            color, marker, size, zorder = BLUE, "o", 43, 3
        else:
            color, marker, size, zorder = "#AAB4B8", "o", 30, 2
        ax.scatter(row["平均遗憾"], row["最大遗憾"], s=size, color=color, marker=marker,
                   edgecolor="white", linewidth=0.65, zorder=zorder)

    label_offsets = {
        "0011": (7, -3), "0001": (7, 3), "1001": (7, 2),
        "1011": (7, -7), "0000": (-7, 2),
    }
    for code in sorted(case_optima | {robust}):
        row = summary.loc[code]
        dx, dy = label_offsets.get(code, (6, 3))
        ax.annotate(code, (row["平均遗憾"], row["最大遗憾"]), xytext=(dx, dy),
                    textcoords="offset points", ha="right" if dx < 0 else "left",
                    color=GREEN if code == robust else INK, fontsize=7,
                    weight="bold" if code == robust else "normal")

    ax.set_xlabel("六种情形的平均遗憾（元/件）")
    ax.set_ylabel("最坏情形遗憾（元/件）")
    ax.set_xlim(-0.1, summary["平均遗憾"].max() + 1.5)
    ax.set_ylim(-0.5, summary["最大遗憾"].max() + 3.0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    legend = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GREEN,
               markeredgecolor="white", markersize=10, label="最小最大遗憾策略"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=6, label="至少一个情形最优"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#AAB4B8",
               markeredgecolor="white", markersize=6, label="其余策略"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False)
    ax.text(-0.18, 1.03, "a", transform=ax.transAxes, weight="bold", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    selected = summary.sort_values(["最大遗憾", "平均遗憾"]).head(7).index
    matrix = regret.loc[selected].to_numpy()
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=matrix.max(),
                   interpolation="nearest")
    ax.set_xticks(np.arange(regret.shape[1]), [f"情形 {int(k)}" for k in regret.columns])
    ax.set_yticks(np.arange(len(selected)), selected)
    ax.set_xlabel("参数情形")
    ax.set_ylabel(r"策略编码  $x_1x_2x_3x_4$")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if value > matrix.max() * 0.58 else INK,
                    weight="bold" if selected[i] == robust else "normal")
    robust_row = list(selected).index(robust)
    ax.add_patch(Rectangle((-0.48, robust_row - 0.46), matrix.shape[1] - 0.04, 0.92,
                           fill=False, ec=GREEN, lw=1.35))
    cbar = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.05)
    cbar.set_label("相对该情形最优解的利润损失（元）")
    cbar.outline.set_visible(False)
    ax.text(-0.16, 1.03, "b", transform=ax.transAxes, weight="bold", fontsize=9)

    row = summary.loc[robust]
    fig.text(0.5, 0.995,
             f"稳健策略 {robust}：平均遗憾 {row['平均遗憾']:.2f} 元，最坏遗憾 {row['最大遗憾']:.2f} 元",
             ha="center", va="top", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.90)
    save(fig, "fig_strategy_regret")


def fig_evsi_phase_map():
    """Economic phase map for whether and how much to sample."""
    df = pd.read_csv(ROOT / "p4" / "data" / "p4_evsi_sensitivity.csv")
    cases = sorted(df["情形"].unique())
    batches = sorted(df["批量B"].unique())
    costs = sorted(df["检测成本d"].unique())
    norm = Normalize(vmin=0, vmax=df["EVSI"].max())

    fig, axes = plt.subplots(1, len(cases), figsize=(7.2, 3.15), sharey=True)
    for panel, (ax, case) in enumerate(zip(axes, cases)):
        sub = df[df["情形"] == case]
        evsi = sub.pivot(index="批量B", columns="检测成本d", values="EVSI").loc[batches, costs]
        nstar = sub.pivot(index="批量B", columns="检测成本d", values="最优n*").loc[batches, costs]
        im = ax.imshow(evsi.to_numpy(), cmap="YlGnBu", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(costs)), [f"{d:g}" for d in costs])
        ax.set_yticks(np.arange(len(batches)), [f"{b:,}" for b in batches])
        ax.set_xlabel(r"单件检测成本  $d$（元）")
        ax.set_title(f"情形 {case}", loc="left", color=INK, pad=7)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        for i, batch in enumerate(batches):
            for j, cost in enumerate(costs):
                e = float(evsi.loc[batch, cost])
                n = int(nstar.loc[batch, cost])
                text_color = "white" if norm(e) > 0.50 else INK
                evsi_label = "EVSI ≈0.00" if e == 0 and n > 0 else f"EVSI {e:.2f}"
                ax.text(j, i - 0.10, evsi_label, ha="center", va="center",
                        color=text_color, fontsize=7, weight="bold" if e > 0 else "normal")
                ax.text(j, i + 0.20, rf"$n^*={n}$", ha="center", va="center",
                        color=text_color, fontsize=7)
                if n > 0:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.46), 0.94, 0.92,
                                           fill=False, ec=ORANGE, lw=1.35))
        ax.text(-0.13, 1.04, chr(97 + panel), transform=ax.transAxes, weight="bold", fontsize=9)
    axes[0].set_ylabel(r"生产批量  $B$（件，对数分级）")
    cbar = fig.colorbar(im, ax=axes, pad=0.025, fraction=0.035)
    cbar.set_label("抽样信息净价值 EVSI（元/件）")
    cbar.outline.set_visible(False)
    fig.text(0.5, 0.995, r"橙色边框表示最优决策为抽样（$n^*>0$）；EVSI 按 0.01 元显示",
             ha="center", va="top", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.18, top=0.87, wspace=0.20)
    save(fig, "fig_evsi_phase_map")


def fig_strategy_upset():
    """UpSet-style view of strategy prevalence in the sampled parameter space."""
    df = pd.read_csv(ROOT / "p2" / "data" / "p2_rules.csv")
    df["策略"] = strategy_code(df)
    counts = df["策略"].value_counts().reindex([f"{i:04b}" for i in range(16)], fill_value=0)
    counts = counts.sort_values(ascending=False)
    codes = counts.index.tolist()
    rates = counts.to_numpy() / len(df)
    decision_rates = df[["x1", "x2", "x3", "x4"]].mean().to_numpy()
    x = np.arange(len(codes))

    fig = plt.figure(figsize=(7.2, 4.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 5.4], height_ratios=[2.2, 1.65],
                          wspace=0.02, hspace=0.04)
    ax_blank = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_blank.axis("off")

    colors = [ORANGE if code == codes[0] else BLUE for code in codes]
    bars = ax_bar.bar(x, rates * 100, color=colors, width=0.72)
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        if i < 6 or rate >= 0.03:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.65,
                        f"{rate:.1%}", ha="center", va="bottom", fontsize=6.4,
                        color=INK, weight="bold" if i == 0 else "normal")
    ax_bar.set_ylabel("参数样本占比（%）")
    ax_bar.set_ylim(0, rates.max() * 100 + 5.0)
    ax_bar.grid(axis="y", color=GRID, lw=0.5)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.spines["left"].set_color(INK)
    ax_bar.spines["bottom"].set_visible(False)

    decisions = [r"$x_1$  配件1检测", r"$x_2$  配件2检测", r"$x_3$  成品检测", r"$x_4$  不合格品拆解"]
    y = np.arange(4)
    for i, code in enumerate(codes):
        bits = np.array([int(v) for v in code])
        active = np.where(bits == 1)[0]
        ax_matrix.scatter(np.full(4, i), y, s=17, facecolor="#E1E6E8", edgecolor="none", zorder=1)
        if len(active):
            ax_matrix.plot([i, i], [active.min(), active.max()], color=INK, lw=1.0, zorder=2)
            ax_matrix.scatter(np.full(len(active), i), active, s=24,
                              color=ORANGE if i == 0 else BLUE, edgecolor="white", lw=0.4, zorder=3)
    ax_matrix.set_yticks(y)
    ax_matrix.set_yticklabels([])
    ax_matrix.set_xticks(x, codes, rotation=90)
    ax_matrix.set_xlabel(r"策略编码  $x_1x_2x_3x_4$（按出现频率排序）")
    ax_matrix.set_ylim(3.55, -0.55)
    ax_matrix.tick_params(axis="both", length=0)
    ax_matrix.spines[:].set_visible(False)
    for row in [0.5, 1.5, 2.5]:
        ax_matrix.axhline(row, color=GRID, lw=0.45, zorder=0)

    side_bars = ax_side.barh(y, decision_rates * 100, color=GREEN, height=0.48)
    for bar, rate, decision in zip(side_bars, decision_rates, decisions):
        ax_side.text(-83, bar.get_y() + bar.get_height() / 2, decision,
                     ha="left", va="center", color=INK, fontsize=7.2)
        ax_side.text(bar.get_width() - 1.0, bar.get_y() + bar.get_height() / 2,
                     f"{rate:.1%}", ha="right", va="center", color="white", fontsize=6.6,
                     weight="bold")
    ax_side.set_xlim(-85, 100)
    ax_side.set_ylim(3.55, -0.55)
    ax_side.set_title("边际采用率", loc="right", color=MUTED, fontsize=7.2, pad=4)
    ax_side.set_yticks([])
    ax_side.spines[:].set_visible(False)
    ax_side.tick_params(axis="x", length=0, labelbottom=False)

    fig.text(0.5, 0.995,
             f"8,000 组参数样本中，策略 {codes[0]} 出现频率最高（{rates[0]:.1%}）",
             ha="center", va="top", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.15, top=0.92)
    save(fig, "fig_strategy_upset")


def main():
    fig_strategy_regret()
    fig_evsi_phase_map()
    fig_strategy_upset()
    print("Highlight figures written to", OUT)


if __name__ == "__main__":
    main()
