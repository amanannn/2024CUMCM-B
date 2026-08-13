# -*- coding: utf-8 -*-
"""Publication-ready EVSI figure for Question 4 from saved CSV results."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
BLUE, ORANGE = OKABE_ITO[:2]
INK, MUTED, GRID = "#253238", "#6B7479", "#D9DEE1"


def main():
    curve = pd.read_csv(DATA / "p4_evsi_curve.csv")
    result = pd.read_csv(DATA / "p4_evsi_results.csv")
    cases = sorted(curve["情形"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), sharex=True)
    for panel, (ax, case) in enumerate(zip(axes.ravel(), cases)):
        sub = curve[curve["情形"] == case].sort_values("n")
        row = result[result["情形"] == case].iloc[0]
        n_star = int(row["最优抽样量n*"])
        value_star = float(row["V(n*)"])
        evsi = float(row["EVSI"])

        ax.plot(sub["n"], sub["V(n)"], color=BLUE, lw=1.55, marker="o", ms=2.7)
        ax.scatter(n_star, value_star, s=34, color=ORANGE, edgecolor="white", lw=0.7, zorder=4)
        if n_star > 0:
            ax.axvline(n_star, color=ORANGE, lw=0.8, ls=(0, (3, 2)))
        annotation = rf"$n^*={n_star}$" + "\n" + rf"EVSI={evsi:.3f}"
        x_text = n_star + 8 if n_star < 60 else n_star - 8
        align = "left" if n_star < 60 else "right"
        y_span = sub["V(n)"].max() - sub["V(n)"].min()
        y_text = value_star - max(y_span * 0.22, 0.07)
        ax.annotate(annotation, xy=(n_star, value_star), xytext=(x_text, y_text),
                    ha=align, va="top", color=ORANGE, fontsize=7,
                    arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7},
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})

        margin = max(y_span * 0.12, 0.04)
        ax.set_ylim(sub["V(n)"].min() - margin, sub["V(n)"].max() + margin)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(f"情形 {case}", loc="left", color=INK, pad=4)
        ax.text(-0.18, 1.03, chr(97 + panel), transform=ax.transAxes, weight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.spines["left"].set_color(INK)
        ax.spines["bottom"].set_color(INK)
        ax.tick_params(colors=INK)

    for ax in axes[-1]:
        ax.set_xlabel(r"每个参数的抽样量  $n$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"事前期望价值  $V(n)$")

    fig.text(0.5, 0.995, "Beta 先验等价样本量 30   |   检测费 2 元/件   |   批量 1,000 件",
             ha="center", va="top", color=MUTED, fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.92, wspace=0.37, hspace=0.38)
    fig.savefig(FIG / "fig5_evsi.png", dpi=400, facecolor="white")
    fig.savefig(FIG / "fig5_evsi.pdf", facecolor="white")
    plt.close(fig)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
