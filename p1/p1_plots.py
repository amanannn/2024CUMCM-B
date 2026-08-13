# -*- coding: utf-8 -*-
"""Generate publication-ready figures for Question 1."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plot_style import OKABE_ITO, apply_style

apply_style()

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

P0 = 0.10
BLUE, ORANGE = OKABE_ITO[:2]
INK = "#253238"
MUTED = "#6B7479"
GRID = "#D9DEE1"


def _finish(fig, name):
    """Export a high-resolution preview and a vector manuscript version."""
    fig.savefig(FIG / f"{name}.png", dpi=400, facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def _polish(ax, *, xgrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _callout(ax, text, xy, xytext, color, align="left"):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        ha=align,
        va="center",
        fontsize=7.5,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.9},
    )


def fig1_oc_curve():
    """Operating-characteristic curves for the two fixed-size plans."""
    oc = pd.read_csv(DATA / "p1_oc_curve.csv")
    p = oc["p"]
    pa29 = oc["n29_P接收"]
    pa22 = oc["n22_P接收"]
    y29, y22 = 0.9**29, 0.9**22

    fig, ax = plt.subplots(figsize=(7.1, 4.15))
    ax.plot(p, pa29, color=BLUE, lw=1.8, label=r"拒收方案：$n=29$")
    ax.plot(p, pa22, color=ORANGE, lw=1.8, ls="--", label=r"接收方案：$n=22$")

    ax.axvspan(0, P0, color="#EEF2F3", zorder=0)
    ax.axvline(P0, color=MUTED, ls=(0, (3, 2)), lw=0.9)
    ax.text(P0 + 0.004, 0.965, r"标称值  $p_0=10\%$", color=MUTED, va="top", fontsize=7.5)
    ax.scatter([P0], [y29], s=30, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter([P0], [y22], s=30, color=ORANGE, marker="s", edgecolor="white", linewidth=0.8, zorder=4)
    _callout(ax, f"{y29:.1%}", (P0, y29), (0.125, 0.135), BLUE)
    _callout(ax, f"{y22:.1%}", (P0, y22), (0.125, 0.225), ORANGE)

    ax.set(xlim=(0, 0.30), ylim=(0, 1.01), xlabel=r"真实次品率  $p$", ylabel="接收概率")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper right", frameon=False, handlelength=2.8)
    _polish(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.97)
    _finish(fig, "fig1_oc_curve")


def fig2_reliability_vs_n():
    """Decision reliability and error probability on one comparable scale."""
    rel = pd.read_csv(DATA / "p1_reliability_vs_n.csv")
    n = rel["n"]
    reliability = rel["拒收判定可靠度_1_0.9^n"]
    error = rel["接收判定出错概率_0.9^n"]
    y29, y22 = 1 - 0.9**29, 0.9**22

    fig, ax = plt.subplots(figsize=(7.1, 4.15))
    ax.plot(n, reliability, color=BLUE, lw=1.8, label="拒收判定可靠度")
    ax.plot(n, error, color=ORANGE, lw=1.8, ls="--", label="接收判定出错概率")

    ax.axhspan(0.95, 1.0, color=BLUE, alpha=0.07, zorder=0)
    ax.axhspan(0.0, 0.10, color=ORANGE, alpha=0.07, zorder=0)
    ax.axhline(0.95, color=BLUE, lw=0.8, ls=(0, (3, 2)), alpha=0.8)
    ax.axhline(0.10, color=ORANGE, lw=0.8, ls=(0, (3, 2)), alpha=0.8)
    ax.vlines(29, 0, y29, color=BLUE, lw=0.8, ls=(0, (2, 2)))
    ax.vlines(22, 0, y22, color=ORANGE, lw=0.8, ls=(0, (2, 2)))
    ax.scatter(29, y29, s=34, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(22, y22, s=34, color=ORANGE, marker="s", edgecolor="white", linewidth=0.8, zorder=4)
    _callout(ax, f"n=29  |  {y29:.1%}", (29, y29), (25.8, 0.84), BLUE, "right")
    _callout(ax, f"n=22  |  {y22:.1%}", (22, y22), (18.7, 0.20), ORANGE, "right")

    ax.set(xlim=(1, 40), ylim=(0, 1.01), xlabel=r"样本量  $n$", ylabel="概率")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="center right", frameon=False, handlelength=2.8)
    _polish(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.97)
    _finish(fig, "fig2_reliability_vs_n")


def fig3_min_n_vs_p0():
    """Sensitivity of the minimum sample size to the nominal defect rate."""
    sens = pd.read_csv(DATA / "p1_sensitivity.csv")

    fig, ax = plt.subplots(figsize=(7.1, 4.15))
    ax.plot(
        sens["p0"], sens["n1_拒收95%"], color=BLUE, lw=1.7, marker="o",
        ms=3.1, markevery=2, label="拒收方案（95% 信度）",
    )
    ax.plot(
        sens["p0"], sens["n2_接收90%"], color=ORANGE, lw=1.7, ls="--",
        marker="s", ms=3.0, markevery=2, label="接收方案（90% 信度）",
    )
    ax.axvline(P0, color=MUTED, ls=(0, (3, 2)), lw=0.9)
    ax.scatter(P0, 29, s=38, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(P0, 22, s=38, color=ORANGE, marker="s", edgecolor="white", linewidth=0.8, zorder=4)
    _callout(ax, "n=29", (P0, 29), (0.086, 37), BLUE, "right")
    _callout(ax, "n=22", (P0, 22), (0.116, 17.8), ORANGE)
    ax.text(P0 + 0.003, 245, r"$p_0=10\%$", color=MUTED, fontsize=7.5)

    ax.set_yscale("log")
    ax.set(xlim=(0.01, 0.20), ylim=(9, 330), xlabel=r"标称次品率  $p_0$", ylabel=r"最小样本量  $n$")
    ax.xaxis.set_major_locator(MultipleLocator(0.025))
    ax.legend(loc="upper right", frameon=False, handlelength=2.8)
    _polish(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.97)
    _finish(fig, "fig3_min_n_vs_p0")


if __name__ == "__main__":
    fig1_oc_curve()
    fig2_reliability_vs_n()
    fig3_min_n_vs_p0()
    print("Figures written to", FIG)
