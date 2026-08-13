# -*- coding: utf-8 -*-
"""
问题 1 图表绘制（数据分析类 → Python/matplotlib）
数据来源：p1/data/ 下的 csv 文件（由 p1_sampling.py 生成）
输出：p1/figures/ 三个 PNG（300dpi）
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

P0 = 0.10


def fig1_oc_curve():
    """图1：OC 曲线 —— 两个方案的接收概率随真实次品率 p 的变化"""
    oc = pd.read_csv(os.path.join(DATA, "p1_oc_curve.csv"))
    p, pa29, pa22 = oc["p"], oc["n29_P接收"], oc["n22_P接收"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(p, pa29, label=f"情形一方案 (n={29}, 次品数≥1拒收)", lw=2)
    ax.plot(p, pa22, label=f"情形二方案 (n={22}, 0件次品接收)", lw=2)
    ax.axvline(P0, color="gray", ls="--", lw=1)
    ax.text(P0 + 0.005, 0.86, r"$p_0=0.10$", color="gray")
    # 标出 p=p0 处的关键点
    ax.plot(P0, 0.9 ** 29, "o", color="C0", ms=6)
    ax.plot(P0, 0.9 ** 22, "o", color="C1", ms=6)
    ax.annotate(f"P(接收|$p_0$)={0.9**29:.4f}", xy=(P0, 0.9 ** 29), xytext=(0.16, 0.12),
                arrowprops=dict(arrowstyle="->", color="C0"), color="C0")
    ax.annotate(f"P(接收|$p_0$)={0.9**22:.4f}", xy=(P0, 0.9 ** 22), xytext=(0.20, 0.30),
                arrowprops=dict(arrowstyle="->", color="C1"), color="C1")
    ax.set_xlabel("真实次品率 $p$")
    ax.set_ylabel("接收概率 P(接收|p)")
    ax.set_title("OC 曲线：两种抽样方案的接收概率（标称值 $p_0=10\\%$）")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_oc_curve.png"), dpi=300)
    plt.close(fig)


def fig2_reliability_vs_n():
    """图2：判定可靠度随样本量 n 的变化（双轴），标注两个最小样本量"""
    rel = pd.read_csv(os.path.join(DATA, "p1_reliability_vs_n.csv"))
    n = rel["n"]
    r_reject = rel["拒收判定可靠度_1_0.9^n"]
    e_accept = rel["接收判定出错概率_0.9^n"]

    fig, ax1 = plt.subplots(figsize=(8, 5.2))
    ax1.plot(n, r_reject, lw=2, color="C0", label="拒收判定可靠度 $1-0.9^n$")
    ax1.axhline(0.95, color="C0", ls="--", lw=1)
    ax1.axvline(29, color="C0", ls=":", lw=1)
    ax1.plot(29, 1 - 0.9 ** 29, "o", color="C0", ms=7)
    ax1.annotate(f"n=29, 可靠度={1-0.9**29:.4f}", xy=(29, 1 - 0.9 ** 29),
                 xytext=(18, 0.42), arrowprops=dict(arrowstyle="->", color="C0"), color="C0")
    ax1.set_xlabel("样本量 $n$")
    ax1.set_ylabel("拒收判定可靠度 $1-(1-p_0)^n$", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.legend(loc="center right")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(n, e_accept, lw=2, color="C1", ls="--", label="接收判定出错概率 $0.9^n$")
    ax2.axhline(0.10, color="C1", ls="--", lw=1)
    ax2.axvline(22, color="C1", ls=":", lw=1)
    ax2.plot(22, 0.9 ** 22, "s", color="C1", ms=7)
    ax2.annotate(f"n=22, 出错概率={0.9**22:.4f}", xy=(22, 0.9 ** 22),
                 xytext=(6, 0.55), arrowprops=dict(arrowstyle="->", color="C1"), color="C1")
    ax2.set_ylabel("接收判定出错概率 $(1-p_0)^n$", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.legend(loc="lower left")
    ax2.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_reliability_vs_n.png"), dpi=300)
    plt.close(fig)


def fig3_min_n_vs_p0():
    """图3：通用公式 —— 最小样本量随标称次品率 p0 的变化（对数纵轴）"""
    sens = pd.read_csv(os.path.join(DATA, "p1_sensitivity.csv"))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(sens["p0"], sens["n1_拒收95%"], "o-", ms=3, lw=1.5, label="情形一：拒收 (95%信度)")
    ax.plot(sens["p0"], sens["n2_接收90%"], "s-", ms=3, lw=1.5, label="情形二：接收 (90%信度)")
    ax.set_yscale("log")
    ax.axvline(P0, color="gray", ls="--", lw=1)
    ax.plot(P0, 29, "o", color="C0", ms=8)
    ax.plot(P0, 22, "s", color="C1", ms=8)
    ax.annotate(r"$p_0=10\%$: n=29", xy=(P0, 29), xytext=(0.075, 26),
                arrowprops=dict(arrowstyle="->", color="C0"), color="C0")
    ax.annotate(r"$p_0=10\%$: n=22", xy=(P0, 22), xytext=(0.115, 18),
                arrowprops=dict(arrowstyle="->", color="C1"), color="C1")
    ax.set_xlabel("标称次品率 $p_0$")
    ax.set_ylabel("最小样本量 $n$（对数轴）")
    ax.set_title("通用公式：最小样本量随标称次品率的变化")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_min_n_vs_p0.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_oc_curve()
    fig2_reliability_vs_n()
    fig3_min_n_vs_p0()
    print("figures:", os.listdir(FIG))
