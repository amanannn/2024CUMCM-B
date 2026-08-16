# -*- coding: utf-8 -*-
"""
问题1 扩展：信度敏感性 + 单调性检验（建模手建议）

1. 信度扫描：情形一拒收信度 (1-α1)、情形二接收信度 (1-α2) 在 80%~99% 变化，
   最小样本量 n1/n2 随之变化（通式 n=⌈ln(1-信度)/ln(1-p0)⌉）；
2. 单调性检验：n 对信度单调不减（信度越高越要多抽），对 p0 单调不增
   （标称值越严越要多抽）——解析导数/差分验证；
3. 输出曲线图 + 数据。

输出：p1/data/p1_confidence_sweep.csv、p1/figures/fig5_confidence_sweep.png
"""
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import apply_style

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

P0 = 0.10


def min_n(conf: float, p0: float = P0) -> int:
    """最小样本量：n = ⌈ln(1-信度)/ln(1-p0)⌉"""
    return int(np.ceil(np.log(1 - conf) / np.log(1 - p0)))


def main():
    conf_grid = np.round(np.arange(0.80, 0.995, 0.005), 3)
    rows = [{"信度(1-α)": c, "拒收方案n1": min_n(c), "接收方案n2": min_n(c)} for c in conf_grid]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p1_confidence_sweep.csv"), index=False, encoding="utf-8-sig")

    # ---- 单调性检验 ----
    n1s, n2s = df["拒收方案n1"].to_numpy(), df["接收方案n2"].to_numpy()
    confs = df["信度(1-α)"].to_numpy()
    mono_conf = bool(np.all(np.diff(n1s) >= 0)) and bool(np.all(np.diff(n2s) >= 0))
    # p0 单调性：p0 增大 → n 不增（固定信度 95%）
    p0_grid = np.linspace(0.01, 0.30, 60)
    n_p0 = [min_n(0.95, p) for p in p0_grid]
    mono_p0 = bool(np.all(np.diff(n_p0) <= 0))
    print(f"[单调性检验] n 随信度单调不减: {'通过 ✓' if mono_conf else '失败 ✗'}")
    print(f"[单调性检验] n 随 p0 单调不增: {'通过 ✓' if mono_p0 else '失败 ✗'}")
    print(df.to_string(index=False))

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(confs, n1s, "-o", ms=3.5, lw=1.8, color="#0072B2", label=r"拒收方案 $n_1$")
    ax.plot(confs, n2s, "-s", ms=3.5, lw=1.8, color="#E69F00", label=r"接收方案 $n_2$")
    ax.axvline(0.95, color="#D55E00", ls="--", lw=1)
    ax.axvline(0.90, color="#009E73", ls="--", lw=1)
    ax.annotate("95%→n=29", xy=(0.95, 29), xytext=(0.87, 55),
                arrowprops=dict(arrowstyle="->", color="#D55E00"), color="#D55E00", fontsize=8)
    ax.annotate("90%→n=22", xy=(0.90, 22), xytext=(0.82, 12),
                arrowprops=dict(arrowstyle="->", color="#009E73"), color="#009E73", fontsize=8)
    ax.set_xlabel("判定信度 (1−α)")
    ax.set_ylabel("最小样本量 n")
    ax.set_title(r"样本量随判定信度的变化（$p_0$=10%）")
    ax.legend()

    ax = axes[1]
    ax.plot(p0_grid, n_p0, "-", lw=2, color="#0072B2")
    ax.axvline(P0, color="gray", ls=":", lw=1)
    ax.annotate(f"$p_0$=10%→n={min_n(0.95)}", xy=(P0, min_n(0.95)), xytext=(0.13, 180),
                arrowprops=dict(arrowstyle="->", color="#009E73"), color="#009E73", fontsize=8)
    ax.set_xlabel(r"标称次品率 $p_0$")
    ax.set_ylabel("最小样本量 n（95%信度）")
    ax.set_title("样本量随标称次品率的变化（单调递减）")
    ax.set_yscale("log")
    ax.grid(axis="y", color="#D9DEE1", lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_confidence_sweep.png"), dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[输出] {DATA}/p1_confidence_sweep.csv, {FIG}/fig5_confidence_sweep.png")


if __name__ == "__main__":
    main()
