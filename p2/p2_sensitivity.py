# -*- coding: utf-8 -*-
"""
问题2 扩展：决策拐点敏感性分析（单变量扫描）

对每个关键参数单变量扫描（其余参数固定），全枚举 16 策略求最优，
观察最优决策随参数变化的"分岔点"——回答"参数变到多少，决策翻转"。

扫描参数（以情形1为基线）：
- d1 配件1检测费、d2 配件2检测费、r 调换损失、f 拆解费、s 市场售价

输出：p2/data/p2_sensitivity_*.csv、p2/figures/fig4_decision_bifurcation.png
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2.p2_solver import eval_strategy  # noqa: E402

CASE1 = dict(p1=0.10, c1=4, d1=2, p2=0.10, c2=18, d2=3, p0=0.10,
             a=6, d0=3, s=56, r=6, f=5)

# 扫描配置: (参数名, 起点, 终点, 步长, 说明)
SCANS = [
    ("d1", 0.5, 6.0, 0.25, "零配件1检测费"),
    ("d2", 0.5, 8.0, 0.25, "零配件2检测费"),
    ("r", 2.0, 40.0, 2.0, "不合格成品调换损失"),
    ("f", 1.0, 15.0, 1.0, "不合格成品拆解费用"),
    ("s", 30.0, 90.0, 5.0, "市场售价"),
]


def best_strategy(case):
    best = max(itertools.product([0, 1], repeat=4),
               key=lambda x: eval_strategy(case, *x)["profit"])
    p = eval_strategy(case, *best)["profit"]
    return best, p


def main():
    rows = []
    for param, lo, hi, step, note in SCANS:
        for v in np.round(np.arange(lo, hi + step / 2, step), 3):
            case = dict(CASE1)
            case[param] = v
            best, profit = best_strategy(case)
            rows.append({"参数": param, "参数值": v, "说明": note,
                         "最优策略": f"({best[0]},{best[1]},{best[2]},{best[3]})",
                         "策略编号": best[0] * 8 + best[1] * 4 + best[2] * 2 + best[3],
                         "期望利润": round(profit, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p2_sensitivity_results.csv"), index=False, encoding="utf-8-sig")

    # ---- 打印分岔点 ----
    print("===== 决策分岔点（情形1 基线，单变量扫描）=====")
    for param, _, _, _, note in SCANS:
        sub = df[df["参数"] == param]
        prev = None
        segs = []
        for _, r in sub.iterrows():
            if r["最优策略"] != prev:
                segs.append((r["参数值"], r["最优策略"]))
                prev = r["最优策略"]
        print(f"\n{note} {param}:")
        for v, s in segs:
            print(f"  参数={v:>6}: 最优策略 {s}")

    # ---- 图：5 参数分岔图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), sharey=False)
    axes = axes.ravel()
    for ax, (param, lo, hi, step, note) in zip(axes[:5], SCANS):
        sub = df[df["参数"] == param]
        ax.plot(sub["参数值"], sub["策略编号"], "-o", ms=4, lw=2, color="#4C72B0")
        ax.set_xlabel(note + f" {param}")
        ax.set_ylabel("最优策略编号 (x1x2x3x4→二进制)")
        ax.set_title(f"{note}扫描", fontsize=11)
        ax.grid(alpha=0.3)
        # 标注分岔值
        prev = None
        for _, r in sub.iterrows():
            if r["最优策略"] != prev:
                ax.annotate(f"{r['最优策略']}\n{r['参数值']:.2f}",
                            xy=(r["参数值"], r["策略编号"]), fontsize=7.5,
                            xytext=(0, -18), textcoords="offset points",
                            ha="center", color="#C44E52")
                prev = r["最优策略"]
    axes[5].axis("off")
    fig.suptitle("问题2 决策分岔图：单参数扫描下的最优策略变化（基线=情形1）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_decision_bifurcation.png"), dpi=300)
    plt.close(fig)
    print(f"\n[输出] {DATA}/p2_sensitivity_results.csv, {FIG}/fig4_decision_bifurcation.png")


if __name__ == "__main__":
    main()
