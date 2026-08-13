# -*- coding: utf-8 -*-
"""
问题2 图表绘制（Python/matplotlib）
数据来源：p2/data/（p2_solver.py、p2_simulate.py 生成）
输出：p2/figures/ 三个 PNG（300dpi）
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


def fig1_strategy_heatmap():
    """图1：6情形 × 16策略 利润热力图"""
    df = pd.read_csv(os.path.join(DATA, "p2_all_strategies.csv"))
    df["策略"] = df.apply(lambda r: f"{int(r.x1)}{int(r.x2)}{int(r.x3)}{int(r.x4)}", axis=1)
    piv = df.pivot(index="情形", columns="策略", values="profit")
    # 策略顺序按二进制 0000→1111
    cols = sorted(piv.columns, key=lambda s: int(s, 2))
    piv = piv[cols]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(piv, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c[:2]} {c[2:]}" for c in cols], fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"情形{k}" for k in piv.index])
    ax.set_xlabel("策略 (x1x2 | x3x4)")
    ax.set_title("各情形 16 种策略的期望利润（元/件成品）")
    for i in range(len(piv.index)):
        for j in range(len(cols)):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="black" if 5 < v < 19 else "white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("期望利润（元）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_strategy_heatmap.png"), dpi=300)
    plt.close(fig)


def fig2_profit_decomposition():
    """图2：各情形最优策略的 收入-成本 分解（堆叠条形）"""
    df = pd.read_csv(os.path.join(DATA, "p2_results.csv"))
    labels = [f"情形{k}\n(x1,x2,x3,x4)=({int(r['x1'])},{int(r['x2'])},{int(r['x3'])},{int(r['x4'])})"
              for k, r in df.iterrows()]

    x = range(len(df))
    w = 0.55
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    costs = {
        "配件成本": df["cost_part"],
        "装配成本": df["cost_assembly"],
        "成品检测": df["cost_final_test"],
        "拆解+回收": df["cost_disassemble"] + df["cost_recover"],
        "调换损失": df["cost_switch"],
    }
    bottom = pd.Series(0.0, index=df.index)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for (name, v), col in zip(costs.items(), colors):
        ax.bar(x, v, w, bottom=bottom, label=name, color=col)
        bottom = bottom + v
    # 收入线 + 利润标注
    ax.plot(x, df["income"], "D--", color="black", ms=7, label="期望收入")
    for i in range(len(df)):
        ax.text(i, df["income"].iloc[i] + 0.6, f"利润{df['profit'].iloc[i]:.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("金额（元/件成品）")
    ax.set_title("各情形最优策略的期望收入与成本分解")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_profit_decomposition.png"), dpi=300)
    plt.close(fig)


def fig3_analytic_vs_sim():
    """图3：解析 vs 蒙特卡洛 对比（误差%）"""
    df = pd.read_csv(os.path.join(DATA, "p2_verify_best.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    labels = [f"情形{k}" for k in df["情形"]]
    x = range(len(df))
    ax.bar(x, df["解析利润"], w := 0.35, label="解析期望利润", color="#4C72B0")
    ax.bar([i + w for i in x], df["模拟利润(N=20万)"], 0.35, label="蒙特卡洛模拟", color="#55A868")
    for i in range(len(df)):
        ax.text(i, df["解析利润"].iloc[i] + 0.3, f"{df['误差%'].iloc[i]:.2f}%",
                ha="center", fontsize=8.5, color="#C44E52")
    ax.set_xticks([i + w / 2 for i in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("期望利润（元/件成品）")
    ax.set_title("解析模型与蒙特卡洛模拟对比（标注为相对误差）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_analytic_vs_sim.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_strategy_heatmap()
    fig2_profit_decomposition()
    fig3_analytic_vs_sim()
    print("figures:", os.listdir(FIG))
