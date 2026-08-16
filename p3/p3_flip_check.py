# -*- coding: utf-8 -*-
"""
问题3 扩展：最优策略单比特翻转检验（建模手建议）

对最优策略 11111111|000|000|11 的 16 个决策变量逐一翻转（0→1 或 1→0），
重算期望利润，记录下降幅度：
- 全部翻转利润均不增 → 验证最优性（局部扰动稳定，配合全枚举=全局最优）；
- 翻转损失大的变量 = 决策的关键变量（边际价值高）；
- 利润不变的翻转 → 冗余决策（如 z1z2z3 在 y=000 下不触发），印证"等价编码"。

输出：p3/data/p3_flip_check.csv、p3/figures/fig6_flip_check.png
"""
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

from p3.p3_solver import eval_strategy  # noqa: E402

# 最优策略：11111111|000|000|11
X0, Y0, Z0, YF0, ZF0 = [1] * 8, [0, 0, 0], [0, 0, 0], 1, 1

VAR_NAMES = [f"x{j+1}" for j in range(8)] + \
            [f"y{i+1}" for i in range(3)] + \
            [f"z{i+1}" for i in range(3)] + \
            ["yf", "zf"]
ORIGINAL = [1] * 8 + [0] * 3 + [0] * 3 + [1, 1]


def profit_of(strategy):
    x, y, z, yf, zf = strategy[:8], strategy[8:11], strategy[11:14], strategy[14], strategy[15]
    return eval_strategy(x, y, z, yf, zf)[0]


def main():
    base_profit = profit_of(ORIGINAL)
    rows = []
    for i, name in enumerate(VAR_NAMES):
        flipped = list(ORIGINAL)
        flipped[i] = 1 - flipped[i]
        p_flip = profit_of(flipped)
        rows.append({"变量": name, "原值": ORIGINAL[i], "翻转后值": flipped[i],
                     "翻转后利润": round(p_flip, 4),
                     "利润变化": round(p_flip - base_profit, 4),
                     "损失(元/件)": round(base_profit - p_flip, 4),
                     "边际价值": "关键" if p_flip < base_profit - 0.01 else
                     ("冗余" if abs(p_flip - base_profit) < 1e-9 else "微小影响")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p3_flip_check.csv"), index=False, encoding="utf-8-sig")

    print(f"最优策略利润（基线）: {base_profit:.4f} 元/件")
    print(df.to_string(index=False))
    p_flip = df["翻转后利润"].to_numpy()
    n_worse = int((p_flip < base_profit - 1e-3).sum())
    n_same = int((np.abs(p_flip - base_profit) < 1e-3).sum())
    n_better = int((p_flip > base_profit + 1e-3).sum())
    print(f"\n[最优性检验] 翻转变差 {n_worse} 个 | 不变 {n_same} 个（冗余/等价） | 变好 {n_better} 个")
    print(f"[结论] {'全部不优于基线，单比特扰动稳定 ✓（配合全枚举=全局最优）' if n_better == 0 else '存在变好翻转——最优解有误！'}")

    # ---- 图：各变量翻转的利润损失（横向条形，按损失排序） ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import OKABE_ITO, apply_style
    apply_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = df.sort_values("利润变化").index
    vals = df.loc[order, "利润变化"]
    colors = [OKABE_ITO[0] if v < 0 else OKABE_ITO[5] for v in vals]
    bars = ax.barh(df.loc[order, "变量"], vals, color=colors, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + (0.05 if v >= 0 else -0.05), b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color="#253238")
    ax.axvline(0, color="#253238", lw=0.8)
    ax.set_xlabel("翻转后利润变化（元/件成品）")
    ax.set_ylabel("被翻转的决策变量")
    ax.set_title("单比特翻转检验：每个决策变量的边际利润贡献")
    ax.set_xlim(min(vals) - 1.2, max(vals) + 0.8)
    ax.grid(axis="x", color="#D9DEE1", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_flip_check.png"), dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[输出] {DATA}/p3_flip_check.csv, {FIG}/fig6_flip_check.png")


if __name__ == "__main__":
    main()
