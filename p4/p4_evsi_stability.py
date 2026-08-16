# -*- coding: utf-8 -*-
"""
问题4 扩展：EVSI 结果稳定性检验（建模手建议——蒙特卡洛仿真测稳定性）

用 10 个不同随机种子独立重复 EVSI 估计（每种子 S_WORLD=200 世界模拟），
统计各情形 n* 与 EVSI 的均值 ± 标准差：
- 若跨种子结果一致（n* 稳定、EVSI 标准差小），证明结论不依赖随机性；
- 报告重复运行的一致性，作为模型可靠性的蒙特卡洛证据。

为控制时间只跑代表性情形（1、2、6）与批量敏感性核心档位。

输出：p4/data/p4_evsi_stability.csv、p4/figures/fig6_evsi_stability.png
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

import p4.p4_evsi as evsi  # noqa: E402
from p2.p2_solver import CASES as P2_CASES  # noqa: E402

N_SEEDS = 10
CASES_STAB = [1, 2, 6]
N_GRID = [0, 20, 40, 60, 80, 100]   # 缩减 n 网格控制时间


def main():
    rows = []
    for k_ in CASES_STAB:
        case = P2_CASES[k_]
        p_set = [case["p1"], case["p2"], case["p0"]]
        priors = [evsi.prior_params(p) for p in p_set]
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 137 + k_)
            vals = evsi.V_all_n(case, priors, N_GRID, rng)
            n_star = max(N_GRID, key=lambda n: vals[n])
            evsi_v = vals[n_star] - vals[0]
            rows.append({"情形": k_, "种子": seed, "n*": n_star,
                         "EVSI": round(evsi_v, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p4_evsi_stability.csv"), index=False, encoding="utf-8-sig")

    print("===== EVSI 稳定性（10 种子独立重复）=====")
    summary = df.groupby("情形").agg(
        n_star_均值=("n*", "mean"), n_star_众数=("n*", lambda s: s.mode().iloc[0]),
        n_star_范围=("n*", lambda s: f"[{s.min()},{s.max()}]"),
        EVSI均值=("EVSI", "mean"), EVSI标准差=("EVSI", "std"),
        EVSI范围=("EVSI", lambda s: f"[{s.min():.3f},{s.max():.3f}]"))
    print(summary.to_string())
    summary.to_csv(os.path.join(DATA, "p4_evsi_stability_summary.csv"), encoding="utf-8-sig")

    # 稳定性结论
    for k_, row in summary.iterrows():
        stable = row["n_star_范围"] in ("[0,0]", "[0,20]") or row["EVSI标准差"] < 0.05
        print(f"情形{k_}: n* 范围 {row['n_star_范围']}, EVSI std {row['EVSI标准差']:.3f} → "
              f"{'稳定 ✓' if stable else '需注意'}")

    # ---- 图：各种子 EVSI 散点 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import OKABE_ITO, apply_style
    apply_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, k_ in enumerate(CASES_STAB):
        sub = df[df["情形"] == k_]
        ax.plot(sub["种子"], sub["EVSI"], "-o", ms=5, lw=1.5,
                color=OKABE_ITO[i], label=f"情形{k_}")
        ax.axhline(sub["EVSI"].mean(), color=OKABE_ITO[i], ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("随机种子编号")
    ax.set_ylabel("EVSI（元/件成品）")
    ax.set_title("EVSI 估计的跨种子稳定性（每种子 200 世界模拟）")
    ax.legend()
    ax.grid(axis="y", color="#D9DEE1", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_evsi_stability.png"), dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[输出] {DATA}/p4_evsi_stability.csv, p4_evsi_stability_summary.csv, {FIG}/fig6_evsi_stability.png")


if __name__ == "__main__":
    main()
