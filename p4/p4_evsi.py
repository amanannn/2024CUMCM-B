# -*- coding: utf-8 -*-
"""
问题4 重做：抽样-决策一体化（EVSI 框架，方法层修正）

此前做法（已被指出缺陷）：给定 n=22 → 置信区间 → 端点重解 = "代数字不是方法"。
正确框架：把抽样建进决策链——

1. 先验：p ~ Beta(a0, b0)（均值=表值 p̂，等价样本量 n_prior，反映对供应商的信任）
2. 抽样：观测 k | p ~ Binomial(n, p)（n 为决策变量）
3. 后验：p | k ~ Beta(a0+k, b0+n−k)
4. 观测驱动的决策：x*(k) = argmax_x E_{p|k}[π(p,x)]   （对观测 k 的决策表）
5. 事前价值：V(n) = E_{p,k}[ max_x E_{p|k}[π(p,x)] ] − 抽样成本
6. 最优抽样量：n* = argmax V(n)；抽样信息价值 EVSI = V(n*) − V(0)

P2 每情形有 3 个次品率参数（p1,p2,p0），各自独立抽样观测 (k1,k2,k0)，
后验期望利润用蒙特卡洛估计（numpy 向量化，16 策略 × M 后验样本批量计算）。

输出：p4/data/p4_evsi_curve.csv, p4_evsi_results.csv, p4_evsi_decision_table.csv
       p4/figures/fig5_evsi.png
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join(os.path.dirname(__file__), "data")
FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2.p2_solver import CASES as P2_CASES  # noqa: E402

D_DET = 2.0           # 每件抽检成本（元/件）
N_PRIOR = 30          # 先验等价样本量（信任强度）
BATCH = 1000          # 批量大小（抽样每批一次，成本摊入 B 件成品）
N_SCAN = list(range(0, 101, 10))  # 抽样量扫描
S_WORLD = 300         # 外层"世界"模拟次数
M_POST = 500          # 内层后验采样次数
SEED = 2024


def prior_params(p_hat, n_prior=N_PRIOR):
    """先验 Beta(a0,b0)：均值=p̂，信息量≈n_prior 次观测"""
    return max(p_hat * n_prior, 1e-6), max((1 - p_hat) * n_prior, 1e-6)


def profit_vec(case, p1, p2, p0, x1, x2, x3, x4):
    """向量化期望利润（口径B，与 p2_solver 一致）。p1,p2,p0 可为数组。"""
    c1, d1 = case["c1"], case["d1"]
    c2, d2 = case["c2"], case["d2"]
    a, d0, s, r, f = case["a"], case["d0"], case["s"], case["r"], case["f"]
    p1, p2, p0 = np.asarray(p1, float), np.asarray(p2, float), np.asarray(p0, float)
    e1 = np.where(x1 == 0, p1, 0.0)
    e2 = np.where(x2 == 0, p2, 0.0)
    q = 1.0 - (1.0 - e1) * (1.0 - e2) * (1.0 - p0)
    Cpart = np.where(x1 == 0, c1, (c1 + d1) / (1.0 - p1)) \
        + np.where(x2 == 0, c2, (c2 + d2) / (1.0 - p2))
    D_recover = x1 * (d1 + c1 * p1) / (1.0 - p1) + x2 * (d2 + c2 * p2) / (1.0 - p2)
    B = a + (d0 if x3 else 0.0)
    if x4:
        income = np.where(x3 == 0, s - q * r / (1.0 - q), s)
        cost = Cpart + (B + q * (f + D_recover)) / (1.0 - q)
    else:
        income = np.where(x3 == 0, s - q * r / (1.0 - q), s)
        cost = (Cpart + B) / (1.0 - q)
    return income - cost


def posterior_expect_all(case, k_obs, n, priors, m=M_POST, rng=None):
    """对 16 个策略计算 E_{p|k}[π]（同一组后验样本配对比较，消除策略间采样噪声）"""
    ps = [beta_dist.rvs(a0 + k, b0 + n - k, size=m, random_state=rng)
          for (a0, b0), k in zip(priors, k_obs)]
    best_x, best_v = None, -1e18
    for x in itertools.product([0, 1], repeat=4):
        v = float(np.mean(profit_vec(case, ps[0], ps[1], ps[2], *x)))
        if v > best_v:
            best_x, best_v = x, v
    return best_x, best_v


def V_all_n(case, priors, n_list, rng):
    """事前价值 V(n)（Common Random Numbers 版，降低世界间噪声）。

    同一组世界随机数（p_true 与反演均匀数 U）评估所有 n：
    观测 k(n) = Binom 分位数 Q(n, p_true, U)，同一世界下 k 随 n 单调相关，
    V(n) 曲线平滑，EVSI = V(n*) − V(0) 估计精确。
    """
    from scipy.stats import binom as binom_dist
    totals = {n: 0.0 for n in n_list}
    for _ in range(S_WORLD):
        p_true = [beta_dist.rvs(a0, b0, random_state=rng) for a0, b0 in priors]
        U = [rng.random() for _ in priors]
        for n in n_list:
            if n == 0:
                k_obs = (0, 0, 0)
            else:
                k_obs = tuple(int(binom_dist.ppf(u, n, p)) for u, p in zip(U, p_true))
            totals[n] += posterior_expect_all(case, k_obs, n, priors, rng=rng)[1]
    # 抽样成本每批一次 3n·d，摊入 B 件成品
    return {n: totals[n] / S_WORLD - 3 * n * D_DET / BATCH for n in n_list}


def main():
    rng = np.random.default_rng(SEED)
    res_rows, curve_rows, table_rows = [], [], []

    for k_ in P2_CASES:
        case = P2_CASES[k_]
        p_set = [case["p1"], case["p2"], case["p0"]]
        priors = [prior_params(p) for p in p_set]
        vals = V_all_n(case, priors, N_SCAN, rng)
        for n in N_SCAN:
            curve_rows.append({"情形": k_, "n": n, "V(n)": round(vals[n], 3)})
        n_star = max(N_SCAN, key=lambda n: vals[n])
        v0 = vals[0]
        res_rows.append({"情形": k_, "最优抽样量n*": n_star,
                         "V(n*)": round(vals[n_star], 3), "V(0)": round(v0, 3),
                         "EVSI": round(vals[n_star] - v0, 3)})

        # 决策表：n* 下 观测 k → 最优策略（k 为三参数统一观测次数的简化展示：0,1,2,3）
        print(f"\n===== 情形{k_}：n*={n_star}，EVSI={vals[n_star]-v0:.3f} 元/件 =====")
        for k_obs in [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (2, 1, 0), (3, 1, 1)]:
            x, v = posterior_expect_all(case, k_obs, n_star, priors, rng=rng)
            table_rows.append({"情形": k_, "观测(k1,k2,k0)": str(k_obs),
                               "最优策略": f"({x[0]},{x[1]},{x[2]},{x[3]})",
                               "后验期望利润": round(v, 3)})
            print(f"  观测 {k_obs} → 策略 ({x[0]},{x[1]},{x[2]},{x[3]})  后验利润 {v:.3f}")

    df_res = pd.DataFrame(res_rows)
    df_curve = pd.DataFrame(curve_rows)
    df_table = pd.DataFrame(table_rows)
    print("\n===== EVSI 汇总 =====")
    print(df_res.to_string(index=False))
    df_res.to_csv(os.path.join(OUT, "p4_evsi_results.csv"), index=False, encoding="utf-8-sig")
    df_curve.to_csv(os.path.join(OUT, "p4_evsi_curve.csv"), index=False, encoding="utf-8-sig")
    df_table.to_csv(os.path.join(OUT, "p4_evsi_decision_table.csv"), index=False, encoding="utf-8-sig")

    # 图：V(n) 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_style import apply_style
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    axes = axes.ravel()
    for ax, k_ in zip(axes, P2_CASES):
        sub = df_curve[df_curve["情形"] == k_]
        ax.plot(sub["n"], sub["V(n)"], "-o", ms=4, lw=2, color="#4C72B0")
        row = df_res[df_res["情形"] == k_].iloc[0]
        ax.axvline(row["最优抽样量n*"], color="#C44E52", ls="--", lw=1.2)
        ax.annotate(f"n*={int(row['最优抽样量n*'])}\nEVSI={row['EVSI']:.2f}",
                    xy=(row["最优抽样量n*"], row["V(n*)"]), fontsize=9,
                    xytext=(8, -30), textcoords="offset points", color="#C44E52")
        ax.set_xlabel("抽样量 n（每参数）")
        ax.set_ylabel("事前期望价值 V(n)")
        ax.set_title(f"情形{k_}", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("EVSI 框架：最优抽样量由抽样信息价值决定（先验 Beta 均值=表值（等价样本 30），检测费 2 元/件，批量 1000 件/批）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_evsi.png"), dpi=300)
    plt.close(fig)
    print(f"\n[输出] {OUT}/p4_evsi_results.csv, p4_evsi_curve.csv, p4_evsi_decision_table.csv")
    print(f"       {FIG}/fig5_evsi.png")


if __name__ == "__main__":
    main()
