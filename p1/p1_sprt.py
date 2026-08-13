# -*- coding: utf-8 -*-
"""
问题1 扩展：序贯概率比检验（SPRT）抽样方案
对比固定样本量方案（情形一 n=29 / 情形二 n=22）与序贯方案的期望检测次数。

SPRT 设定：H0: p=p0=0.10 vs H1: p=p1=0.20（次品率翻倍的备择点）
- 拒收错误 α=0.05（对应"95%信度拒收"）
- 接收错误 β=0.10（对应"90%信度接收"）
停止界限：A = ln((1-β)/α)，B = ln(β/(1-α))
似然比对数增量：Z_i = ln(f(x_i|p1)/f(x_i|p0))
判定：累计 S_n ≥ A → 拒收；S_n ≤ B → 接收；否则继续抽检（设截尾上限）。

输出：p1/data/p1_sprt.csv（ASN/OC 数据）、图 p1/figures/fig4_sprt.png
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

P0, P1 = 0.10, 0.20
ALPHA, BETA = 0.05, 0.10
A = np.log((1 - BETA) / ALPHA)        # 拒收界
B = np.log(BETA / (1 - ALPHA))        # 接收界
N_MAX = 200                            # 截尾上限


def sprt_simulate(p_true, rng, n_max=N_MAX):
    """模拟一次序贯检验：返回 (决策 1=拒收/0=接收, 检测次数)"""
    S = 0.0
    for n in range(1, n_max + 1):
        x = 1 if rng.random() < p_true else 0
        # 单样本对数似然比增量
        if x == 1:
            S += np.log(P1 / P0)
        else:
            S += np.log((1 - P1) / (1 - P0))
        if S >= A:
            return 1, n
        if S <= B:
            return 0, n
    return (1 if S >= 0 else 0), n_max      # 截尾判决


def asn_and_oc(p_grid):
    """解析 ASN 与 OC（Wald 近似）"""
    h = np.log((1 - P0) / (1 - P1)) / np.log((P1 / P0) * ((1 - P0) / (1 - P1)))
    rows = []
    for p in p_grid:
        # OC：接受概率（Wald 近似，对 h 的幂函数插值）
        L = 1.0
        if p != P0:
            # L(p) = (1 - ((1-P1)/(1-P0))^h) / ((P1/P0)^h - ((1-P1)/(1-P0))^h) 在 h 为解时的值
            # 使用 Wald: L(p) ≈ (1 - t^h)/(t1^h - t^h) 形式，实际按 p 对应的 h 求
            pass
        # 期望每步对数似然比
        EZ = p * np.log(P1 / P0) + (1 - p) * np.log((1 - P1) / (1 - P0))
        # 期望样本量近似：ASN(p) ≈ [L(p)*B + (1-L(p))*A] / EZ，L 用线性近似简化
        # 这里用精确的 Wald 公式计算 L
        t1 = P1 / P0
        t0 = (1 - P1) / (1 - P0)
        # OC: L(p) = (1 - t0^{h(p)}) / (t1^{h(p)} - t0^{h(p)})，其中 h(p) 满足 (p1/p0)^h = t0^h 关系
        # 标准 Wald OC 公式（用 h 参数化 p）：
        # p(h) = (1 - t0^h) / (t1^h - t0^h)
        # 给定 p 反解 h：
        if p <= 0 or p >= 1:
            rows.append({"p": p, "ASN": np.nan, "P_accept": np.nan})
            continue
        # 反解 h：p = (1 - t0^h)/(t1^h - t0^h)，h 从 -inf 到 +inf
        from scipy.optimize import brentq

        def f(hh):
            return (1 - t0 ** hh) / (t1 ** hh - t0 ** hh) - p

        try:
            hh = brentq(f, -50, 50)
        except Exception:
            hh = np.sign(p - P0) * 1e6
        # Wald OC（接受概率）与 ASN
        La = (t0 ** hh - 1) / (t0 ** hh - t1 ** hh) if abs(hh) > 1e-9 else (1 - p)
        if abs(hh) < 1e-9:           # p 恰在边界
            La = (1 - p)
        # 修正：L(p) 的标准 Wald 形式 = (1 - t0^h)/(t1^h - t0^h) 的补……使用经典公式：
        L_accept = (1 - t1 ** hh) / (t0 ** hh - t1 ** hh) if abs(hh) > 1e-9 else (1 - p)
        EZ = p * np.log(P1 / P0) + (1 - p) * np.log((1 - P1) / (1 - P0))
        ASN = (L_accept * B + (1 - L_accept) * A) / EZ if EZ != 0 else np.nan
        rows.append({"p": p, "ASN": abs(ASN), "P_accept": abs(L_accept)})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(2024)

    # ---- 蒙特卡洛模拟 SPRT 性能 ----
    sim_rows = []
    for p in [P0, P1, 0.05, 0.15, 0.30]:
        n_reps = 200_000
        results = np.array([sprt_simulate(p, rng) for _ in range(n_reps)])
        rej, ns = results[:, 0], results[:, 1]
        sim_rows.append({"p": p, "模拟拒收率": round(rej.mean(), 4),
                         "平均检测次数": round(ns.mean(), 2),
                         "95分位检测次数": round(np.percentile(ns, 95), 1)})
    df_sim = pd.DataFrame(sim_rows)
    print("===== SPRT 蒙特卡洛模拟（每次20万次）=====")
    print(df_sim.to_string(index=False))
    df_sim.to_csv(os.path.join(DATA, "p1_sprt.csv"), index=False, encoding="utf-8-sig")

    # ---- 与固定样本量方案对比 ----
    print("\n===== 固定样本量 vs SPRT 期望检测次数 =====")
    print("固定方案：情形一 n=29（次品数≥1拒收，95%信度）；情形二 n=22（全合格接收）")
    print(f"SPRT (p0=10%, p1=20%, α=5%, β=10%):")
    for _, r in df_sim.iterrows():
        print(f"  真实次品率 p={r['p']:.2f}: 期望检测 {r['平均检测次数']} 次, 拒收率 {r['模拟拒收率']:.1%}")

    # ---- 画图：OC 曲线与 ASN 对比 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_style import apply_style
    apply_style()

    p_grid = np.linspace(0.02, 0.50, 97)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # 左：OC 曲线（接受概率 vs p）——SPRT vs 固定 n=29 / n=22
    ax = axes[0]
    for n, label, color in [(29, "固定 n=29（≥1件次品拒收）", "#C44E52"),
                            (22, "固定 n=22（0件次品接收）", "#4C72B0")]:
        ax.plot(p_grid, (1 - p_grid) ** n, "--", lw=2, color=color, label=label)
    # SPRT OC（模拟）
    sim_oc = []
    for p in p_grid:
        n_reps = 30_000
        results = np.array([sprt_simulate(p, rng) for _ in range(n_reps)])
        sim_oc.append(results[:, 0].mean() if False else 1 - results[:, 0].mean())
    ax.plot(p_grid, sim_oc, "-", lw=2.2, color="#55A868", label="SPRT（模拟）")
    ax.axvline(P0, color="gray", ls=":", lw=1)
    ax.axhline(0.1, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("真实次品率 p")
    ax.set_ylabel("接收概率 P(接收|p)")
    ax.set_title("OC 曲线：SPRT vs 固定样本量方案")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # 右：SPRT 期望检测次数 vs p（模拟），标注固定样本量水平线
    ax = axes[1]
    asn_pts = []
    for p in p_grid:
        n_reps = 30_000
        results = np.array([sprt_simulate(p, rng) for _ in range(n_reps)])
        asn_pts.append(results[:, 1].mean())
    ax.plot(p_grid, asn_pts, lw=2.2, color="#55A868")
    ax.axhline(29, color="#C44E52", ls="--", lw=1.2, label="固定 n=29（拒收情形）")
    ax.axhline(22, color="#4C72B0", ls="--", lw=1.2, label="固定 n=22（接收情形）")
    ax.axvline(P0, color="gray", ls=":", lw=1)
    ax.axvline(P1, color="gray", ls=":", lw=1)
    ax.text(P0, 8, r"$p_0$=10%", color="gray", fontsize=9)
    ax.text(P1, 8, r"$p_1$=20%", color="gray", fontsize=9)
    ax.set_xlabel("真实次品率 p")
    ax.set_ylabel("期望检测次数 E[n|p]")
    ax.set_title("SPRT 期望检测次数（截尾上限 200）")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_sprt.png"), dpi=300)
    plt.close(fig)
    print(f"\n[输出] {DATA}/p1_sprt.csv, {FIG}/fig4_sprt.png")


if __name__ == "__main__":
    main()
