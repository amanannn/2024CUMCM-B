# -*- coding: utf-8 -*-
"""
问题2 蒙特卡洛模拟交叉验证（与 p2_solver.py 解析结果对比）

模拟口径与解析模型（口径B）完全一致（流水线稳态混合假设）：
以"服务一个客户"（最终售出一件合格品）为单位，逐件模拟完整流程——
配件采购/检测循环 → 装配 → 成品检测/市场 → 退回 → 拆解循环（回收件按 x_i 策略再处理）
或 报废后重新采购再生产（x4=0），直至产出一件合格品。

对每个情形的 16 个策略均做模拟，与解析期望利润对比，验证解析模型正确性。
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

CASES = {
    1: dict(p1=0.10, c1=4, d1=2, p2=0.10, c2=18, d2=3, p0=0.10, a=6, d0=3, s=56, r=6, f=5),
    2: dict(p1=0.20, c1=4, d1=2, p2=0.20, c2=18, d2=3, p0=0.20, a=6, d0=3, s=56, r=6, f=5),
    3: dict(p1=0.10, c1=4, d1=2, p2=0.10, c2=18, d2=3, p0=0.10, a=6, d0=3, s=56, r=30, f=5),
    4: dict(p1=0.20, c1=4, d1=1, p2=0.20, c2=18, d2=1, p0=0.20, a=6, d0=2, s=56, r=30, f=5),
    5: dict(p1=0.10, c1=4, d1=8, p2=0.20, c2=18, d2=1, p0=0.10, a=6, d0=2, s=56, r=10, f=5),
    6: dict(p1=0.05, c1=4, d1=2, p2=0.05, c2=18, d2=3, p0=0.05, a=6, d0=3, s=56, r=10, f=40),
}

MAX_LOOP = 200  # 拆解循环上限（防御性；q<1 时实际远到不了）


def simulate_one(case: dict, x1: int, x2: int, x3: int, x4: int, rng: np.random.Generator):
    """模拟"服务一个客户"（最终售出一件合格品）的净利润，与解析口径B一致。

    x4=1：不合格成品拆解，回收件按 x_i 策略再处理（免采购）；
    x4=0：不合格成品报废，企业重新采购配件再生产，直到产出一件合格品。
    """
    p1, c1, d1 = case["p1"], case["c1"], case["d1"]
    p2, c2, d2 = case["p2"], case["c2"], case["d2"]
    p0, a, d0, s, r, f = case["p0"], case["a"], case["d0"], case["s"], case["r"], case["f"]

    revenue, cost = 0.0, 0.0
    loop = 0
    first = True

    while True:
        loop += 1
        if loop > MAX_LOOP:            # 理论不会触发（q<1），防御性兜底
            return revenue - cost - 1e9

        # ---- 配件环节 ----
        if first or not x4:            # 初装 或 报废后重新采购
            while True:
                cost += c1
                if x1:
                    cost += d1
                    if rng.random() < p1:
                        continue
                    part1_def = False
                    break
                part1_def = rng.random() < p1
                break
            while True:
                cost += c2
                if x2:
                    cost += d2
                    if rng.random() < p2:
                        continue
                    part2_def = False
                    break
                part2_def = rng.random() < p2
                break
            first = False
        else:                          # 拆解回收件：免采购，按 x_i 策略再处理
            if x1:
                while True:
                    cost += d1
                    if rng.random() < p1:
                        cost += c1
                    else:
                        part1_def = False
                        break
            else:
                part1_def = rng.random() < p1
            if x2:
                while True:
                    cost += d2
                    if rng.random() < p2:
                        cost += c2
                    else:
                        part2_def = False
                        break
            else:
                part2_def = rng.random() < p2

        # ---- 装配 → 成品环节 ----
        cost += a
        prod_def = part1_def or part2_def or (rng.random() < p0)

        if x3:                          # 成品检测
            cost += d0
            if not prod_def:            # 合格卖出，客户成交
                revenue += s
                return revenue - cost
        else:                           # 不检测，直接进市场
            revenue += s
            if not prod_def:            # 合格，客户成交
                return revenue - cost
            revenue -= s                # 不合格被退回：退款
            cost += r                   # 调换损失

        # 不合格成品：拆解 或 报废（报废则下一轮重新采购配件）
        if x4:
            cost += f                   # 拆解


def simulate_strategy(case: dict, x1: int, x2: int, x3: int, x4: int,
                      n: int, seed: int = 2024) -> float:
    rng = np.random.default_rng(seed)
    profits = np.array([simulate_one(case, x1, x2, x3, x4, rng) for _ in range(n)])
    return profits.mean()


def main():
    from p2_solver import eval_strategy  # noqa: F401

    N_ALL = 40_000      # 16 策略对比时的模拟量
    N_BEST = 200_000    # 最优策略加大模拟量精确验证

    rows = []
    for k, case in CASES.items():
        strategies = list(itertools.product([0, 1], repeat=4))
        sim_profits = {x: simulate_strategy(case, *x, N_ALL, seed=2024 + k * 10) for x in strategies}
        analytic = {x: eval_strategy(case, *x)["profit"] for x in strategies}
        best_analytic = max(strategies, key=lambda x: analytic[x])
        best_sim = max(strategies, key=lambda x: sim_profits[x])

        for x in strategies:
            rows.append({
                "情形": k,
                "策略(x1,x2,x3,x4)": f"({x[0]},{x[1]},{x[2]},{x[3]})",
                "解析利润": round(analytic[x], 4),
                "模拟利润": round(sim_profits[x], 4),
                "误差%": round(100 * abs(sim_profits[x] - analytic[x]) / max(abs(analytic[x]), 1e-9), 3),
                "解析最优": "★" if x == best_analytic else "",
                "模拟最优": "★" if x == best_sim else "",
            })

    df = pd.DataFrame(rows).sort_values(["情形", "解析利润"], ascending=[True, False])
    df.to_csv(os.path.join(OUT, "p2_simulation.csv"), index=False, encoding="utf-8-sig")

    # 最优策略加大模拟量
    best_rows = []
    for k, case in CASES.items():
        from p2_solver import eval_strategy as ev
        best = max(itertools.product([0, 1], repeat=4), key=lambda x: ev(case, *x)["profit"])
        sim = simulate_strategy(case, *best, N_BEST, seed=999 + k)
        ana = ev(case, *best)["profit"]
        best_rows.append({"情形": k, "策略": f"({best[0]},{best[1]},{best[2]},{best[3]})",
                          "解析利润": round(ana, 4), "模拟利润(N=20万)": round(sim, 4),
                          "误差%": round(100 * abs(sim - ana) / abs(ana), 4)})
    dfb = pd.DataFrame(best_rows)
    print("===== 最优策略：解析 vs 蒙特卡洛（N=20万）=====")
    print(dfb.to_string(index=False))
    print("\n===== 16策略全对比（N=4万，误差统计）=====")
    print(f"最大误差%: {df['误差%'].max()}, 平均误差%: {df['误差%'].mean():.3f}")
    ok = df.groupby("情形").apply(
        lambda g: g[g["解析最优"] == "★"]["模拟最优"].iloc[0] == "★", include_groups=False)
    print("各情形解析最优==模拟最优:", dict(ok))
    dfb.to_csv(os.path.join(OUT, "p2_verify_best.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
