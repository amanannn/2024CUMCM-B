# -*- coding: utf-8 -*-
"""
问题3 蒙特卡洛模拟交叉验证（与 p3_solver.py 解析递推对比，口径B一致）

模拟"服务一个客户"（最终售出一件合格品）：
半成品生产（配件采购/回收→装配→检测→拆解或丢弃重做）→ 成品装配（3件半成品）
→ 成品检测/市场 → 退回 → 拆解回收半成品（mode=recover）或报废重新采购（mode=purchase）。
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

P = 0.10
PARTS = [(2, 1), (8, 1), (12, 2), (2, 1), (8, 1), (12, 2), (8, 1), (12, 2)]
SEMI_OF = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
A_H, D_H, F_H = 8, 4, 6
A_F, D_F, F_F, S, R = 8, 6, 10, 200, 40
MAX_LOOP = 200


def acquire(j, mode, x, rng):
    """配件环节：返回 (成本, 是否次品)。mode='purchase'首件采购；'recover'首件免采购。"""
    c, d = PARTS[j]
    cost = 0.0
    while True:
        if mode == "purchase" or cost > 0:
            cost += c
        if x[j]:
            cost += d
            if rng.random() < P:
                mode = "purchase"          # 丢弃后补货必须采购
                continue
            return cost, False
        return cost, rng.random() < P


def produce_semi(i, mode, x, y, z, q_i, rng):
    """获得一件可装配的半成品 i：返回 (成本, 输出是否次品)。

    mode: 'purchase' 全新（配件采购）| 'recover' 回收配件（免采购首件）
          | 'semi' 回收半成品（免配件免装配，直接检测/传递，稳态混合下次品率 q_i）
    y_i=1 时输出必合格（检测拦截）；y_i=0 时输出含次品状态。
    """
    cost = 0.0
    first = True
    while True:
        if first and mode == "semi":
            if y[i]:
                cost += D_H
                if rng.random() < q_i:          # 回收半成品不合格（稳态重采样）
                    if z[i]:                    # 拆解回收配件，重新组装
                        cost += F_H
                        mode = "recover"
                    else:                       # 丢弃，重新采购组装
                        mode = "purchase"
                    first = False
                    continue
                return cost, False
            return cost, rng.random() < q_i     # 不检测，直接装配（状态重采样）
        # 配件环节（purchase 采购 / recover 回收件）
        part_defs = []
        for j in SEMI_OF[i]:
            c_, d_ = acquire(j, mode, x, rng)
            cost += c_
            part_defs.append(d_)
        cost += A_H
        semi_def = any(part_defs) or rng.random() < P
        if y[i]:
            cost += D_H
            if not semi_def:
                return cost, False
            if z[i]:
                cost += F_H
                mode = "recover"
            else:
                mode = "purchase"
        else:
            return cost, semi_def
        first = False


def simulate_one(x, y, z, yf, zf, rng):
    revenue, cost = 0.0, 0.0
    first, loop = True, 0
    # 半成品实际不合格率（与解析定义一致，供回收半成品稳态重采样）
    qs = []
    for i in range(3):
        q_ok = 1.0 - P
        for j in SEMI_OF[i]:
            q_ok *= 1.0 - P * (1 - x[j])
        qs.append(1.0 - q_ok)
    while True:
        loop += 1
        if loop > MAX_LOOP:
            return revenue - cost - 1e9
        mode = "purchase" if (first or not zf) else "semi"
        first = False
        # 生产/再处理 3 件半成品
        semi_defs = []
        for i in range(3):
            c_, d_ = produce_semi(i, mode, x, y, z, qs[i], rng)
            cost += c_
            semi_defs.append(d_)
        # 装配成品
        cost += A_F
        prod_def = any(semi_defs) or rng.random() < P
        if yf:                          # 成品检测
            cost += D_F
            if not prod_def:            # 合格卖出
                revenue += S
                return revenue - cost
        else:                           # 直接进市场
            revenue += S
            if not prod_def:
                return revenue - cost
            revenue -= S                # 退回退款
            cost += R                   # 调换损失
        # 不合格成品：拆解（回收半成品，下轮 recover）或报废（下轮重新采购）
        if zf:
            cost += F_F


def main():
    # 最优策略（从 solver 重新枚举，保证一致）
    from p3_solver import eval_strategy as ev

    best = max(itertools.product([0, 1], repeat=16),
               key=lambda b: ev(list(b[0:8]), list(b[8:11]), list(b[11:14]), b[14], b[15])[0])
    bx, by, bz, byf, bzf = (list(best[0:8]), list(best[8:11]), list(best[11:14]), best[14], best[15])

    strategies = {
        "★最优": (bx, by, bz, byf, bzf),
        "全部检测+全部拆解": ([1] * 8, [1] * 3, [1] * 3, 1, 1),
        "全部检测+不拆解": ([1] * 8, [1] * 3, [0] * 3, 1, 0),
        "全部不检测": ([0] * 8, [0] * 3, [0] * 3, 0, 0),
        "全部不检测+成品拆解": ([0] * 8, [0] * 3, [0] * 3, 0, 1),
    }
    rows = []
    for name, (x, y, z, yf, zf) in strategies.items():
        n = 200_000 if name == "★最优" else 50_000
        rng = np.random.default_rng(2024)
        profits = np.array([simulate_one(x, y, z, yf, zf, rng) for _ in range(n)])
        ana = ev(x, y, z, yf, zf)[0]
        rows.append({"策略": name, "解析利润": round(ana, 4), "模拟利润": round(profits.mean(), 4),
                     "误差%": round(100 * abs(profits.mean() - ana) / abs(ana), 3)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, "p3_simulation.csv"), index=False, encoding="utf-8-sig")
    print(f"[输出] {OUT}/p3_simulation.csv")


if __name__ == "__main__":
    main()
