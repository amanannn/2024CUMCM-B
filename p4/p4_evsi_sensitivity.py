# -*- coding: utf-8 -*-
"""
问题4 EVSI 敏感性：什么条件下抽样才值得？

扫描两个维度：
- 批量 B ∈ {100, 1000, 10000}（抽样成本每批一次，摊入 B 件成品）
- 抽检成本 d ∈ {0.5, 2.0, 5.0}（元/件）

结论预期：批量越大，抽样成本摊得越薄，抽样信息价值越划算 → n* 增大。
这回答了"问题1 为什么值得设计抽样方案"：批量足够大时抽检经济上合理。

输出：p4/data/p4_evsi_sensitivity.csv
"""
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import p4.p4_evsi as evsi  # noqa: E402
from p2.p2_solver import CASES as P2_CASES  # noqa: E402

SENS = {"B": [100, 1000, 10000], "d": [0.5, 2.0, 5.0]}
CASES_SENS = [1, 6]


def main():
    rng = np.random.default_rng(7)
    rows = []
    for k_ in CASES_SENS:
        case = P2_CASES[k_]
        p_set = [case["p1"], case["p2"], case["p0"]]
        priors = [evsi.prior_params(p) for p in p_set]
        for B in SENS["B"]:
            evsi.BATCH = B
            for d in SENS["d"]:
                evsi.D_DET = d
                vals = evsi.V_all_n(case, priors, evsi.N_SCAN, rng)
                n_star = max(evsi.N_SCAN, key=lambda n: vals[n])
                rows.append({"情形": k_, "批量B": B, "检测成本d": d,
                             "最优n*": n_star, "V(n*)": round(vals[n_star], 2),
                             "V(0)": round(vals[0], 2), "EVSI": round(vals[n_star] - vals[0], 2)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p4_evsi_sensitivity.csv"), index=False, encoding="utf-8-sig")
    print("===== EVSI 敏感性：最优抽样量 n*（情形1/6，行=批量，列=检测成本）=====")
    for k_ in CASES_SENS:
        sub = df[df["情形"] == k_]
        print(f"\n--- 情形{k_} ---")
        print("最优 n*:")
        print(sub.pivot(index="批量B", columns="检测成本d", values="最优n*").to_string())
        print("EVSI（元/件成品）:")
        print(sub.pivot(index="批量B", columns="检测成本d", values="EVSI").to_string())
    print(f"\n[输出] {OUT}/p4_evsi_sensitivity.csv")


if __name__ == "__main__":
    main()
