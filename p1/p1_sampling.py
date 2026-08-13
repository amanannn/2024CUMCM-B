# -*- coding: utf-8 -*-
"""
2024 年高教社杯全国大学生数学建模竞赛 B 题 · 问题 1
抽样检测方案设计（判定可靠度读法，已与队伍确认方案 A）

方案框架
--------
抽检 n 件零配件，次品数 X ~ B(n, p)（二项分布，精确计算）。
判定规则统一为：出现 ≥1 件次品 → 拒收；0 件次品 → 接收。

- 情形一（95% 信度下认定次品率超标 → 拒收）：
  判定"超标"的可靠度 ≥ 95%，即 P(X≥1 | p=p0) = 1-(1-p0)^n ≥ 0.95
  最小样本量 n1 = ceil(ln0.05/ln(1-p0))，p0=10% 时 n1=29。
- 情形二（90% 信度下认定次品率不超标 → 接收）：
  "不超标"判定出错概率 ≤ 10%，即 P(X=0 | p=p0) = (1-p0)^n ≤ 0.10
  最小样本量 n2 = ceil(ln0.10/ln(1-p0))，p0=10% 时 n2=22。

用 scipy.stats.binom 精确计算并验证最小性，结果存入 p1/data/。
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

P0 = 0.10        # 标称次品率
ALPHA_1 = 0.05   # 情形一：拒收判定可靠度 1-α1 = 95%
ALPHA_2 = 0.10   # 情形二：接收判定出错概率 α2 = 10%（可靠度 90%）

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)


def min_n_scenario1(p0: float, alpha: float = ALPHA_1) -> int:
    """情形一：最小 n，使 P(X≥1 | p0) ≥ 1-alpha（精确二项分布）"""
    n = 1
    while True:
        if 1.0 - binom.pmf(0, n, p0) >= 1.0 - alpha:
            return n
        n += 1


def min_n_scenario2(p0: float, alpha: float = ALPHA_2) -> int:
    """情形二：最小 n，使 P(X=0 | p0) ≤ alpha（精确二项分布）"""
    n = 1
    while True:
        if binom.pmf(0, n, p0) <= alpha:
            return n
        n += 1


def main():
    n1 = min_n_scenario1(P0)
    n2 = min_n_scenario2(P0)
    n1_formula = math.ceil(math.log(ALPHA_1) / math.log(1 - P0))
    n2_formula = math.ceil(math.log(ALPHA_2) / math.log(1 - P0))
    assert n1 == n1_formula and n2 == n2_formula, "精确搜索与解析公式不一致！"

    # ---- 核心结果表 ----
    rows = [
        {
            "情形": "一（95%信度拒收）",
            "样本量n": n1,
            "判定规则": "次品数≥1 拒收",
            "P(拒收|p=0.10)": round(1.0 - binom.pmf(0, n1, P0), 4),
            "P(接收|p=0.10)": round(binom.pmf(0, n1, P0), 4),
            "判定可靠度": round(1.0 - binom.pmf(0, n1, P0), 4),
            "最小性验证": f"n={n1-1} 时可靠度 {1.0-binom.pmf(0, n1-1, P0):.4f} < 95%，不可行",
            "解析公式": f"ceil(ln0.05/ln0.90) = {n1_formula}",
        },
        {
            "情形": "二（90%信度接收）",
            "样本量n": n2,
            "判定规则": "0件次品 接收",
            "P(拒收|p=0.10)": round(1.0 - binom.pmf(0, n2, P0), 4),
            "P(接收|p=0.10)": round(binom.pmf(0, n2, P0), 4),
            "判定可靠度": round(1.0 - binom.pmf(0, n2, P0), 4),
            "最小性验证": f"n={n2-1} 时出错概率 {binom.pmf(0, n2-1, P0):.4f} > 10%，不可行",
            "解析公式": f"ceil(ln0.10/ln0.90) = {n2_formula}",
        },
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, "p1_results.csv"), index=False, encoding="utf-8-sig")

    # ---- OC 曲线数据：P(接收|p) = P(X=0|p) = (1-p)^n ----
    p_grid = np.linspace(0.0, 0.50, 501)
    oc = pd.DataFrame({
        "p": p_grid,
        "n29_P接收": (1 - p_grid) ** n1,
        "n22_P接收": (1 - p_grid) ** n2,
    })
    oc.to_csv(os.path.join(OUT, "p1_oc_curve.csv"), index=False, encoding="utf-8-sig")

    # ---- 敏感性：标称值 p0 变化时的最小样本量（通用公式验证）----
    p0_grid = np.round(np.arange(0.01, 0.2005, 0.005), 3)
    sens = pd.DataFrame({
        "p0": p0_grid,
        "n1_拒收95%": [min_n_scenario1(p0) for p0 in p0_grid],
        "n2_接收90%": [min_n_scenario2(p0) for p0 in p0_grid],
    })
    sens.to_csv(os.path.join(OUT, "p1_sensitivity.csv"), index=False, encoding="utf-8-sig")

    # ---- 可靠度随 n 变化（画图用）----
    n_grid = np.arange(1, 41)
    rel = pd.DataFrame({
        "n": n_grid,
        "拒收判定可靠度_1_0.9^n": 1 - 0.9 ** n_grid,
        "接收判定出错概率_0.9^n": 0.9 ** n_grid,
    })
    rel.to_csv(os.path.join(OUT, "p1_reliability_vs_n.csv"), index=False, encoding="utf-8-sig")

    print(f"\n标称值 p0={P0}：情形一 n1={n1}，情形二 n2={n2}（解析公式一致）")
    print(f"[输出] {OUT}/ p1_results.csv, p1_oc_curve.csv, p1_sensitivity.csv, p1_reliability_vs_n.csv")


if __name__ == "__main__":
    main()
