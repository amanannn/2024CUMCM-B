# -*- coding: utf-8 -*-
"""
2024 国赛 B 题 · 问题2：生产决策（解析期望模型 + 16 策略全枚举）

模型口径（与队伍确认）
--------------------
决策变量：x1(配件1检测) x2(配件2检测) x3(成品检测) x4(不合格成品拆解)，均为 0/1。
以"最终产出一件成品"为单位（企业持续补货直到凑齐合格件），期望净利润最大化。

关键假设：
1. 检测无误差；
2. 题目给的成品次品率 p0 是"正品零配件装配后的次品率"（附录说明1），
   故成品实际不合格率 q = 1-(1-e1)(1-e2)(1-p0)，e_i = 0（检测）或 p_i（不检测）；
3. 拆解回收零配件按题目"重复步骤(1)"执行检测决策：
   - x_i=1：回收件免费，但需再检测（次品丢弃后补货 c_i），每轮期望 (d_i+c_i*p_i)/(1-p_i)；
   - x_i=0：回收件直接装配，次品率 p_i（流水线稳态混合假设，各轮独立同分布）；
4. x3=0 时流入市场的不合格品被退回：退款 + 调换损失 r，退回品按 x4 处理。

期望公式（统一口径：每服务一个客户=最终售出一件合格品，与题目"调换"要求一致）：
- x4=1（拆解循环）：期望轮数 1/(1-q)，回收件免采购按策略再处理；
  成本 = C_part + [B + q*(f+D)]/(1-q)，收入：x3=1 → s；x3=0 → s - q*r/(1-q)
- x4=0（报废）：不合格成品报废后必须重新采购配件再生产，直到合格品产出，
  期望生产次数 1/(1-q)；成本 = (C_part + B)/(1-q)，收入同上式
- 初装配件成本 C_part = Σ_i [(1-x_i)*c_i + x_i*(c_i+d_i)/(1-p_i)]
- 每轮循环基础成本 B = a + x3*d0；拆解触发概率 q，拆解+回收件成本 f + D，
  D = Σ_i x_i*(d_i+c_i*p_i)/(1-p_i)

输出：p2/data/p2_results.csv（最优策略+成本分解）、p2_all_strategies.csv（16策略全表）
"""
import itertools
import os
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# 情形参数: p1,c1,d1, p2,c2,d2, p0, a, d0, s, r, f
CASES = {
    1: dict(p1=0.10, c1=4, d1=2, p2=0.10, c2=18, d2=3, p0=0.10, a=6, d0=3, s=56, r=6, f=5),
    2: dict(p1=0.20, c1=4, d1=2, p2=0.20, c2=18, d2=3, p0=0.20, a=6, d0=3, s=56, r=6, f=5),
    3: dict(p1=0.10, c1=4, d1=2, p2=0.10, c2=18, d2=3, p0=0.10, a=6, d0=3, s=56, r=30, f=5),
    4: dict(p1=0.20, c1=4, d1=1, p2=0.20, c2=18, d2=1, p0=0.20, a=6, d0=2, s=56, r=30, f=5),
    5: dict(p1=0.10, c1=4, d1=8, p2=0.20, c2=18, d2=1, p0=0.10, a=6, d0=2, s=56, r=10, f=5),
    6: dict(p1=0.05, c1=4, d1=2, p2=0.05, c2=18, d2=3, p0=0.05, a=6, d0=3, s=56, r=10, f=40),
}


def eval_strategy(case: dict, x1: int, x2: int, x3: int, x4: int):
    """返回利润及各项成本分解"""
    p1, c1, d1 = case["p1"], case["c1"], case["d1"]
    p2, c2, d2 = case["p2"], case["c2"], case["d2"]
    p0, a, d0, s, r, f = case["p0"], case["a"], case["d0"], case["s"], case["r"], case["f"]

    e1 = p1 if x1 == 0 else 0.0          # 进入装配的配件1次品率
    e2 = p2 if x2 == 0 else 0.0
    q = 1.0 - (1.0 - e1) * (1.0 - e2) * (1.0 - p0)   # 成品实际不合格率

    # 初装配件环节（最终成品口径，含重购）：期望采购+检测
    C_part = sum(
        (c if x == 0 else (c + d) / (1.0 - p))          # noqa: B023
        for p, c, d, x in [(p1, c1, d1, x1), (p2, c2, d2, x2)]
    )
    # 拆解循环中回收件环节（x_i=1 时再检测+补货）
    D = sum(
        x * (d + c * p) / (1.0 - p)
        for p, c, d, x in [(p1, c1, d1, x1), (p2, c2, d2, x2)]
    )
    B = a + (d0 if x3 else 0.0)          # 每轮基础成本：装配+成品检测

    if x4 == 1:                          # 拆解循环：期望轮数 1/(1-q)
        income = s if x3 else s - q * r / (1.0 - q)
        cost_part = C_part
        cost_cycle = (B + q * (f + D)) / (1.0 - q)     # 装配+成品检测+拆解+回收件
        cost_assembly = a / (1.0 - q)
        cost_final_test = d0 / (1.0 - q) if x3 else 0.0
        cost_disassemble = q * f / (1.0 - q)
        cost_recover = q * D / (1.0 - q)
        cost_switch = q * r / (1.0 - q) if x3 == 0 else 0.0
    else:                                # 不拆解：报废后重新采购再生产，直到合格品（期望 1/(1-q) 次）
        income = s - q * r / (1.0 - q) if x3 == 0 else s
        cost_part = C_part / (1.0 - q)
        cost_cycle = B / (1.0 - q)
        cost_assembly = a / (1.0 - q)
        cost_final_test = d0 / (1.0 - q) if x3 else 0.0
        cost_disassemble = 0.0
        cost_recover = 0.0
        cost_switch = q * r / (1.0 - q) if x3 == 0 else 0.0

    profit = income - cost_part - cost_cycle
    return dict(
        x1=x1, x2=x2, x3=x3, x4=x4, q=q,
        income=round(income, 4),
        cost_part=round(cost_part, 4),
        cost_assembly=round(cost_assembly, 4),
        cost_final_test=round(cost_final_test, 4),
        cost_disassemble=round(cost_disassemble, 4),
        cost_recover=round(cost_recover, 4),
        cost_switch=round(cost_switch, 4),
        profit=round(profit, 4),
    )


def main():
    all_rows, best_rows = [], []
    for k, case in CASES.items():
        results = [eval_strategy(case, *x) for x in itertools.product([0, 1], repeat=4)]
        best = max(results, key=lambda r: r["profit"])
        for r in results:
            r["情形"] = k
            r["是否最优"] = "★" if r is best else ""
            all_rows.append(r)
        best_rows.append({"情形": k, **{kk: vv for kk, vv in best.items() if kk != "是否最优"}})

    df_all = pd.DataFrame(all_rows).sort_values(["情形", "profit"], ascending=[True, False])
    df_best = pd.DataFrame(best_rows)
    df_best["策略(x1,x2,x3,x4)"] = df_best.apply(
        lambda r: f"({r.x1},{r.x2},{r.x3},{r.x4})", axis=1)
    df_best = df_best[["情形", "x1", "x2", "x3", "x4", "策略(x1,x2,x3,x4)", "q", "income", "cost_part",
                       "cost_assembly", "cost_final_test", "cost_disassemble",
                       "cost_recover", "cost_switch", "profit"]]

    print("===== 各情形最优策略 =====")
    print(df_best.to_string(index=False))
    df_best.to_csv(os.path.join(OUT, "p2_results.csv"), index=False, encoding="utf-8-sig")
    df_all.to_csv(os.path.join(OUT, "p2_all_strategies.csv"), index=False, encoding="utf-8-sig")

    # 模型自检（业务直觉；基于 all_rows 找最优，避免字符串索引）
    def best_of(k):
        r = [x for x in all_rows if x["情形"] == k]
        return max(r, key=lambda x: x["profit"])

    b3, b5 = best_of(3), best_of(5)
    assert b3["x3"] == 1, f"自检失败：情形3 应检测成品(调换损失30)，实际 x3={b3['x3']}"
    assert b5["x1"] == 0, f"自检失败：情形5 配件1检测费8过贵应不检测，实际 x1={b5['x1']}"
    print("\n[自检通过] 情形3 成品必检测（调换损失30大）；情形5 配件1不检测（检测费8过贵）")
    print("[说明] 情形6 拆解费40过贵 → 不拆解(x4=0)；其余情形拆解费5元，回收配件划算 → 均拆解")
    print(f"[输出] {OUT}/p2_results.csv, p2_all_strategies.csv")


if __name__ == "__main__":
    main()
