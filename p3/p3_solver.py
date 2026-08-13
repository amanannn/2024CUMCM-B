# -*- coding: utf-8 -*-
"""
2024 国赛 B 题 · 问题3：2道工序8零配件多级生产决策（级联递推 + 2^16 全枚举）

结构：8 零配件 → 3 半成品（1={1,2,3}, 2={4,5,6}, 3={7,8}）→ 成品（半成品1+2+3）
决策变量（16个0-1）：x[1..8]配件检测, y[1..3]半成品检测, z[1..3]半成品拆解, yf成品检测, zf成品拆解

模型（延续问题2口径B，每服务一个客户=最终售出一件合格品）：
- 配件 j 进入装配次品率 e_j = p*(1-x_j)；初装成本 C_j = (1-x_j)c + x_j(c+d)/(1-p)；
  回收件再处理成本 D_j = x_j*(d + c*p)/(1-p)（免采购）
- 半成品 i 实际不合格率 q_i = 1 - Π(1-e_j)*(1-p_h)
  - y_i=1: 检测拦截，不合格拆解(z_i)或丢弃重做，期望成本（首轮配件 ΣC_j，
    后续轮回收件 ΣD_j / 重购 ΣC_j，注意回收件与初装成本之差）
    E_i = [ΣC_j + a_h + d_h + q_i*z_i*(ΣD_j - ΣC_j + f_h)]/(1-q_i)
  - 回收半成品再处理（成品拆解循环中，稳态混合下回收件次品率 = q_i，首轮直接检测）
    R_i = [d_h + q_i*(a_h + z_i*ΣD_j + (1-z_i)*ΣC_j + z_i*f_h)]/(1-q_i)
  - y_i=0: E_i = ΣC_j + a_h，次品率 q_i 传递；R_i = 0
- 成品实际不合格率 q_f = 1 - Π(1-eh_i)*(1-p_f)，eh_i = q_i*(1-y_i)
  - 收入：yf=1 → s；yf=0 → s - q_f*r/(1-q_f)
  - 成本：zf=1（拆解循环）→ ΣE_i + [a_f + yf*d_f + q_f*(f_f + ΣR_i)]/(1-q_f)
          zf=0（报废重做）→ [ΣE_i + a_f + yf*d_f]/(1-q_f)

输出：p3/data/p3_results.csv（最优+基准对比）、p3_top_strategies.csv（Top20）
"""
import itertools
import os
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

P = 0.10                      # 全部零配件/半成品/成品次品率
PARTS = [(2, 1), (8, 1), (12, 2), (2, 1), (8, 1), (12, 2), (8, 1), (12, 2)]  # (c, d)
SEMI_OF = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}   # 半成品 -> 配件索引(0-based)
A_H, D_H, F_H = 8, 4, 6       # 半成品：装配成本 / 检测成本 / 拆解费用
A_F, D_F, F_F, S, R = 8, 6, 10, 200, 40   # 成品：装配 / 检测 / 拆解 / 售价 / 调换损失


def eval_strategy(x, y, z, yf, zf):
    """x,y,z 为长度 8/3/3 的 0/1 列表；返回 (profit, 明细 dict)"""
    C = [c if not xj else (c + d) / (1 - P) for xj, (c, d) in zip(x, PARTS)]
    D = [xj * (d + c * P) / (1 - P) for xj, (c, d) in zip(x, PARTS)]

    E, Qh, Rh = [0.0] * 3, [0.0] * 3, [0.0] * 3
    for i in range(3):
        js = SEMI_OF[i]
        q_i = 1.0 - (1.0 - P) * float(__import__("math").prod(1 - P * (1 - x[j]) for j in js))
        Qh[i] = q_i
        sumC = sum(C[j] for j in js)
        sumD = sum(D[j] for j in js)
        if y[i]:
            E[i] = (sumC + A_H + D_H + q_i * z[i] * (sumD - sumC + F_H)) / (1 - q_i)
            Rh[i] = (D_H + q_i * (A_H + z[i] * sumD + (1 - z[i]) * sumC + z[i] * F_H)) / (1 - q_i)
        else:
            E[i] = sumC + A_H
            Rh[i] = 0.0

    eh = [Qh[i] * (1 - y[i]) for i in range(3)]
    q_f = 1.0 - (1.0 - P) * float(__import__("math").prod(1 - eh[i] for i in range(3)))

    income = S if yf else S - q_f * R / (1 - q_f)
    sumE = sum(E)
    sumR = sum(Rh)
    Bf = A_F + (D_F if yf else 0.0)
    if zf:
        cost = sumE + (Bf + q_f * (F_F + sumR)) / (1 - q_f)
    else:
        cost = (sumE + Bf) / (1 - q_f)

    return income - cost, dict(q_f=q_f, q_h=Qh, income=income, cost=cost,
                               cost_semi=sumE, cost_final=(cost - sumE))


def fmt_strategy(x, y, z, yf, zf):
    return ("".join(map(str, x)) + "|" + "".join(map(str, y)) + "|"
            + "".join(map(str, z)) + f"|{yf}{zf}")


def main():
    best = None
    rows = []
    for bits in itertools.product([0, 1], repeat=16):
        x, y, z = list(bits[0:8]), list(bits[8:11]), list(bits[11:14])
        yf, zf = bits[14], bits[15]
        profit, detail = eval_strategy(x, y, z, yf, zf)
        rows.append((profit, detail, x, y, z, yf, zf))
        if best is None or profit > best[0]:
            best = (profit, detail, x, y, z, yf, zf)

    rows.sort(key=lambda r: r[0], reverse=True)
    profit, detail, x, y, z, yf, zf = best
    print("===== 最优策略（2^16 全枚举）=====")
    print(f"配件检测 x : {''.join(map(str, x))}  (1=检测)")
    print(f"半成品检测y: {''.join(map(str, y))}  (1=检测)")
    print(f"半成品拆解z: {''.join(map(str, z))}  (1=拆解)")
    print(f"成品检测yf : {yf}   成品拆解zf: {zf}")
    print(f"期望利润 = {profit:.4f} 元/件（收入 {detail['income']:.4f} - 成本 {detail['cost']:.4f}）")
    print(f"半成品实际不合格率 q_h = {[round(q, 4) for q in detail['q_h']]}")
    print(f"成品实际不合格率 q_f   = {detail['q_f']:.4f}")

    # 基准策略对比（论文用）
    baselines = {
        "全部检测+全部拆解": ([1] * 8, [1] * 3, [1] * 3, 1, 1),
        "全部检测+不拆解": ([1] * 8, [1] * 3, [0] * 3, 1, 0),
        "全部不检测": ([0] * 8, [0] * 3, [0] * 3, 0, 0),
        "全部不检测+成品拆解": ([0] * 8, [0] * 3, [0] * 3, 0, 1),
    }
    b_rows = []
    for name, (bx, by, bz, byf, bzf) in baselines.items():
        p_, d_ = eval_strategy(bx, by, bz, byf, bzf)
        b_rows.append({"策略": name, "策略码": fmt_strategy(bx, by, bz, byf, bzf),
                       "利润": round(p_, 4), "收入": round(d_["income"], 4),
                       "成本": round(d_["cost"], 4), "q_f": round(d_["q_f"], 4)})
    b_rows.append({"策略": "★最优", "策略码": fmt_strategy(x, y, z, yf, zf),
                   "利润": round(profit, 4), "收入": round(detail["income"], 4),
                   "成本": round(detail["cost"], 4), "q_f": round(detail["q_f"], 4)})
    df_b = pd.DataFrame(b_rows)
    print("\n===== 基准策略对比 =====")
    print(df_b.to_string(index=False))
    df_b.to_csv(os.path.join(OUT, "p3_results.csv"), index=False, encoding="utf-8-sig")

    top = pd.DataFrame([{
        "策略码": fmt_strategy(r[2], r[3], r[4], r[5], r[6]),
        "利润": round(r[0], 4), "收入": round(r[1]["income"], 4), "成本": round(r[1]["cost"], 4),
        "q_f": round(r[1]["q_f"], 4), "q_h1": round(r[1]["q_h"][0], 4),
        "q_h2": round(r[1]["q_h"][1], 4), "q_h3": round(r[1]["q_h"][2], 4),
    } for r in rows[:20]])
    top.to_csv(os.path.join(OUT, "p3_top_strategies.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[输出] {OUT}/p3_results.csv, p3_top_strategies.csv (Top20)")


if __name__ == "__main__":
    main()
