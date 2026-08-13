# -*- coding: utf-8 -*-
"""
问题3 扩展：决策拐点敏感性分析（单变量扫描，2^16 全枚举）

扫描参数（表2 基线）：s 售价、f_f 成品拆解费、d_h 半成品检测费、f_h 半成品拆解费。
每参数值全枚举 65536 策略，输出最优策略的"画像"（Σx 检测配件数、Σy 检测半成品数、
Σz 拆解半成品数、yf、zf），观察决策骨架随参数的变化与分岔点。

输出：p3/data/p3_sensitivity_results.csv、p3/figures/fig5_sensitivity.png
"""
import itertools
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

PARTS = [(2, 1), (8, 1), (12, 2), (2, 1), (8, 1), (12, 2), (8, 1), (12, 2)]

# 扫描配置: (参数名, 起点, 终点, 步长, 说明, 修改函数)
SCANS = []


def scan_s(v):
    return {}, v                    # 修改 eval 后处理的参数


# 用参数覆盖方式：p3_solver 的 eval_strategy 使用模块级常量，这里复制一份参数化实现
def eval_p3(x, y, z, yf, zf, overrides):
    """基于 p3_solver.eval_strategy 的参数化版本（只改 S/A_F/D_F/F_F/P）"""
    P = overrides.get("P", 0.10)
    A_F = overrides.get("A_F", 8)
    D_F = overrides.get("D_F", 6)
    F_F = overrides.get("F_F", 10)
    S = overrides.get("S", 200)
    R = overrides.get("R", 40)
    A_H = overrides.get("A_H", 8)
    D_H = overrides.get("D_H", 4)
    F_H = overrides.get("F_H", 6)

    C = [c if not xj else (c + d) / (1 - P) for xj, (c, d) in zip(x, PARTS)]
    D = [xj * (d + c * P) / (1 - P) for xj, (c, d) in zip(x, PARTS)]
    SEMI_OF = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
    E, Qh, Rh = [0.0] * 3, [0.0] * 3, [0.0] * 3
    for i in range(3):
        js = SEMI_OF[i]
        q_ok = 1.0 - P
        for j in js:
            q_ok *= 1.0 - P * (1 - x[j])
        q_i = 1.0 - q_ok
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
    q_ok = 1.0 - P
    for i in range(3):
        q_ok *= 1.0 - eh[i]
    q_f = 1.0 - q_ok
    income = S if yf else S - q_f * R / (1 - q_f)
    sumE, sumR = sum(E), sum(Rh)
    Bf = A_F + (D_F if yf else 0.0)
    if zf:
        cost = sumE + (Bf + q_f * (F_F + sumR)) / (1 - q_f)
    else:
        cost = (sumE + Bf) / (1 - q_f)
    return income - cost


def best_p3(overrides):
    best = max(itertools.product([0, 1], repeat=16),
               key=lambda b: eval_p3(list(b[0:8]), list(b[8:11]), list(b[11:14]), b[14], b[15], overrides))
    return best


def main():
    scans = [
        ("S", 80, 320, 20, "市场售价"),
        ("F_F", 1, 40, 3, "成品拆解费"),
        ("D_H", 1, 12, 1, "半成品检测费(统一)"),
        ("F_H", 1, 25, 2, "半成品拆解费(统一)"),
    ]
    rows = []
    for param, lo, hi, step, note in scans:
        for v in np.round(np.arange(lo, hi + step / 2, step), 3):
            best = best_p3({param: v})
            x, y, z, yf, zf = best[0:8], best[8:11], best[11:14], best[14], best[15]
            rows.append({"参数": param, "参数值": v, "说明": note,
                         "Σx(检测配件数)": sum(x), "Σy(检测半成品数)": sum(y),
                         "Σz(拆解半成品数)": sum(z), "yf(成品检测)": yf, "zf(成品拆解)": zf,
                         "策略码": f"{''.join(map(str, x))}|{''.join(map(str, y))}"
                                   f"|{''.join(map(str, z))}|{yf}{zf}"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p3_sensitivity_results.csv"), index=False, encoding="utf-8-sig")

    print("===== 问题3 决策分岔点（单变量扫描）=====")
    for param, _, _, _, note in scans:
        sub = df[df["参数"] == param]
        prev = None
        print(f"\n{note} {param}:")
        for _, r in sub.iterrows():
            key = (r["Σx(检测配件数)"], r["Σy(检测半成品数)"], r["zf(成品拆解)"])
            if key != prev:
                print(f"  {param}={r['参数值']:>6}: Σx={r['Σx(检测配件数)']} Σy={r['Σy(检测半成品数)']} "
                      f"yf={r['yf(成品检测)']} zf={r['zf(成品拆解)']}  [{r['策略码']}]")
                prev = key

    # ---- 图：策略画像随参数变化 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes = axes.ravel()
    for ax, (param, _, _, _, note) in zip(axes, scans):
        sub = df[df["参数"] == param]
        ax.plot(sub["参数值"], sub["Σx(检测配件数)"], "-o", ms=4, label="检测配件数 Σx", color="#4C72B0")
        ax.plot(sub["参数值"], sub["Σy(检测半成品数)"], "-s", ms=4, label="检测半成品数 Σy", color="#55A868")
        ax.set_xlabel(note + f" {param}")
        ax.set_ylabel("数量")
        ax.set_ylim(-0.3, 8.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(sub["参数值"], sub["zf(成品拆解)"], "--d", ms=4, label="成品拆解 zf", color="#C44E52")
        ax2.set_ylabel("zf", color="#C44E52")
        ax2.set_ylim(-0.1, 1.3)
        ax2.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{note}扫描", fontsize=11)
    fig.suptitle("问题3 最优决策画像随参数的变化（2^16 全枚举）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_sensitivity.png"), dpi=300)
    plt.close(fig)
    print(f"\n[输出] {DATA}/p3_sensitivity_results.csv, {FIG}/fig5_sensitivity.png")


if __name__ == "__main__":
    main()
