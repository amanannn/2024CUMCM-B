# -*- coding: utf-8 -*-
"""
问题3 扩展：决策规则提炼（决策树，2^16 枚举数据）

在表2 参数邻域采样 M 组参数（次品率、检测费扰动），每组全枚举 65536 策略，
以"配件 j 是否被检测"为标签训练可解释决策树，提炼企业可套用的规则。

特征（配件 j 视角）：c_j 单价、d_j 检测费、p_j 次品率、
同半成品配件数、同半成品配件平均单价、全局 f_f/s。

输出：p3/data/p3_rules_tree.txt、p3/figures/fig6_rule_tree_part_*.png（选取代表性配件）
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

from p3.p3_sensitivity import eval_p3  # noqa: E402

PARTS = [(2, 1), (8, 1), (12, 2), (2, 1), (8, 1), (12, 2), (8, 1), (12, 2)]
SEMI_OF = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
M_SAMPLE = 150
SEED = 7


def main():
    from sklearn.tree import DecisionTreeClassifier, export_text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(M_SAMPLE):
        P = rng.uniform(0.05, 0.20)
        over = {"P": P, "S": rng.uniform(150, 260), "F_F": rng.uniform(2, 30),
                "D_H": rng.uniform(1, 8), "F_H": rng.uniform(1, 15)}
        best = max(itertools.product([0, 1], repeat=16),
                   key=lambda b: eval_p3(list(b[0:8]), list(b[8:11]), list(b[11:14]),
                                         b[14], b[15], over))
        x = best[0:8]
        for j in range(8):
            semi_idx = [i for i, js in SEMI_OF.items() if j in js][0]
            semi_avg_c = np.mean([PARTS[k][0] for k in SEMI_OF[semi_idx]])
            rows.append({
                "配件j": j, "c_j": PARTS[j][0], "d_j": PARTS[j][1], "p_j": P,
                "同半成品配件数": len(SEMI_OF[semi_idx]), "半成品平均单价": semi_avg_c,
                "S": over["S"], "F_F": over["F_F"], "检测y": x[j],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p3_rules.csv"), index=False, encoding="utf-8-sig")

    FEAT = ["c_j", "d_j", "p_j", "同半成品配件数", "半成品平均单价", "S", "F_F"]
    idx = rng.permutation(len(df))
    n_tr = int(0.8 * len(df))
    tr, va = df.iloc[idx[:n_tr]], df.iloc[idx[n_tr:]]

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.02, random_state=SEED)
    clf.fit(tr[FEAT], tr["检测y"])
    acc = clf.score(va[FEAT], va["检测y"])
    imp = dict(zip(FEAT, clf.feature_importances_))

    print("===== 问题3 规则提炼：配件 j 是否应检测（决策树）=====")
    print(f"留出集准确率 {acc:.4f}")
    print("特征重要性: " + ", ".join(f"{k}={v:.3f}" for k, v in
                                      sorted(imp.items(), key=lambda kv: -kv[1])[:5]))
    tree_txt = export_text(clf, feature_names=FEAT)
    print(tree_txt)
    with open(os.path.join(DATA, "p3_rules_tree.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"准确率 {acc:.4f}\n" + tree_txt)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    plot_tree(clf, feature_names=FEAT, class_names=["不检测", "检测"], filled=True,
              fontsize=8, ax=ax, impurity=False, proportion=True, rounded=True)
    ax.set_title(f"配件检测决策规则树（留出准确率 {acc:.4f}）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_rule_tree_part.png"), dpi=300)
    plt.close(fig)

    # 每配件被检测的比例（洞察表）
    rate = df.groupby("配件j")["检测y"].mean().round(3)
    print("\n各配件在采样中最优解中被检测的比例:")
    print(rate.to_string())
    print(f"\n[输出] {DATA}/p3_rules.csv, p3_rules_tree.txt, {FIG}/fig6_rule_tree_part.png")


if __name__ == "__main__":
    main()
