# -*- coding: utf-8 -*-
"""
问题2 扩展：决策规则提炼（机器学习方法落地）

用参数空间采样生成"参数→最优策略"数据集，训练可解释的决策树，
提炼企业可直接套用的阈值规则（如"配件1应检测 ⟺ 检测费低于某个临界值"）。

流程：
1. 在 6 情形参数附近均匀采样 N 组参数（含向题目 6 情形外扩展的域）；
2. 每组全枚举 16 策略 → 最优策略标签（4 个 0-1 决策）；
3. 训练决策树（max_depth 限制保证可解释），留出集验证准确率；
4. 输出规则文本（树路径转 if-then 规则）+ 特征重要性。

输出：p2/data/p2_rules.csv（数据集）、p2_rules_tree.txt（规则）、p2/figures/fig5_rules_tree.png（树图）
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

from p2.p2_solver import eval_strategy  # noqa: E402

N_SAMPLE = 8000
SEED = 2024

# 参数采样域（围绕 6 情形扩展）
DOMAIN = dict(
    p1=(0.02, 0.35), c1=(1.0, 15.0), d1=(0.3, 10.0),
    p2=(0.02, 0.35), c2=(5.0, 40.0), d2=(0.3, 10.0),
    p0=(0.02, 0.35), a=(2.0, 15.0), d0=(0.5, 8.0),
    s=(30.0, 120.0), r=(2.0, 50.0), f=(1.0, 45.0),
)
FEATURES = list(DOMAIN.keys())
TARGETS = ["x1", "x2", "x3", "x4"]


def sample_params(rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in DOMAIN.items()}


def main():
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_SAMPLE):
        case = sample_params(rng)
        best = max(itertools.product([0, 1], repeat=4),
                   key=lambda x: eval_strategy(case, *x)["profit"])
        rows.append({**{k: round(case[k], 4) for k in FEATURES},
                     **{t: v for t, v in zip(TARGETS, best)}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "p2_rules.csv"), index=False, encoding="utf-8-sig")

    # 训练/验证划分
    idx = rng.permutation(len(df))
    n_tr = int(0.8 * len(df))
    tr, va = df.iloc[idx[:n_tr]], df.iloc[idx[n_tr:]]
    X_tr, X_va = tr[FEATURES], va[FEATURES]

    print("===== 决策规则提炼（决策树，max_depth=4，留出验证）=====")
    rules_text = []
    for t, name in zip(TARGETS, ["配件1检测", "配件2检测", "成品检测", "不合格成品拆解"]):
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.01, random_state=SEED)
        clf.fit(X_tr, tr[t])
        acc = clf.score(X_va, va[t])
        imp = dict(zip(FEATURES, clf.feature_importances_))
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:4]
        print(f"\n[{name}] 留出集准确率 {acc:.4f}  特征重要性: " +
              ", ".join(f"{k}={v:.3f}" for k, v in top))
        rules_text.append(f"===== {name} (准确率 {acc:.4f}) =====\n"
                          + export_text(clf, feature_names=FEATURES))
        print(export_text(clf, feature_names=FEATURES))
        # 树图
        fig, ax = plt.subplots(figsize=(13, 6))
        plot_tree(clf, feature_names=FEATURES, class_names=["否", "是"], filled=True,
                  fontsize=8, ax=ax, impurity=False, proportion=True, rounded=True)
        ax.set_title(f"{name} 决策规则树（准确率 {acc:.4f}）")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig5_rule_tree_{t}.png"), dpi=300)
        plt.close(fig)

    with open(os.path.join(DATA, "p2_rules_tree.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rules_text))
    print(f"\n[输出] {DATA}/p2_rules.csv, p2_rules_tree.txt, {FIG}/fig5_rule_tree_*.png")


if __name__ == "__main__":
    main()
