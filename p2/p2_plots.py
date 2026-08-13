# -*- coding: utf-8 -*-
"""Publication-ready figures for Question 2, generated from saved CSV data."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plot_style import OKABE_ITO, apply_style

apply_style()
BASE = Path(__file__).resolve().parent
DATA, FIG = BASE / "data", BASE / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, VERMILION, SKY, PURPLE = OKABE_ITO[:6]
INK, MUTED, GRID = "#253238", "#6B7479", "#D9DEE1"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, SKY]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=400, facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def polish(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)


def strategy_code(row):
    return "".join(str(int(row[k])) for k in ("x1", "x2", "x3", "x4"))


def fig1_strategy_heatmap():
    df = pd.read_csv(DATA / "p2_all_strategies.csv")
    df["策略"] = df.apply(strategy_code, axis=1)
    piv = df.pivot(index="情形", columns="策略", values="profit")
    cols = sorted(piv.columns, key=lambda s: int(s, 2))
    piv = piv[cols]
    values = piv.to_numpy()
    norm = TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)), [f"{c[:2]}|{c[2:]}" for c in cols])
    ax.set_yticks(np.arange(len(piv)), [f"情形 {k}" for k in piv.index])
    ax.set_xlabel(r"策略编码  $x_1x_2\,|\,x_3x_4$")
    ax.tick_params(axis="both", length=0)
    ax.spines[:].set_visible(False)

    for i in range(values.shape[0]):
        best = int(np.argmax(values[i]))
        ax.add_patch(Rectangle((best - 0.48, i - 0.45), 0.96, 0.90, fill=False, ec=INK, lw=1.15))
        for j, value in enumerate(values[i]):
            rgba = im.cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.1,
                    color="white" if luminance < 0.54 else INK,
                    weight="bold" if j == best else "normal")

    cbar = fig.colorbar(im, ax=ax, pad=0.018, fraction=0.026)
    cbar.set_label("期望利润（元/件）")
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.93, bottom=0.19, top=0.98)
    save(fig, "fig1_strategy_heatmap")


def fig2_profit_decomposition():
    df = pd.read_csv(DATA / "p2_results.csv")
    df["策略码"] = df.apply(strategy_code, axis=1)
    costs = pd.DataFrame({
        "配件": df["cost_part"],
        "装配": df["cost_assembly"],
        "成品检测": df["cost_final_test"],
        "拆解与回收": df["cost_disassemble"] + df["cost_recover"],
        "调换损失": df["cost_switch"],
    })
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    left = np.zeros(len(df))
    for (name, vals), color in zip(costs.items(), PALETTE):
        ax.barh(y, vals, left=left, height=0.58, color=color, label=name)
        left += vals.to_numpy()
    ax.scatter(df["income"], y, marker="D", s=27, color=INK, edgecolor="white", lw=0.6,
               label="期望收入", zorder=4)
    for i, (cost, income, profit) in enumerate(zip(left, df["income"], df["profit"])):
        ax.plot([cost, income], [i, i], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(max(cost, income) + 0.8, i, rf"$\pi$={profit:.2f}", va="center", color=INK, fontsize=7.2)

    labels = [f"情形 {int(k)}   ({code})" for k, code in zip(df["情形"], df["策略码"])]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("金额（元/件成品）")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.7,
              columnspacing=1.2)
    polish(ax, "x")
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.82)
    save(fig, "fig2_profit_decomposition")


def fig3_analytic_vs_sim():
    df = pd.read_csv(DATA / "p2_verify_best.csv")
    y = np.arange(len(df))
    analytic = df["解析利润"].to_numpy()
    simulated = df["模拟利润(N=20万)"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    for i, (a, s) in enumerate(zip(analytic, simulated)):
        ax.plot([a, s], [i, i], color=GRID, lw=2.2, zorder=1)
    ax.scatter(analytic, y, s=31, color=BLUE, label="解析模型", zorder=3)
    ax.scatter(simulated, y, s=35, marker="D", facecolor="white", edgecolor=ORANGE,
               linewidth=1.2, label="蒙特卡洛模拟", zorder=3)
    right = max(analytic.max(), simulated.max())
    for i, err in enumerate(df["误差%"]):
        ax.text(right + 0.45, i, f"误差 {err:.2f}%", va="center", color=MUTED, fontsize=7.2)
    ax.set_yticks(y, [f"情形 {int(k)}" for k in df["情形"]])
    ax.invert_yaxis()
    ax.set_xlabel("期望利润（元/件成品）")
    ax.set_xlim(min(analytic.min(), simulated.min()) - 1.0, right + 3.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    polish(ax, "x")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.84)
    save(fig, "fig3_analytic_vs_sim")


def _segments(sub):
    out = []
    start = 0
    codes = sub["策略编号"].to_numpy()
    xs = sub["参数值"].to_numpy()
    for i in range(1, len(sub) + 1):
        if i == len(sub) or codes[i] != codes[start]:
            out.append((xs[start], xs[i - 1], int(codes[start])))
            start = i
    return out


def fig4_decision_bifurcation():
    df = pd.read_csv(DATA / "p2_sensitivity_results.csv")
    order = ["d1", "d2", "r", "f", "s"]
    symbols = {"d1": r"$d_1$", "d2": r"$d_2$", "r": r"$r$", "f": r"$f$", "s": r"$s$"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75))
    axes = axes.ravel()

    for panel, (ax, param) in enumerate(zip(axes[:5], order)):
        sub = df[df["参数"] == param].sort_values("参数值")
        x, code = sub["参数值"], sub["策略编号"]
        ax.step(x, code, where="post", color=BLUE, lw=1.6)
        ax.scatter(x, code, color=BLUE, s=8, zorder=3)
        for lo, hi, c in _segments(sub):
            mid = (lo + hi) / 2
            bits = f"{c:04b}"
            ax.annotate(bits, xy=(mid, c), xytext=(0, 7), textcoords="offset points",
                        color=VERMILION, ha="center", va="bottom", fontsize=6.7)
        ax.set_xlabel(f"{sub['说明'].iloc[0]}  {symbols[param]}")
        ax.set_ylabel("最优策略编号")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.text(-0.16, 1.02, chr(97 + panel), transform=ax.transAxes, weight="bold")
        polish(ax)

    axes[5].axis("off")
    summary = []
    for param in order:
        segs = _segments(df[df["参数"] == param].sort_values("参数值"))
        if len(segs) == 1:
            summary.append(f"{symbols[param]}：策略 {segs[0][2]:04b} 保持不变")
        else:
            summary.append(f"{symbols[param]}：{segs[0][2]:04b} → {segs[1][2]:04b}")
    axes[5].text(0.02, 0.93, "策略稳定性", color=INK, weight="bold", va="top", fontsize=8.5)
    for i, line in enumerate(summary):
        axes[5].text(0.02, 0.78 - i * 0.14, line, color=INK, va="top", fontsize=7.4)
    axes[5].text(0.02, 0.04, r"策略编号按 $x_1x_2x_3x_4$ 的二进制值编码", color=MUTED, fontsize=6.8)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.98, wspace=0.40, hspace=0.48)
    save(fig, "fig4_decision_bifurcation")


def fig5_rule_trees():
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    df = pd.read_csv(DATA / "p2_rules.csv")
    features = ["p1", "c1", "d1", "p2", "c2", "d2", "p0", "a", "d0", "s", "r", "f"]
    targets = ["x1", "x2", "x3", "x4"]
    names = ["配件 1 检测", "配件 2 检测", "成品检测", "不合格成品拆解"]
    display = [r"$p_1$", r"$c_1$", r"$d_1$", r"$p_2$", r"$c_2$", r"$d_2$",
               r"$p_0$", r"$a$", r"$d_0$", r"$s$", r"$r$", r"$f$"]
    rng = np.random.default_rng(2024)
    idx = rng.permutation(len(df))
    n_train = int(0.8 * len(df))
    train, valid = df.iloc[idx[:n_train]], df.iloc[idx[n_train:]]

    for target, name in zip(targets, names):
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.01, random_state=2024)
        clf.fit(train[features], train[target])
        accuracy = clf.score(valid[features], valid[target])
        fig, ax = plt.subplots(figsize=(9.4, 4.35))
        plot_tree(clf, feature_names=display, class_names=["否", "是"], filled=True,
                  impurity=False, proportion=True, rounded=False, fontsize=7, ax=ax)
        for text in ax.texts:
            patch = text.get_bbox_patch()
            if patch:
                label = text.get_text()
                patch.set_facecolor("#E4F2ED" if "class = 是" in label else "#EEF2F3")
                patch.set_edgecolor("#AAB4B8")
                patch.set_linewidth(0.55)
        ax.set_title(f"{name}决策树   |   留出集准确率 {accuracy:.1%}", loc="left", color=INK, pad=8)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.91)
        save(fig, f"fig5_rule_tree_{target}")


def main():
    fig1_strategy_heatmap()
    fig2_profit_decomposition()
    fig3_analytic_vs_sim()
    fig4_decision_bifurcation()
    fig5_rule_trees()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
