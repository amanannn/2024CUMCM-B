# -*- coding: utf-8 -*-
"""
统一科研绘图风格（Nature 顶刊风格）——所有绘图脚本共用。

用法：from plot_style import apply_style; apply_style()

风格要素：去顶/右边框、刻度向内、小字号（8pt 系）、Arial+中文 fallback、
无网格、Okabe-Ito 色盲友好配色、300dpi、tight bbox。
"""
import matplotlib as mpl

# Okabe-Ito 色盲友好调色板（Nature 常用）
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#56B4E9", "#CC79A7", "#F0E442", "#000000"]


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
