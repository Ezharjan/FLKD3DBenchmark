"""Presentation-grade figure suite for the FLKD3D benchmark.

Reads aggregated results via :mod:`flkd_results` (no training, no retraining) and
renders a set of PNG + PDF figures under ``outputs/figures/``.

    python visualizations/make_figures.py \
        --output_root outputs --out outputs/figures

Figures (skipped gracefully when the underlying data is absent):
    flkd_heatmap_<ds>        FL x KD combined accuracy heatmap + marginals
    flkd_synergy_<ds>        combined minus best-single-method synergy heatmap
    fl_ranking_<ds>          FL strategy accuracy bars vs centralised
    fl_convergence_<ds>      FL convergence curves with +/-std bands
    fl_comm_pareto_<ds>      FL accuracy vs communication cost
    kd_ranking_<ds>          KD method accuracy bars vs teacher
    kd_compression_<ds>      accuracy retained + teacher->student compression
    efficiency_pareto_<ds>   accuracy vs size & latency, all families
    best_combos_<ds>         top FL+KD leaderboard
    seed_robustness_<ds>     per-family accuracy spread across seeds
    cross_dataset_summary    best per family across datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from flkd_results import (load_all, FL_ORDER, KD_ORDER, FAMILY_ORDER,
                          FAMILY_LABEL, flkd_pivot, synergy_matrix, best_combos,
                          export)
import viz_style as vs


# --------------------------------------------------------------------------- #
# 1. Combined FL x KD heatmap (with marginal means)                             #
# --------------------------------------------------------------------------- #
def fig_flkd_heatmap(res, outdir, ds):
    piv = flkd_pivot(res, ds, "acc_mean")
    if piv is None:
        return
    vals = piv.values.astype(float)
    nrow, ncol = vals.shape
    fig = plt.figure(figsize=(1.18 * ncol + 4.5, 0.82 * nrow + 4.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[ncol, 1.6, 0.4],
                          height_ratios=[1.4, nrow], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])

    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    im = ax.imshow(vals, aspect="auto", cmap=vs.ACC_CMAP, vmin=vmin, vmax=vmax)
    
    # 1. Increased fontsize for x and y tick labels
    ax.set_xticks(np.arange(ncol)) 
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=22)
    ax.set_yticks(np.arange(nrow)) 
    ax.set_yticklabels(piv.index, fontsize=22)
    
    # 4. Added labelpad=2 to bring labels closer, and increased fontsize
    ax.set_xlabel("KD strategy", fontsize=25, labelpad=2) 
    ax.set_ylabel("FL teacher", fontsize=25, labelpad=2)
    
    rng = (vmax - vmin) or 1.0
    
    # 1. Bumped up the min and max limits for the dynamic font size calculation
    fs = max(12, min(35, int(280 / max(ncol, nrow))))
    for i in range(nrow):
        for j in range(ncol):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs,
                        color="white" if (v - vmin) / rng < 0.55 else "#1A1A1A",
                        fontweight="bold")
                
    bi, bj = np.unravel_index(np.nanargmax(vals), vals.shape)
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                           edgecolor="#00E5FF", linewidth=3.5))
    ax.grid(False)

    col_mean = np.nanmean(vals, axis=0)
    row_mean = np.nanmean(vals, axis=1)
    ax_top.bar(np.arange(ncol), col_mean, color=[vs.kd_color(c) for c in piv.columns],
               edgecolor="white", linewidth=0.8)
    ax_top.set_ylim(max(0, np.nanmin(col_mean) - 5), 100)
    
    # 1. Increased fontsize for the top "mean(%)" label and its ticks
    ax_top.set_ylabel("mean\n(%)", fontsize=18)
    ax_top.tick_params(labelbottom=False, labelsize=21) 
    vs.despine(ax_top, ("left",))
    ax_top.grid(axis="y", alpha=0.25)
    for j, v in enumerate(col_mean):
        # 1. Increased fontsize for the digits floating above the top bars
        ax_top.text(j, v + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=16, fontweight="bold")

    ax_right.barh(np.arange(nrow), row_mean, color=[vs.fl_color(r) for r in piv.index],
                  edgecolor="white", linewidth=0.8)
    ax_right.set_xlim(max(0, np.nanmin(row_mean) - 5), 100)
    ax_right.invert_yaxis()
    
    # 1. Increased fontsize for the right "mean(%)" label and its ticks
    ax_right.set_xlabel("mean (%)", fontsize=22)
    ax_right.tick_params(labelleft=False, labelsize=18)
    vs.despine(ax_right, ("bottom",))
    ax_right.grid(axis="x", alpha=0.25)

    cb = fig.colorbar(im, cax=cax)
    # 1. Increased fontsize for colorbar label and the tick numbers on the colorbar
    cb.set_label("Top-1 accuracy (%)", fontsize=18)
    cb.ax.tick_params(labelsize=18)
    
    # 2. Changed y=0.97 to y=0.93 to push the title down closer to the plots. Added fontsize.
    fig.suptitle(f"Combined FL × KD accuracy — {ds}", y=0.93, fontsize=24, fontweight="bold")
    
    bestfl, bestkd = piv.index[bi], piv.columns[bj]
    
    # 3. Changed xy=(0.5, -0.16) to xy=(0.5, -0.11) to pull the annotation up closer to the x-axis
    ax.annotate(f"best: {bestfl}+{bestkd} = {vals[bi,bj]:.1f}%",
                xy=(1.2, -0.123), xycoords="axes fraction", ha="right",
                fontsize=22, color="#0077A3", fontweight="bold")
                
    vs.save(fig, outdir, f"flkd_heatmap_{ds}")


# --------------------------------------------------------------------------- #
# 2. Synergy heatmap                                                            #
# --------------------------------------------------------------------------- #
def fig_flkd_synergy(res, outdir, ds):
    syn = synergy_matrix(res, ds)
    if syn is None:
        return
    vals = syn.values.astype(float)
    nrow, ncol = vals.shape
    lim = np.nanmax(np.abs(vals)) or 1.0
    fig, ax = plt.subplots(figsize=(1.15 * ncol + 4, 0.8 * nrow + 3))
    im = ax.imshow(vals, aspect="auto", cmap=vs.SYN_CMAP, vmin=-lim, vmax=lim)
    ax.set_xticks(np.arange(ncol)); ax.set_xticklabels(syn.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(nrow)); ax.set_yticklabels(syn.index)
    ax.set_xlabel("KD strategy"); ax.set_ylabel("FL teacher")
    fs = max(8, min(13, int(160 / max(ncol, nrow))))
    for i in range(nrow):
        for j in range(ncol):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=fs,
                        color="#1A1A1A" if abs(v) < lim * 0.6 else "white",
                        fontweight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Synergy: combined − best single (pp)")
    ax.set_title(f"Where does FL+KD beat its parts? — {ds}")
    vs.save(fig, outdir, f"flkd_synergy_{ds}")


# --------------------------------------------------------------------------- #
# Generic family ranking bar                                                     #
# --------------------------------------------------------------------------- #
def _ranking_bar(res, outdir, ds, family, title, name, color_fn, ref_label):
    sub = res.agg[(res.agg.dataset == ds) & (res.agg.family == family)].copy()
    if sub.empty:
        return
    sub = sub.sort_values("acc_mean", ascending=False)
    base = res.baseline_acc(ds)
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(max(9, 0.72 * len(sub) + 3), 6.6))
    colors = [color_fn(m) for m in sub["method"]]
    bars = ax.bar(x, sub["acc_mean"] * 100, color=colors, edgecolor="white",
                  linewidth=1.0, zorder=3, width=0.74)
    ax.errorbar(x, sub["acc_mean"] * 100, yerr=sub["acc_std"] * 100, fmt="none",
                ecolor="#222222", elinewidth=1.6, capsize=4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(sub["method"], rotation=42, ha="right")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{title} — {ds}")
    top = float((sub["acc_mean"] * 100 + sub["acc_std"] * 100).max()) * 1.10 + 3
    if base is not None:
        ax.axhline(base * 100, color="#EF476F", linestyle="--", linewidth=2.4,
                   zorder=5, label=f"{ref_label} ({base*100:.1f}%)")
        top = max(top, base * 100 * 1.05)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=14)
    lo = max(0, float(sub["acc_mean"].min() * 100) - 12)
    ax.set_ylim(lo, min(102, top))
    for b, v in zip(bars, sub["acc_mean"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    vs.despine(ax)
    ax.grid(axis="y", alpha=0.25)
    vs.save(fig, outdir, name)


def fig_fl_ranking(res, outdir, ds):
    _ranking_bar(res, outdir, ds, "fl", "Federated strategy accuracy",
                 f"fl_ranking_{ds}", vs.fl_color, "Centralised")


def fig_kd_ranking(res, outdir, ds):
    _ranking_bar(res, outdir, ds, "kd", "Knowledge-distillation accuracy",
                 f"kd_ranking_{ds}", vs.kd_color, "Teacher (centralised)")


# --------------------------------------------------------------------------- #
# FL convergence with confidence bands                                          #
# --------------------------------------------------------------------------- #
def fig_fl_convergence(res, outdir, ds):
    methods = sorted({m for (cds, fam, m) in res.curves
                      if cds == ds and fam == "fl"})
    if not methods:
        return
    finals, series = {}, {}
    for m in methods:
        mc = res.mean_curve(ds, "fl", m)
        if mc is None:
            continue
        rounds, mean, std = mc
        series[m] = (rounds, mean, std)
        finals[m] = np.nanmax(mean)
    order = sorted(series, key=lambda m: finals[m], reverse=True)
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for m in order:
        rounds, mean, std = series[m]
        c = vs.fl_color(m)
        ax.plot(rounds, mean * 100, color=c, marker="o", markersize=4,
                markevery=max(1, len(mean) // 12), label=m, zorder=3)
        ax.fill_between(rounds, (mean - std) * 100, (mean + std) * 100,
                        color=c, alpha=0.13, zorder=2)
    base = res.baseline_acc(ds)
    if base is not None:
        ax.axhline(base * 100, color="#EF476F", linestyle="--", linewidth=2.2,
                   label=f"Centralised ({base*100:.1f}%)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test instance accuracy (%)")
    ax.set_title(f"Federated convergence — {ds}")
    ax.legend(loc="upper left", borderaxespad=0.5,
              ncol=2, framealpha=0.95, handlelength=1.6, fontsize=18)
    vs.despine(ax)
    vs.save(fig, outdir, f"fl_convergence_{ds}")


# --------------------------------------------------------------------------- #
# FL accuracy vs communication cost                                             #
# --------------------------------------------------------------------------- #
def fig_fl_comm_pareto(res, outdir, ds):
    sub = res.agg[(res.agg.dataset == ds) & (res.agg.family == "fl")].dropna(
        subset=["comm_mb_round"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("acc_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for r in sub.itertuples():
        ax.scatter(r.comm_mb_round, r.acc_mean * 100, s=320, color=vs.fl_color(r.method),
                   edgecolor="white", linewidth=1.5, zorder=3, label=r.method)
        ax.annotate(r.method, (r.comm_mb_round, r.acc_mean * 100),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=10, fontweight="bold", color="#333333")
    base = res.baseline_acc(ds)
    if base is not None:
        ax.axhline(base * 100, color="#EF476F", linestyle="--", linewidth=2.2,
                   label=f"Centralised ({base*100:.1f}%)")
    ax.set_xlabel("Communication / round (MB)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Accuracy vs communication cost — {ds}")
    vs.despine(ax)
    ax.margins(x=0.18, y=0.18)
    vs.save(fig, outdir, f"fl_comm_pareto_{ds}")


# --------------------------------------------------------------------------- #
# KD compression dividend (accuracy retained + teacher->student cost)           #
# --------------------------------------------------------------------------- #
def fig_kd_compression(res, outdir, ds):
    kd = res.agg[(res.agg.dataset == ds) & (res.agg.family == "kd")].dropna(
        subset=["acc_mean"]).copy()
    if kd.empty:
        return
    kd = kd.sort_values("acc_mean", ascending=True)  # best ends on top after barh
    teacher = res.agg[(res.agg.dataset == ds) & (res.agg.family == "baseline")]
    t_acc = float(teacher["acc_mean"].iloc[0] * 100) if not teacher.empty else None
    t_size = float(teacher["size_mb"].iloc[0]) if not teacher.empty \
        else float(kd["teacher_size_mb"].mean())
    t_lat = float(teacher["inference_ms"].iloc[0]) if not teacher.empty \
        else float(kd["teacher_inference_ms"].mean())
    s_size = float(kd["size_mb"].mean())
    s_lat = float(kd["inference_ms"].mean())
    size_red = (1 - s_size / t_size) * 100 if t_size else 0
    lat_red = (1 - s_lat / t_lat) * 100 if t_lat else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.8),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})

    # left: accuracy retained per KD method
    y = np.arange(len(kd))
    bars = ax1.barh(y, kd["acc_mean"] * 100, height=0.72, zorder=3,
                    color=[vs.kd_color(m) for m in kd["method"]], edgecolor="white",
                    linewidth=1.0)
    ax1.errorbar(kd["acc_mean"] * 100, y, xerr=kd["acc_std"] * 100, fmt="none",
                 ecolor="#222222", elinewidth=1.4, capsize=3, zorder=4)
    ax1.set_yticks(y); ax1.set_yticklabels(kd["method"])
    if t_acc is not None:
        ax1.axvline(t_acc, color="#EF476F", linestyle="--", linewidth=2.2,
                    zorder=5, label=f"Teacher ({t_acc:.1f}%)")
        ax1.legend(loc="lower left", framealpha=0.95, fontsize=18)
    lo = max(0, float(kd["acc_mean"].min() * 100) - 6)
    ax1.set_xlim(lo, 102)
    ax1.set_xlabel("Student top-1 accuracy (%)")
    ax1.set_title("Accuracy retained per KD method", fontsize=18, fontweight="normal")
    for b, v, err in zip(bars, kd["acc_mean"] * 100, kd["acc_std"] * 100):
        # Calculate x-position: bar width + error bar length + visual padding
        text_x_pos = b.get_width() + err + 0.25 
        ax1.text(text_x_pos, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}", va="center", ha="left", fontsize=10, fontweight="bold")
    vs.despine(ax1); ax1.grid(axis="x", alpha=0.25)

    # right: compression dividend (student relative to teacher)
    metrics = ["Model size", "Inference time"]
    yp = np.arange(len(metrics))
    h = 0.34
    ax2.barh(yp + h / 2, [1.0, 1.0], h, color="#EF476F", edgecolor="white",
             linewidth=1.0, label="Teacher", zorder=3)
    ax2.barh(yp - h / 2, [s_size / t_size, s_lat / t_lat], h, color="#06D6A0",
             edgecolor="white", linewidth=1.0, label="Student", zorder=3)
    teacher_abs = [f"{t_size:.2f} MB", f"{t_lat:.0f} ms"]
    student_abs = [f"{s_size:.2f} MB", f"{s_lat:.0f} ms"]
    red_txt = [f"−{size_red:.0f}%  ({t_size/s_size:.1f}× smaller)",
               f"−{lat_red:.0f}%  ({t_lat/s_lat:.1f}× faster)"]
    for i in range(len(metrics)):
        ax2.text(1.02, yp[i] + h / 2, teacher_abs[i], va="center", ha="left",
                 fontsize=20, color="#B5364F")
        sx = (s_size / t_size) if i == 0 else (s_lat / t_lat)
        ax2.text(sx + 0.02, yp[i] - h / 2, student_abs[i], va="center", ha="left",
                 fontsize=20, fontweight="bold", color="#0B7A5B")
        ax2.text(0.5, yp[i], red_txt[i], va="center", ha="center",
                 fontsize=20, fontweight="bold", color="#1A1A1A")
    ax2.set_yticks(yp); ax2.set_yticklabels(metrics)
    ax2.set_xlim(0, 1.4); ax2.set_xlabel("Relative to teacher (×)", fontsize=18)
    ax2.set_title("Compression dividend (shared student)", fontsize=18, fontweight="normal")
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=18)
    vs.despine(ax2); ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Distillation: accuracy kept, cost shed — {ds}", y=0.97)
    vs.save(fig, outdir, f"kd_compression_{ds}")


# --------------------------------------------------------------------------- #
# Efficiency Pareto (accuracy vs size & latency, all families)                  #
# --------------------------------------------------------------------------- #
def fig_efficiency_pareto(res, outdir, ds):
    sub = res.agg[(res.agg.dataset == ds)].dropna(subset=["size_mb", "acc_mean"]).copy()
    if sub.empty or sub["family"].nunique() < 2:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.2))
    draw_order = ["combined", "fl", "kd", "baseline"]
    zmap = {"combined": 3, "fl": 4, "kd": 5, "baseline": 6}
    for fam in draw_order:
        s = sub[sub.family == fam]
        if s.empty:
            continue
        a = 0.55 if fam == "combined" else 0.9
        ax1.scatter(s["size_mb"], s["acc_mean"] * 100, s=160,
                    color=vs.FAMILY_COLOR[fam], edgecolor="white", linewidth=1.2,
                    alpha=a, label=FAMILY_LABEL[fam], zorder=zmap[fam])
        si = s.dropna(subset=["inference_ms"])
        ax2.scatter(si["inference_ms"], si["acc_mean"] * 100, s=160,
                    color=vs.FAMILY_COLOR[fam], edgecolor="white", linewidth=1.2,
                    alpha=a, label=FAMILY_LABEL[fam], zorder=zmap[fam])

    def pareto(ax, xcol):
        pts = sub.dropna(subset=[xcol]).sort_values(xcol)
        fx, fy, best = [], [], -1
        for r in pts.itertuples():
            yv = getattr(r, "acc_mean") * 100
            if yv > best + 1e-9:
                best = yv
                fx.append(getattr(r, xcol)); fy.append(yv)
        ax.step(fx, fy, where="post", color="#1A1A1A", linestyle=(0, (5, 2)),
                linewidth=2.6, alpha=0.85, zorder=2.5, label="Pareto frontier")
        ax.scatter(fx, fy, s=70, facecolor="none", edgecolor="#1A1A1A",
                   linewidth=2.0, zorder=7)

    pareto(ax1, "size_mb"); pareto(ax2, "inference_ms")
    ax1.set_xlabel("Model size (MB)"); ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_title("Accuracy vs model size")
    ax2.set_xlabel("Inference time (ms / batch)"); ax2.set_ylabel("Top-1 accuracy (%)")
    ax2.set_title("Accuracy vs latency")
    for ax in (ax1, ax2):
        # vs.despine(ax); ax.legend(framealpha=0.95, loc="lower right")
        vs.despine(ax); ax.legend(framealpha=0.95, loc="lower right", fontsize=18)
    fig.suptitle(f"Efficiency frontier — {ds}", y=1.0)
    vs.save(fig, outdir, f"efficiency_pareto_{ds}")


# --------------------------------------------------------------------------- #
# Top FL+KD leaderboard                                                         #
# --------------------------------------------------------------------------- #
def fig_best_combos(res, outdir, ds, top=12):
    sub = best_combos(res, ds, top)
    if sub.empty:
        return
    sub = sub.iloc[::-1]
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(sub) + 2.6))
    colors = [vs.fl_color(m.split("+")[0]) for m in sub["method"]]
    
    bars = ax.barh(y, sub["acc_mean"] * 100, color=colors, edgecolor="white",
                   linewidth=1.0, zorder=3, height=0.72)
    
    ax.errorbar(sub["acc_mean"] * 100, y, xerr=sub["acc_std"] * 100, fmt="none",
                ecolor="#222222", elinewidth=1.5, capsize=3, zorder=4)
    
    ax.set_yticks(y)
    ax.set_yticklabels(sub["method"], fontsize=12) # Slightly larger method names
    
    # GOAL 3: Make x-axis digits larger
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=16)
    ax.set_title(f"Top {len(sub)} FL+KD combinations ({ds})", x=0.5, y=0.95, ha="center", fontsize=16, fontweight="bold")
    
    # GOAL 1: Make x-axis limits more compact dynamically
    # Find the lowest bound (min value minus its error bar) and highest bound
    min_val = float((sub["acc_mean"] - sub["acc_std"]).min() * 100)
    max_val = float((sub["acc_mean"] + sub["acc_std"]).max() * 100)
    
    # Set tighter bounds: lower gets -1.5 padding, upper gets +2.5 to fit the text
    min_val = float((sub["acc_mean"] - sub["acc_std"]).min() * 100)
    max_val = float((sub["acc_mean"] + sub["acc_std"]).max() * 100)
    lo = max(0, min_val - 0.5)
    hi = max_val + 0.3
    ax.set_xlim(lo, hi)
    # ax.set_xlim(97, 100.8)
    
    for b, v, s in zip(bars, sub["acc_mean"] * 100, sub["acc_std"] * 100):
        # GOAL 2: Avoid overlap by anchoring text AFTER the error bar (v + s)
        text_x_pos = v + s + 0.07 
        
        # GOAL 3: Increase text annotation digit size (e.g., to 12 or 14)
        ax.text(text_x_pos, b.get_y() + b.get_height() / 2,
                f"{v:.1f}±{s:.1f}", va="center", ha="left", fontsize=12,
                fontweight="bold")
                
    vs.despine(ax)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    vs.save(fig, outdir, f"best_combos_{ds}")

# --------------------------------------------------------------------------- #
# Per-seed robustness (spread across seeds, per family)                         #
# --------------------------------------------------------------------------- #
def fig_seed_robustness(res, outdir, ds):
    runs = res.runs[(res.runs.dataset == ds)].dropna(subset=["best_instance_acc"])
    fams = [f for f in FAMILY_ORDER if f in set(runs.family)]
    if len(fams) < 1:
        return
    fig, ax = plt.subplots(figsize=(2.3 * len(fams) + 3, 6.8))
    data = [runs[runs.family == f]["best_instance_acc"].values * 100 for f in fams]
    parts = ax.violinplot(data, showextrema=False)
    for pc, f in zip(parts["bodies"], fams):
        pc.set_facecolor(vs.FAMILY_COLOR[f]); pc.set_alpha(0.30); pc.set_edgecolor("none")
    for i, (f, d) in enumerate(zip(fams, data), start=1):
        jitter = np.random.RandomState(0).normal(0, 0.05, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=26, color=vs.FAMILY_COLOR[f],
                   edgecolor="white", linewidth=0.5, alpha=0.7, zorder=3)
        med = np.median(d)
        ax.plot([i - 0.28, i + 0.28], [med, med], color="#111111", linewidth=2.6, zorder=4)
    ax.set_xticks(np.arange(1, len(fams) + 1))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in fams])
    ax.set_ylabel("Top-1 accuracy across runs (%)")
    ax.set_title(f"Per-method / per-seed spread — {ds}")
    vs.despine(ax); ax.grid(axis="y", alpha=0.25)
    vs.save(fig, outdir, f"seed_robustness_{ds}")


# --------------------------------------------------------------------------- #
# Cross-dataset summary                                                         #
# --------------------------------------------------------------------------- #
def fig_cross_dataset(res, outdir):
    agg = res.agg
    if agg.empty or agg["dataset"].nunique() < 2:
        return
    best = (agg.sort_values("acc_mean", ascending=False)
            .groupby(["dataset", "family"], as_index=False).first())
    datasets = sorted(best["dataset"].unique())
    fams = [f for f in FAMILY_ORDER if f in set(best["family"])]
    x = np.arange(len(datasets))
    w = 0.8 / max(1, len(fams))
    fig, ax = plt.subplots(figsize=(max(10, 3.1 * len(datasets) + 3), 6.8))
    for i, fam in enumerate(fams):
        vals, errs = [], []
        for d in datasets:
            cell = best[(best.dataset == d) & (best.family == fam)]
            vals.append(float(cell["acc_mean"].iloc[0] * 100) if not cell.empty else np.nan)
            errs.append(float(cell["acc_std"].iloc[0] * 100) if not cell.empty else 0.0)
        bars = ax.bar(x + i * w, [0 if np.isnan(v) else v for v in vals], w,
                      yerr=errs, capsize=3, label=FAMILY_LABEL[fam],
                      color=vs.FAMILY_COLOR[fam], edgecolor="white", linewidth=1.0,
                      zorder=3, error_kw=dict(elinewidth=1.4, ecolor="#333"))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=10, rotation=90,
                        fontweight="bold")
    ax.set_xticks(x + w * (len(fams) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.set_ylabel("Best top-1 accuracy (%)")
    ax.set_title("Best accuracy per family across datasets")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.95, fontsize=18)
    vs.despine(ax); ax.grid(axis="y", alpha=0.25)
    vs.save(fig, outdir, "cross_dataset_summary")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("FLKD3D figures")
    ap.add_argument("--output_root", default="outputs")
    ap.add_argument("--model", default="pointnet2_cls_ssg")
    ap.add_argument("--student", default="small_pointnet2")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()

    vs.setup_style()
    res = load_all(args.output_root, args.model, args.student, args.datasets)
    if res.runs.empty:
        print("No runs found under", args.output_root)
        return
    outdir = Path(args.out)
    export(res, outdir / "data")

    fig_cross_dataset(res, outdir)
    for ds in res.datasets:
        print(f"[figures] {ds}")
        fig_flkd_heatmap(res, outdir, ds)
        fig_flkd_synergy(res, outdir, ds)
        fig_fl_ranking(res, outdir, ds)
        fig_kd_ranking(res, outdir, ds)
        fig_fl_convergence(res, outdir, ds)
        fig_fl_comm_pareto(res, outdir, ds)
        fig_kd_compression(res, outdir, ds)
        fig_efficiency_pareto(res, outdir, ds)
        fig_best_combos(res, outdir, ds)
        fig_seed_robustness(res, outdir, ds)
    print(f"\nAll figures (PNG+PDF) written under {outdir}")


if __name__ == "__main__":
    main()
