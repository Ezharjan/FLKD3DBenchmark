"""FLKD3D results loader / aggregator (read-only over ``outputs/``).

This module reads the ``metrics.jsonl`` + ``config.json`` that every run writes
under the dataset-namespaced output tree and turns them into tidy structures the
figure builder consumes.  Nothing here trains or mutates results — it only
*reads* what is already on disk, so it is safe to run anywhere.

Public API
----------
``load_all(output_root, model, student) -> Results``
    Walk the tree and return a :class:`Results` bundle (per-run records,
    mean/std aggregates, per-method convergence curves, metadata).

Run as a script to additionally dump tidy CSV exports::

    python visualizations/flkd_results.py --output_root outputs \
        --out outputs/figures/data
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical orderings (mirror flkd.orchestrate) so rows/cols stay comparable    #
# across datasets.  Unknown methods are appended after the known ones.          #
# --------------------------------------------------------------------------- #
FAMILY_ORDER = ["baseline", "fl", "kd", "combined"]
FAMILY_LABEL = {
    "baseline": "Centralised",
    "fl": "Federated",
    "kd": "Distilled",
    "combined": "FL + KD",
}
FL_ORDER = ["fedavg", "fedprox", "scaffold", "feddyn", "fedavgm", "fedadam",
            "fedyogi", "fedadagrad", "fedmedian", "fedbn", "moon", "ditto",
            "fednova"]
KD_ORDER = ["vanilla", "feature", "attention", "self", "logit_mse", "cosine",
            "crd", "dkd", "rkd", "sp"]
SEED_RE = re.compile(r"_s(\d+)$")


def order_key(order: List[str]):
    """Return a sort key that places ``order`` first, then everything else A-Z."""
    rank = {m: i for i, m in enumerate(order)}
    return lambda m: (rank.get(m, len(order)), m)


# --------------------------------------------------------------------------- #
# Low level IO                                                                  #
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate a half-written trailing line
    except FileNotFoundError:
        pass
    return rows


def _read_json(path: Path) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _seed_of(name: str) -> int:
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# Per-run parsing                                                               #
# --------------------------------------------------------------------------- #
def _parse_run(run_dir: Path, dataset: str, family: str, method: str
               ) -> Tuple[Optional[dict], Optional[List[tuple]]]:
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        return None, None
    cfg = _read_json(run_dir / "config.json")
    evals = [r for r in rows if r.get("split") == "eval"]
    summ = next((r for r in rows if r.get("split") == "summary"), {})

    def best(key):
        vals = [r[key] for r in evals if r.get(key) is not None]
        return max(vals) if vals else None

    def mean_of(key):
        vals = [r[key] for r in evals if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    rec = {
        "dataset": dataset,
        "family": family,
        "method": method,
        "seed": _seed_of(run_dir.name),
        "best_instance_acc": summ.get("best_instance_acc") or best("instance_acc"),
        "best_class_acc": summ.get("best_class_acc") or best("class_acc"),
        "last_instance_acc": summ.get("last_instance_acc"),
        # student/model size: prefer student size for KD, model size otherwise
        "size_mb": summ.get("student_size_mb", summ.get("model_size_mb")),
        "teacher_size_mb": summ.get("teacher_size_mb"),
        "inference_ms": summ.get("student_inference_ms",
                                 summ.get("inference_time_ms")),
        "teacher_inference_ms": summ.get("teacher_inference_ms"),
        "size_reduction_pct": summ.get("size_reduction_percent"),
        "time_reduction_pct": summ.get("time_reduction_percent"),
        "comm_mb_round": summ.get("comm_mb_per_round") or mean_of("comm_mb"),
        "round_seconds": mean_of("round_seconds"),
        "num_classes": summ.get("num_classes", cfg.get("num_category")),
        "epochs_ran": max((int(r["epoch"]) for r in evals if "epoch" in r),
                          default=None),
        # selected hyper-parameters (for the dashboard "run config" panel)
        "num_clients": cfg.get("num_clients"),
        "comm_rounds": cfg.get("communication_rounds"),
        "local_epochs": cfg.get("local_epochs"),
        "partition": cfg.get("partition"),
        "dirichlet_alpha": cfg.get("dirichlet_alpha"),
    }
    curve = [(int(r["epoch"]), r.get("instance_acc"), r.get("class_acc"))
             for r in evals if "epoch" in r and r.get("instance_acc") is not None]
    curve.sort()
    return rec, curve


# --------------------------------------------------------------------------- #
# Results bundle                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Results:
    runs: pd.DataFrame                       # one row per (dataset,family,method,seed)
    agg: pd.DataFrame                        # mean/std across seeds
    curves: Dict[tuple, List[List[tuple]]]   # (ds,family,method) -> [seed curves]
    datasets: List[str] = field(default_factory=list)

    # ---- convergence helpers -------------------------------------------- #
    def mean_curve(self, dataset: str, family: str, method: str
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (rounds, mean_acc, std_acc) aligned across seeds, or None."""
        seeds = self.curves.get((dataset, family, method))
        if not seeds:
            return None
        max_len = max(len(c) for c in seeds)
        if max_len == 0:
            return None
        mat = np.full((len(seeds), max_len), np.nan)
        for i, c in enumerate(seeds):
            for (e, inst, _cls) in c:
                if 1 <= e <= max_len and inst is not None:
                    mat[i, e - 1] = inst
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
        rounds = np.arange(1, max_len + 1)
        return rounds, mean, std

    def baseline_acc(self, dataset: str) -> Optional[float]:
        row = self.agg[(self.agg.dataset == dataset)
                       & (self.agg.family == "baseline")]
        return float(row["acc_mean"].iloc[0]) if not row.empty else None


# --------------------------------------------------------------------------- #
# Tree walk                                                                     #
# --------------------------------------------------------------------------- #
def _discover_datasets(root: Path) -> List[str]:
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_", "figures")):
            continue
        if (d / "classification").exists() or (d / "flkd").exists():
            out.append(d.name)
    return out


def load_all(output_root: str = "outputs",
             model: str = "pointnet2_cls_ssg",
             student: str = "small_pointnet2",
             datasets: Optional[List[str]] = None) -> Results:
    root = Path(output_root)
    datasets = datasets or _discover_datasets(root)
    records: List[dict] = []
    curves: Dict[tuple, List[List[tuple]]] = {}

    def add(run_dir: Path, ds: str, family: str, method: str):
        rec, curve = _parse_run(run_dir, ds, family, method)
        if rec is None:
            return
        records.append(rec)
        curves.setdefault((ds, family, method), []).append(curve or [])

    for ds in datasets:
        base = root / ds
        # ---- centralised baseline ---------------------------------------
        for d in sorted((base / "classification").glob(f"{model}_s*")):
            if d.is_dir():
                add(d, ds, "baseline", "central")
        # ---- federated --------------------------------------------------
        fed = base / "flkd" / "federated" / "classification"
        for d in sorted(fed.glob("fl_*")):
            if not d.is_dir():
                continue
            mid = d.name
            mid = mid[len(f"fl_{model}_"):] if mid.startswith(f"fl_{model}_") else mid[3:]
            mid = SEED_RE.sub("", mid)
            add(d, ds, "fl", mid)
        # ---- knowledge distillation (+ combined) ------------------------
        kdd = base / "flkd" / "knowledge_distillation" / "classification"
        for d in sorted(kdd.glob("kd_*")):
            if not d.is_dir():
                continue
            mid = d.name
            mid = mid[len(f"kd_{model}_"):] if mid.startswith(f"kd_{model}_") else mid[3:]
            mid = SEED_RE.sub("", mid)
            if mid.endswith(f"_{student}"):
                mid = mid[:-(len(student) + 1)]
            family = "combined" if "+" in mid else "kd"
            add(d, ds, family, mid)

    runs = pd.DataFrame(records)
    agg = _aggregate(runs)
    return Results(runs=runs, agg=agg, curves=curves,
                   datasets=[d for d in datasets if d in set(runs.get("dataset", []))])


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["dataset", "family", "method"], as_index=False).agg(
        acc_mean=("best_instance_acc", "mean"),
        acc_std=("best_instance_acc", "std"),
        acc_min=("best_instance_acc", "min"),
        acc_max=("best_instance_acc", "max"),
        cls_mean=("best_class_acc", "mean"),
        cls_std=("best_class_acc", "std"),
        n=("seed", "nunique"),
        size_mb=("size_mb", "mean"),
        teacher_size_mb=("teacher_size_mb", "mean"),
        inference_ms=("inference_ms", "mean"),
        teacher_inference_ms=("teacher_inference_ms", "mean"),
        size_reduction_pct=("size_reduction_pct", "mean"),
        time_reduction_pct=("time_reduction_pct", "mean"),
        comm_mb_round=("comm_mb_round", "mean"),
        round_seconds=("round_seconds", "mean"),
    )
    g["acc_std"] = g["acc_std"].fillna(0.0)
    g["cls_std"] = g["cls_std"].fillna(0.0)
    # convenience percentage columns
    for c in ("acc_mean", "acc_std", "acc_min", "acc_max", "cls_mean", "cls_std"):
        g[c + "_pct"] = (g[c] * 100).round(3)
    return g


# --------------------------------------------------------------------------- #
# Derived analysis helpers (shared by figures + dashboard)                      #
# --------------------------------------------------------------------------- #
def flkd_pivot(res: Results, dataset: str, value: str = "acc_mean"
               ) -> Optional[pd.DataFrame]:
    """Pivot the combined family into an FL(row) x KD(col) accuracy matrix (%)."""
    sub = res.agg[(res.agg.dataset == dataset)
                  & (res.agg.family == "combined")].copy()
    if sub.empty:
        return None
    parts = sub["method"].str.split("+", n=1, expand=True)
    sub["fl"], sub["kd"] = parts[0], parts[1]
    piv = sub.pivot_table(index="fl", columns="kd", values=value, aggfunc="mean") * 100
    rows = [m for m in FL_ORDER if m in piv.index] + \
           [m for m in piv.index if m not in FL_ORDER]
    cols = [m for m in KD_ORDER if m in piv.columns] + \
           [m for m in piv.columns if m not in KD_ORDER]
    return piv.reindex(index=rows, columns=cols)


def synergy_matrix(res: Results, dataset: str) -> Optional[pd.DataFrame]:
    """combined(fl+kd) accuracy minus max(FL-only, KD-only) for that pair (pp)."""
    piv = flkd_pivot(res, dataset, "acc_mean")
    if piv is None:
        return None
    agg = res.agg[res.agg.dataset == dataset]
    fl_only = {r.method: r.acc_mean * 100
               for r in agg[agg.family == "fl"].itertuples()}
    kd_only = {r.method: r.acc_mean * 100
               for r in agg[agg.family == "kd"].itertuples()}
    syn = piv.copy()
    for fl in piv.index:
        for kd in piv.columns:
            base = max(fl_only.get(fl, np.nan), kd_only.get(kd, np.nan))
            syn.loc[fl, kd] = piv.loc[fl, kd] - base
    return syn


def best_combos(res: Results, dataset: str, top: int = 12) -> pd.DataFrame:
    sub = res.agg[(res.agg.dataset == dataset)
                  & (res.agg.family == "combined")].copy()
    return sub.sort_values("acc_mean", ascending=False).head(top)


# --------------------------------------------------------------------------- #
# Export                                                                        #
# --------------------------------------------------------------------------- #
def export(res: Results, out: str) -> Path:
    """Write tidy per-run and mean/std aggregate tables as CSV."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res.runs.to_csv(out / "runs.csv", index=False)
    res.agg.to_csv(out / "agg.csv", index=False)
    print(f"  exported runs.csv, agg.csv -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser("FLKD3D results loader/exporter")
    ap.add_argument("--output_root", default="outputs")
    ap.add_argument("--model", default="pointnet2_cls_ssg")
    ap.add_argument("--student", default="small_pointnet2")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out", default="outputs/figures/data")
    args = ap.parse_args()

    res = load_all(args.output_root, args.model, args.student, args.datasets)
    if res.runs.empty:
        print("No runs found under", args.output_root)
        return
    n_by = res.runs.groupby(["dataset", "family"]).size()
    print(f"Loaded {len(res.runs)} runs across {res.datasets}:")
    print(n_by.to_string())
    export(res, args.out)


if __name__ == "__main__":
    main()
