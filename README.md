# Benchmarking Federated Learning and Knowledge Distillation for 3D Point Cloud Classification


<h5 align="center">

[![Venue](https://img.shields.io/badge/ECCV-2026-b31b1b.svg)](#)
[![Website](https://img.shields.io/badge/🎤%20Project-Website-blue)](https://ezharjan.github.io/FLKD3DBenchmark)
[![License](https://img.shields.io/badge/⚖️%20Code%20License-MIT-yellow)](https://github.com/Ezharjan/FLKD3DBenchmark/blob/master/LICENSE)

 <br>

</h5>

Welcome to the official code repository for "Benchmarking Federated Learning and Knowledge Distillation for 3D Point Cloud Classification **(ECCV 2026)**".

🔍 For more details, please refer to the project page: [https://ezharjan.github.io/FLKD3DBenchmark/](https://ezharjan.github.io/FLKD3DBenchmark/).

This repository extends the canonical PointNet / PointNet++ codebase with **Federated Learning (FL)**, **Knowledge Distillation (KD)** and combined **FL+KD** pipelines on 3D point clouds. Alongside the classic methods it adds **13 FL algorithms** (including the adaptive-server FedYogi/FedAdagrad, client-local FedBN, and the non-IID MOON/Ditto/FedNova), **11 KD losses** (the 10 single-teacher objectives benchmarked in the 13×10 combined grid, plus a multi-teacher objective excluded from the default grid; including DKD, RKD and Similarity-Preserving), a heterogeneous-architecture DGCNN student, Dirichlet heterogeneity partitions, and multi-seed variance reporting. The whole grid runs on a single modern CUDA GPU and scales across several: **bf16 mixed precision + TF32** auto-enabled on GPUs that support them (exact fp32 elsewhere), multi-worker data loading, parse-once dataset caching, **parallel multi-GPU execution**, and **automatic crash-resume** at the round/epoch granularity.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Project Structure](#4-project-structure)
5. [Efficiency & Auto-Resume](#5-efficiency--auto-resume)
6. [Federated Learning Methods (13)](#6-federated-learning-methods-13)
7. [Knowledge Distillation Methods (11)](#7-knowledge-distillation-methods-11)
8. [Running the Benchmark in Parallel](#8-running-the-benchmark-in-parallel)
9. [Output Layout](#9-output-layout)
10. [Evaluation & Visualization](#10-evaluation--visualization)
11. [Step-by-Step Single Runs](#11-step-by-step-single-runs)
12. [Ablation Studies](#12-ablation-studies)
13. [Testing & Validation](#13-testing--validation)
14. [Citation](#14-citation)

---

## 1. Project Overview

### Supported Tasks

| Task | Role | Models | Dataset |
|------|------|--------|---------|
| Classification | **Benchmark** — FL / KD / FL+KD, multi-seed, parallel | PointNet, PointNet++ (SSG/MSG), DGCNN (student) | ModelNet10/40, OmniObject3D, YCB, GazeboSim, Craniosynostosis |
| Part Segmentation | Standalone baseline (inherited; **not** benchmarked) | `pointnet_part_seg`, `pointnet2_part_seg_ssg`, `pointnet2_part_seg_msg` | ShapeNet Part |
| Semantic Segmentation | Standalone baseline (inherited; **not** benchmarked) | `pointnet_sem_seg`, `pointnet2_sem_seg`, `pointnet2_sem_seg_msg` | S3DIS |

*Note: **SSG** = single-scale grouping, **MSG** = multi-scale grouping — the two PointNet++
set-abstraction variants (single vs. multiple ball-query radii per layer). This is an
**architecture** choice, **not** a task — "SSG" has nothing to do with "segmentation". The
task lives in the model name itself: `*_cls_*` = classification, `*_part_seg_*` / `*_sem_seg_*`
= segmentation.*

> **Scope note.** This repository's contribution — all the **Federated Learning,
> Knowledge Distillation, FL+KD, multi-seed, and parallel-orchestration** machinery — applies to
> **classification** (the five datasets above). **Part Segmentation** and **Semantic Segmentation**
> are the standard single-GPU **PointNet/PointNet++ baselines inherited from the upstream code**:
> the scripts, models, and data loaders are present and runnable on their own datasets (ShapeNet
> Part / S3DIS — see [§11](#11-step-by-step-single-runs)), but they are **not** part of the FL/KD
> benchmark grid and are **not** touched by the orchestrator (`orchestrate.py`). Their datasets are
> also separate from the five classification datasets documented in
> [`data/README.md`](data/README.md) and must be obtained yourself (§3).

### Key Features

- **13 Federated Learning strategies**: FedAvg, FedProx, SCAFFOLD, FedDyn, FedAvgM, FedAdam, **FedYogi**, **FedAdagrad**, FedMedian, **FedBN**, MOON, Ditto, FedNova (the CLI also accepts `vanilla` as an alias of FedAvg). FedYogi/FedAdagrad/FedBN are new.
- **11 Knowledge Distillation losses**: vanilla (Hinton), feature-L2, attention transfer, self-distillation, logit-MSE, cosine, CRD, **DKD** (Decoupled KD), **RKD** (Relational KD), **SP** (Similarity-Preserving), plus `multi_teacher` ensembling.
- **3 student backbones** (`small_pointnet`, `small_pointnet2`, `dgcnn`), covering same-family and heterogeneous teacher/student pairs.
- **3 client partitions**: IID, extreme `label_skew` (default), and `dirichlet(alpha)` for varying degrees of label overlap.
- **Parallel execution**: a planner enumerates every job; a single-node **multi-GPU work queue** runs them across all visible GPUs with correct phase dependencies, and `--mode plan` can emit per-job manifests for an external scheduler.
- **Throughput**: bf16 autocast + TF32 on GPUs that support them (exact fp32 with the cuDNN autotuner otherwise), multi-worker `pin_memory` loaders, **worker-side augmentation** (overlapped with GPU compute), and **parse-once** dataset caching (eliminates repeated OBJ/PLY parsing).
- **Auto-resume**: every trainer checkpoints full state each round/epoch (models, optimizers, per-strategy state, RNG) and resumes exactly where it stopped — re-running only re-does unfinished work.
- **Multi-seed variance reporting** with mean ± std summaries, and **evaluation from a full confusion matrix**: overall accuracy (OA) and **macro per-class accuracy** (independent of batch size).
- **Figures**: every plot saved as **PNG and PDF**, zero margins with large fonts.

---

## 2. Environment Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU (any modern consumer or data-center card; CPU also works for debugging)
- Conda (recommended)

### Installation Steps

```bash
# Step 1: Create + activate the environment
conda create -n cg python=3.10 -y
conda activate cg

# Step 2: Install PyTorch with CUDA support.
# Pick the CUDA build matching your driver (e.g. CUDA 12.x):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 3: Install remaining dependencies
pip install -r requirements.txt        # numpy matplotlib pandas tqdm sympy Pillow scipy h5py plyfile
```

> `plyfile` (the last entry) is **optional**: it speeds up reading the PLY datasets
> (OmniObject3D / YCB / Craniosynostosis), but if it is not installed the loader
> automatically falls back to a built-in, dependency-free PLY parser — see [§3](#3-dataset-preparation).

### Verify Installation

```bash
# Checks deps + that all FL/KD/engine symbols exist
python flkd/test_imports.py

# Verify CUDA is available
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

All training scripts also run on CPU (`--use_cpu`) for debugging; AMP/TF32 are silently disabled off-GPU.

---

## 3. Dataset Preparation

Every dataset has its own folder under `data/`. To keep the repository light and respect each dataset's license, the data itself is **not** included — each folder ships empty (with a `.gitkeep`) next to a short README giving the official download link, the expected on-disk layout, the license, and the citation. **Before training on a dataset, open its `data/<dataset>/README.md` and follow the steps there.** See [`data/README.md`](data/README.md) for an overview of all five classification datasets.

**ModelNet40/10** and **OmniObject3D** are the sound, headline classification benchmarks; **YCB / GazeboSim / Craniosynostosis** are kept as *diagnostic* datasets (see note below).

| Dataset | `--dataset` | Layout | Status |
|---|---|---|---|
| ModelNet40 | `modelnet40` | `data/modelnet40_normal_resampled/` (+ train/test txt) | headline |
| ModelNet10 | `modelnet10` | subset of the above | headline |
| OmniObject3D | `omni_object3d` | `data/omni_object3d/<class>/*.ply` | headline |
| YCB | `ycb` | flat `*.ply`; class = filename stem | diagnostic |
| GazeboSim | `gazebosim` | flat `*.obj`; class = filename stem | diagnostic |
| Craniosynostosis | `craniosynostosis` | `data/craniosynostosis/instances_*/*.ply` | diagnostic |

**Robustness built in.** Non-ModelNet datasets use **deterministic stratified per-class splits** (every class with ≥2 samples appears in both train and test) and **singleton handling** via `--min_class_samples` (classes with fewer than the threshold are dropped and labels re-compacted). This is why the diagnostic datasets (which are dominated by singleton classes, e.g. GazeboSim has ~861 classes over ~1030 files) run without crashing — but treat their numbers as diagnostic, not headline. Use `--min_class_samples 2` (or higher) to focus on classes that actually have train+test support.

**PLY reading is dependency-optional.** The PLY datasets (OmniObject3D, YCB, Craniosynostosis) are read with the [`plyfile`](https://pypi.org/project/plyfile/) package when it is installed; if it is missing, the loader automatically falls back to a built-in, dependency-free PLY parser that handles both ASCII and binary (little/big-endian) PLY, with or without per-vertex normals, and skips mesh faces. PLY datasets therefore load whether or not `plyfile` is present — installing it is a faster convenience, not a hard requirement. (`.obj` meshes and `.txt`/`.xyz`/`.pts` clouds are parsed directly with NumPy and need no extra packages; HDF5 submodels need `h5py`.)

### Segmentation datasets (optional — only for the standalone baselines)

These are needed **only** if you want to run the inherited Part/Semantic-Segmentation
baselines ([§11](#11-step-by-step-single-runs)). They are **not** required for — and play no
part in — the FL/KD classification benchmark, and are intentionally not listed in
`data/README.md`. Download them from their official sources and lay them out exactly as the
segmentation scripts expect:

```bash
# ShapeNet Part (used by train_partseg.py / test_partseg.py):
data/shapenetcore_partanno_segmentation_benchmark_v0_normal/

# S3DIS (used by train_semseg.py / test_semseg.py): place the raw release under
data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/
# then preprocess it into data/stanford_indoor3d/ (the path the loader reads):
python data_utils/collect_indoor3d_data.py
```

---

## 4. Project Structure

```
FLKD3DBenchmark/
├── train_classification.py        # centralised baseline (AMP + resume)
├── experiment.py                  # run dir / logging / metrics.jsonl helper
├── provider.py                    # augmentation used by the standalone seg baselines (classification uses pc_ops)
│
├── models/                        # PointNet / PointNet++ backbones
├── data_utils/
│   ├── dataset_factory.py         # multi-dataset loader (cache + stratified split)
│   ├── pc_ops.py                  # torch-free point ops (augment/sample/split)  [unit-tested]
│   └── ModelNetDataLoader.py …
│
├── flkd/
│   ├── engine.py                  # perf setup, bf16 autocast, evaluate(), resume helpers
│   ├── fl_utils.py                # 13 FL strategies (aggregators + client updates)
│   ├── kd_utils.py                # 11 KD losses (10 benchmarked + multi-teacher) + 3 student backbones
│   ├── advanced_fl_train_classification.py   # FL trainer (AMP, per-round resume)
│   ├── advanced_kd_train_classification.py   # KD trainer (AMP, per-epoch resume)
│   ├── orchestrate.py             # grid entry point: planner / manifest / local-parallel / aggregate
│   ├── run_all_strategies.py      # backward-compat shim -> orchestrate.py
│   ├── unit_test.py               # torch unit/smoke suite (synthetic, CPU, ~1 min)
│   ├── selfcheck_notorch.py       # torch-free CI check (data/orchestrate/viz)
│   └── smoke_test.py / test_imports.py
│
├── visualizations/
│   ├── flkd_results.py            # read-only results loader / aggregator
│   ├── make_figures.py            # PNG/PDF figure suite (zero-margin, large fonts)
│   └── viz_style.py               # shared plot style + palettes
│
└── outputs/                       # all results (git-ignored), namespaced per dataset
```

---

## 5. Efficiency & Auto-Resume

**Throughput knobs** (sensible defaults; override per run):

| Flag | Default | Effect |
|---|---|---|
| (auto) | GPU-aware | **GPUs without native bf16**: exact fp32 + cuDNN autotuner. **GPUs with native bf16**: bf16 autocast + TF32 auto-enabled |
| `--no_amp` | off | force exact fp32 everywhere (debug) |
| `--num_workers N` | `-1` (auto) | DataLoader workers; `-1` = min(8, available CPUs) |
| `--cache_dataset` / `--no_cache` | on | parse-once preprocessing of folder datasets |
| `--batch_size` | 24 | raise on larger-memory GPUs (e.g. 48–64) for higher utilisation |
| `--num_point` | 1024 | points per cloud |

Precision is selected automatically per GPU. On GPUs **without native bf16** it runs **exact fp32** (with the cuDNN autotuner): these PointNet++ models are small and gather/FPS-bound, so fp32 is already fast and throughput instead comes from running many jobs in parallel. On GPUs that have **native bf16**, the engine automatically enables **bf16 autocast + TF32** — bf16 keeps the fp32 exponent range so the distance/FPS/ball-query math is stable and needs **no gradient scaler** (also keeping the hand-written SCAFFOLD/FedDyn SGD correct).

**Auto-resume.** Every trainer writes `checkpoints/resume.pth` atomically each round (FL) or epoch (KD/baseline), capturing models, optimizers, LR schedulers (KD/baseline only), per-strategy FL state (control variates, server momentum/Adam state, MOON history, Ditto personal models) and RNG state. On restart it continues exactly; a fully finished run exits immediately. Re-running the orchestrator is therefore idempotent and only re-launches unfinished jobs.

---

## 6. Federated Learning Methods (13)

| Strategy | Description | Key flags |
|---|---|---|
| `fedavg` / `vanilla` | Weighted FedAvg (McMahan et al., 2017) | – |
| `fedprox` | FedAvg + proximal term (Li et al., 2020) | `--mu 0.01` |
| `scaffold` | Control-variate correction (Karimireddy et al., 2020) | – |
| `feddyn` | Dynamic regularisation (Acar et al., 2021) | `--alpha 0.01` |
| `fedavgm` | Server SGD + momentum (Hsu et al., 2019) | `--server_lr 1.0 --server_momentum 0.9` |
| `fedadam` | Server Adam (Reddi et al., 2021) | `--adaptive_server_lr 0.05` |
| `fedyogi` | Server Yogi (Reddi et al., 2021) | `--adaptive_server_lr 0.05` |
| `fedadagrad` | Server Adagrad (Reddi et al., 2021) | `--adaptive_server_lr 0.05` |
| `fedmedian` | Coordinate-median (Yin et al., 2018) | – |
| `fedbn` | Client-local BatchNorm (Li et al., 2021) | – |
| `moon` | Model-contrastive FL (Li, He, Song 2021) | `--moon_mu 1.0 --moon_temperature 0.5` |
| `ditto` | Personalised FL (Li, Hu, Beirami, Smith 2021) | `--ditto_lambda 0.1` |
| `fednova` | Normalised averaging (Wang et al., 2020) | – (forces SGD locals) |

Common FL flags: `--num_clients 5 --local_epochs 5 --communication_rounds 20 --partition {iid,label_skew,dirichlet} --dirichlet_alpha 0.5 --seed 42`.

**FedNova** automatically switches client optimisers to SGD because its τ-normalisation is only valid for plain SGD locals.

**Personalised-FL evaluation note (FedBN, Ditto).** All methods are scored on the same shared test set using the *global* model, for a consistent protocol. FedBN keeps BatchNorm client-local, so its global model uses BN-averaged statistics as a proxy; Ditto's per-client personalised models are likewise not the global model. Treat the global-model numbers for these two as a consistent proxy; the per-round `metrics.jsonl` also records the last-round (non-peeked) accuracy if you prefer to report that instead of best-validation.

---

## 7. Knowledge Distillation Methods (11)

| Strategy | Description | Uses features? | Key flags |
|---|---|---|---|
| `vanilla` / `basic` | Hinton KD: CE + KL(T²) | no | `--alpha 0.5 --temperature 2.0` |
| `feature` | L2 on normalised features | yes | `--alpha 0.5 --beta 0.5` |
| `attention` | Attention transfer (Zagoruyko & Komodakis, 2017) | yes | `--alpha 0.5 --beta 0.5` |
| `self` | Born-Again self-distillation (Furlanello et al., 2018) | no | `--alpha 0.5 --temperature 2.0` |
| `logit_mse` | MSE on model outputs | no | `--alpha 0.5` |
| `cosine` | Cosine on output vectors | no | `--alpha 0.5` |
| `crd` | Contrastive Representation Distillation (Tian et al., 2020) | yes | `--crd_temperature 0.07` |
| `dkd` | Decoupled KD: TCKD + NCKD (Zhao et al., 2022) | no | `--dkd_alpha 1.0 --dkd_beta 8.0` |
| `rkd` | Relational KD: distance + angle (Park et al., 2019) | yes | `--alpha 0.5 --beta 0.5` |
| `sp` | Similarity-Preserving (Tung & Mori, 2019) | yes | `--alpha 0.5 --beta 0.5` |
| `multi_teacher` | Weighted teacher ensemble | no | `--teacher_paths …` |

Students (`--student_model`): `small_pointnet` (~0.3 MB), `small_pointnet2` (~1.4 MB, same family), `dgcnn` (~1.0 MB, heterogeneous); for reference the PointNet++ SSG teacher is ~5.7 MB (sizes given at 40 classes). All three expose a penultimate feature so feature-based KD works for every student.

Convention: the backbones return `log_softmax`. Because softmax is shift-invariant and `log_softmax` is idempotent on log-probabilities, `softmax(log_softmax(z)/T) == softmax(z/T)` exactly — so the soft-target KL and hard-label CE terms follow the standard formulation.

---

## 8. Running the Benchmark in Parallel

Everything runs through `flkd/orchestrate.py`, which enumerates every atomic job (baseline / FL / KD / combined FL+KD) for a preset, executes the phases in dependency order, and relies on per-run auto-resume so any invocation is idempotent.

> TL;DR — run the whole pipeline end-to-end on the GPUs you have:
> ```bash
> python flkd/orchestrate.py --mode local_parallel --preset standard --gpus 0,1,2,3
> python flkd/orchestrate.py --mode local          --preset smoke                 # single-GPU quick check
> ```

### A) Local multi-GPU (one work queue over all your GPUs)

```bash
# Runs the training phases across the listed GPUs, then builds the tables + figures:
python flkd/orchestrate.py --mode local_parallel --preset standard --gpus 0,1,2,3,4,5,6,7
python flkd/orchestrate.py --mode aggregate       --preset standard
```

One persistent worker per GPU pulls jobs from a shared queue, so a GPU grabs the next job the instant it is free (no GPU idles while work remains). Independent phases are overlapped in dependency *levels* — baseline + FL run together, then KD + combined. It is fully resumable: re-run the same command after any interruption and it only re-does unfinished work.

### B) Single GPU

```bash
python flkd/orchestrate.py --mode local     --preset standard   # all phases, serially
python flkd/orchestrate.py --mode aggregate --preset standard   # tables + figures
```

**Presets**

| Preset | Datasets | Seeds | FL | KD | Combined | Rounds/Epochs |
|---|---|---|---|---|---|---|
| `smoke` | modelnet10 | 42 | 4 | 3 | 1×1 | 1 / 1 (tiny, minutes) |
| `standard` *(default)* | all 6 (incl. diagnostic) | 42,7,123 | all 13 | 10 | 13×10 on all 6 | 20 / 200 |
| `full` | all 6 (incl. diagnostic) | 42,7,123 | all 13 | 10 | 13×10 on all 6 | 20 / 200 |

> **`standard` runs the entire combinatorial grid.** The combined FL+KD phase crosses every one of the 13 FL teachers with all 10 KD objectives (13×10 = 130 pairs) on each of the six dataset configs, repeated over three seeds — so `standard` is identical in scope to `full`. Use `smoke` for a quick functional check; pass explicit overrides (e.g. `--datasets modelnet40`, `--fl_strategies …`, `--no_combined`) to run a narrower slice.
>
> The FL grid runs the 13 distinct algorithms (`vanilla` is a FedAvg alias, never run twice). The KD grid runs the 10 single-teacher losses; `multi_teacher` needs ≥2 teacher checkpoints and is excluded from the default grid (run it manually if wanted). The presets default to **`--batch_size 48`**; raise it on larger-memory GPUs or drop to 24–32 on smaller cards.

### C) Manual split across GPUs (different seeds / strategies per card)

Because outputs are dataset-namespaced and every run auto-resumes, you can shard the grid by hand and merge results later:

```bash
# GPU 0: seed 42 ; GPU 1: seed 7 ; GPU 2: seed 123  (standard preset, FL+KD only)
CUDA_VISIBLE_DEVICES=0 python flkd/orchestrate.py --mode local --preset standard --seeds 42  --no_combined &
CUDA_VISIBLE_DEVICES=1 python flkd/orchestrate.py --mode local --preset standard --seeds 7   --no_combined &
CUDA_VISIBLE_DEVICES=2 python flkd/orchestrate.py --mode local --preset standard --seeds 123 --no_combined &
wait
python flkd/orchestrate.py --mode aggregate --preset standard      # merge -> tables + figures

# or split by strategy / dataset:
CUDA_VISIBLE_DEVICES=0 python flkd/orchestrate.py --mode local --preset standard --fl_strategies scaffold moon ditto --phases p1_fl &
CUDA_VISIBLE_DEVICES=1 python flkd/orchestrate.py --mode local --preset standard --datasets omni_object3d &
```

`--mode plan` writes the per-phase manifests to `outputs/_manifest/` (one shell command per line) without running anything — handy to inspect the job list or feed it to an external scheduler or job array.

---

## 9. Output Layout

Everything is namespaced by dataset so multiple datasets never collide:

```
outputs/<dataset>/
├── classification/<model>_s<seed>/                       # centralised baseline
├── flkd/federated/classification/fl_<model>_<strat>_s<seed>/
└── flkd/knowledge_distillation/classification/
    ├── kd_<model>_<kd>_<student>_s<seed>/                # KD (centralised teacher)
    └── kd_<model>_<fl>+<kd>_<student>_s<seed>/           # combined FL+KD
```

Each run directory contains:
- `config.json` — frozen CLI args.
- `logs/train.log` — human-readable log (also streamed to stdout).
- `metrics.jsonl` — one JSON object per line, `split ∈ {train, eval, summary}`, with epoch/round, `instance_acc`, `class_acc`, per-round wall-clock (`round_seconds`) and communication MB (`comm_mb`).
- `checkpoints/best_model.pth`, `last_model.pth`, and `resume.pth` (resume state).

Aggregated tables and figures are written to `outputs/figures/`.

---

## 10. Evaluation & Visualization

Aggregation + plotting is a single data-driven module (no torch needed):

```bash
python visualizations/make_figures.py \
    --output_root outputs --datasets modelnet40 modelnet10 omni_object3d \
    --model pointnet2_cls_ssg --student small_pointnet2 \
    --out outputs/figures
```

It reads every `metrics.jsonl`, builds **mean ± std across seeds**, and writes:

- Tidy tables under `outputs/figures/data/`: `runs.csv` (one row per run) and `agg.csv` (mean/std per method group).
- Figures (each as **`.png` and `.pdf`**, zero-margin, large fonts): `fl_ranking_<ds>`, `fl_convergence_<ds>`, `fl_comm_pareto_<ds>`, `kd_ranking_<ds>`, `kd_compression_<ds>`, `flkd_heatmap_<ds>` (FL×KD grid), `flkd_synergy_<ds>`, `efficiency_pareto_<ds>` (accuracy vs size & latency), `best_combos_<ds>`, `seed_robustness_<ds>`, and `cross_dataset_summary`.

Evaluation: training runs in fp32 unless the GPU natively supports bf16, and **accuracy is always measured in fp32**, with overall accuracy and macro per-class accuracy computed from a full confusion matrix (independent of batch size). All methods share the same test split and protocol. When a figure's data is absent it is skipped.

### Comparison Metrics

| Metric | Description |
|--------|-------------|
| Instance Accuracy (best) | Best test-instance accuracy across rounds/epochs |
| Class Accuracy (best) | Macro per-class accuracy (balanced) from the confusion matrix |
| Seed | Seed used for the run |
| Size (MB) | Total parameter+buffer memory footprint |
| Inference Time (ms) | Average forward pass time per batch |
| Mean / Std | Aggregated mean and std across seeds (in summary CSV) |

---

## 11. Step-by-Step Single Runs

```bash
# 1) Centralised baseline / KD teacher
python train_classification.py --model pointnet2_cls_ssg --dataset modelnet40 \
    --batch_size 64 --epoch 200 --seed 42 --log_dir pointnet2_cls_ssg_s42

# 2) Federated training (any of the 13 strategies)
python flkd/advanced_fl_train_classification.py --model pointnet2_cls_ssg \
    --fl_strategy fedyogi --dataset modelnet40 --num_clients 5 --local_epochs 5 \
    --communication_rounds 20 --partition label_skew --seed 42 --log_dir fl_fedyogi_s42

# 3) Knowledge distillation from the centralised teacher
python flkd/advanced_kd_train_classification.py --teacher_model pointnet2_cls_ssg \
    --student_model small_pointnet2 --kd_strategy dkd --dataset modelnet40 --epoch 200 \
    --teacher_path outputs/modelnet40/classification/pointnet2_cls_ssg_s42/checkpoints/best_model.pth \
    --seed 42 --log_dir kd_dkd_s42

# 4) Combined FL+KD (distil from an FL teacher checkpoint)
python flkd/advanced_kd_train_classification.py --teacher_model pointnet2_cls_ssg \
    --student_model small_pointnet2 --kd_strategy logit_mse --dataset modelnet40 \
    --teacher_path outputs/modelnet40/flkd/federated/classification/fl_pointnet2_cls_ssg_scaffold_s42/checkpoints/best_model.pth \
    --seed 42 --log_dir kd_scaffold+logit_mse_s42
```

Every command auto-resumes if interrupted; pass `--no_resume` to force a fresh start.

### Part / Semantic Segmentation (standalone PointNet/PointNet++ baselines — not FL/KD)

These are the upstream single-GPU baselines; they have no federated/distillation/orchestration
path and require the optional segmentation datasets from [§3](#3-dataset-preparation).

```bash
# Part segmentation
python train_partseg.py --model pointnet2_part_seg_msg --batch_size 16 --log_dir pointnet2_part_seg_msg --normal
python test_partseg.py  --log_dir pointnet2_part_seg_msg --normal

# Semantic segmentation
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py  --log_dir pointnet2_sem_seg --test_area 5 --visual
```

---

## 12. Ablation Studies

```bash
# Realistic non-IID (Dirichlet) instead of worst-case label-skew
python flkd/orchestrate.py --mode local --preset standard --partition dirichlet \
    --dirichlet_alpha 0.1 --fl_strategies fedavg fedprox scaffold moon --phases p1_fl

# Heterogeneous DGCNN student
python flkd/orchestrate.py --mode local --preset standard --student_model dgcnn \
    --kd_strategies vanilla feature attention crd rkd --phases p2_kd

# "Recovery-illusion" controls: no-CE (alpha=1) and unlabeled proxy data
python flkd/orchestrate.py --mode local --preset standard --no_hard_labels --phases p2_kd
python flkd/orchestrate.py --mode local --preset standard --unlabeled      --phases p2_kd

# Temperature sweep on a single KD run
python flkd/advanced_kd_train_classification.py --kd_strategy vanilla \
    --temperature_sweep 1 2 4 8 --teacher_path <teacher.pth> --dataset modelnet40 --log_dir kd_Tsweep
```

Every FL run logs the per-round wall-clock time and an estimate of per-round communication MB to `metrics.jsonl`; every KD run logs the per-epoch wall-clock time. These are summarised in `outputs/figures/data/agg.csv`.

---

## 13. Testing & Validation

```bash
conda activate cg
python flkd/test_imports.py        # deps + API surface
python flkd/selfcheck_notorch.py   # torch-free: data ops, orchestrate plan, PNG+PDF viz
python flkd/unit_test.py           # torch suite: all FL aggregators+updates, 11 KD losses,
                                   #   the evaluator, resume helpers, tiny end-to-end FL round + KD epoch
python flkd/smoke_test.py          # runs both of the above
```

`unit_test.py` runs on CPU with synthetic data (~1 min, no dataset files needed) and includes a regression test ensuring attention-transfer KD is non-degenerate.

---

## 14. Citation

**If you use this benchmark, please cite:**

```bibtex
@inproceedings{aizierjiang26benchmark,
  title={Benchmarking Federated Learning and Knowledge Distillation for 3D Point Cloud Classification},
  author={Aizierjiang Aiersilan},
  booktitle={European Conference on Computer Vision},
  organization={Springer},
  year={2026}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).

This codebase builds directly on **PointNet** and **PointNet++**; if you use it, please also cite:

```bibtex
@inproceedings{qi2017pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={CVPR}, year={2017}}
@inproceedings{qi2017pointnetplusplus,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={NeurIPS}, year={2017}}
```

For the federated-learning and knowledge-distillation methods, please cite the respective original publications (FedAvg, FedProx, SCAFFOLD, FedDyn, FedAvgM, FedAdam, FedYogi, FedAdagrad, FedMedian, FedBN, MOON, Ditto, FedNova; Hinton-KD, Attention-Transfer, BAN, CRD, DKD, RKD, SP, DGCNN).

---

## Acknowledgements

This project is built on top of **PointNet** and **PointNet++** (Qi et al.). The backbone architectures, data loaders, and training/evaluation scripts are adapted from their open-source PyTorch implementation, on which the federated-learning and knowledge-distillation components in this repository are built. We gratefully acknowledge the original authors for making their work and code openly available.

We are equally grateful to the authors and maintainers of the datasets used in this project for releasing them to the research community — ModelNet, ShapeNet and ShapeNet Part, the Stanford Large-Scale 3D Indoor Spaces dataset (S3DIS), OmniObject3D, the YCB Object and Model Set, Google Scanned Objects (hosted on Gazebo Fuel), and the craniosynostosis statistical-shape-model scans. Each dataset remains the property of its original authors; please refer to the corresponding `data/<dataset>/README.md` for its source, license, and citation.

---

## License

This project is for research and educational purposes. For this project, the **MIT license** applies (see [`LICENSE`](LICENSE)). For the PointNet / PointNet++ parts, see the original PointNet / PointNet++ repositories for their license details.
