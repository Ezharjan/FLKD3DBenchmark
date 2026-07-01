"""Parallel orchestrator for the FL+KD 3D point-cloud classification benchmark.

Enumerates every atomic training job (centralised baseline / FL / KD / combined
FL+KD) for a chosen *preset* and *datasets*, then either:

  * ``--mode plan``          writes a per-phase job manifest to
                             ``<output_root>/_manifest/`` (one command per line)
                             for an external scheduler or job array;
  * ``--mode local``         runs all phases serially (single GPU);
  * ``--mode local_parallel``runs jobs across several local GPUs via a work
                             queue (``--gpus 0,1,2,3``);
  * ``--mode aggregate``     builds the cross-method tables + figures.

Phases & dependencies
---------------------
  p0_baseline  -> centralised teachers          (no deps)
  p1_fl        -> federated runs                (no deps)
  p2_kd        -> KD from the centralised teacher (needs p0)
  p3_combined  -> KD distilled from the p1 FL teacher checkpoints (needs p1)
  p4_aggregate -> tables + visualisations       (needs all)

Every individual trainer auto-resumes and exits early when already complete, so
re-submitting the array (or re-running ``local``) is fully idempotent.  With
``--skip_done`` (default) the planner omits jobs whose ``best_model.pth`` already
exists, so array sizes shrink as work completes.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from queue import Queue, Empty
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_FL = ['fedavg', 'fedprox', 'scaffold', 'feddyn', 'fedavgm', 'fedadam',
          'fedyogi', 'fedadagrad', 'fedmedian', 'fedbn', 'moon', 'ditto', 'fednova']
ALL_KD = ['vanilla', 'feature', 'attention', 'self', 'logit_mse', 'cosine',
          'crd', 'dkd', 'rkd', 'sp']
# Every classification dataset config the benchmark supports: the three headline
# benchmarks (modelnet40, modelnet10, omni_object3d) plus the three diagnostic
# sets (ycb, gazebosim, craniosynostosis).  Six --dataset tags over five sources
# (ModelNet40/10 share one source).
ALL_DATASETS = ['modelnet40', 'modelnet10', 'omni_object3d', 'ycb', 'gazebosim', 'craniosynostosis']

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    'smoke': dict(
        datasets=['modelnet10'], seeds=[42],
        fl=['fedavg', 'fedprox', 'scaffold', 'moon'], kd=['vanilla', 'logit_mse', 'crd'],
        combined_fl=['fedavg'], combined_kd=['vanilla'],
        communication_rounds=1, local_epochs=1, epoch=1, num_clients=3,
        batch_size=8, num_point=256, partition='label_skew',
    ),
    'standard': dict(
        # Full combinatorial space: every dataset, every FL teacher x every KD
        # objective in the combined phase (13 x 10 = 130 pairs per dataset).
        datasets=ALL_DATASETS, seeds=[42, 7, 123],
        fl=ALL_FL, kd=ALL_KD,
        combined_fl=ALL_FL, combined_kd=ALL_KD,
        combined_datasets=ALL_DATASETS,
        communication_rounds=20, local_epochs=5, epoch=200, num_clients=5,
        batch_size=48, num_point=1024, partition='label_skew',
    ),
    'full': dict(
        # Identical scope to 'standard' (kept as an explicit alias for the whole grid).
        datasets=ALL_DATASETS,
        seeds=[42, 7, 123], fl=ALL_FL, kd=ALL_KD,
        combined_fl=ALL_FL, combined_kd=ALL_KD, combined_datasets=ALL_DATASETS,
        communication_rounds=20, local_epochs=5, epoch=200, num_clients=5,
        batch_size=48, num_point=1024, partition='label_skew',
    ),
}


def parse_args():
    p = argparse.ArgumentParser('FL+KD parallel orchestrator')
    p.add_argument('--mode', choices=['plan', 'local', 'local_parallel', 'aggregate'], default='plan')
    p.add_argument('--preset', choices=list(PRESETS), default='standard')
    p.add_argument('--model', default='pointnet2_cls_ssg')
    p.add_argument('--student_model', default='small_pointnet2',
                   choices=['small_pointnet', 'small_pointnet2', 'dgcnn'])
    p.add_argument('--output_root', default='outputs')
    p.add_argument('--data_root', default='data')
    p.add_argument('--num_workers', type=int, default=-1)
    p.add_argument('--gpus', default='0', help='comma list for local_parallel, e.g. 0,1,2,3')
    p.add_argument('--no_amp', action='store_true', default=False)
    p.add_argument('--min_class_samples', type=int, default=1)
    p.add_argument('--max_runtime_hours', type=float, default=0.0,
                   help='forwarded to every trainer: graceful wall-clock budget per job in hours '
                        '(0 = unlimited). Set below any external time limit so each job stops, '
                        'checkpoints and resumes cleanly instead of being killed mid-epoch.')
    # Overrides (default: take from preset)
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--combined_datasets', nargs='+', default=None,
                   help='restrict only the combined (FL+KD) phase to these datasets; '
                        "handy for chunking p3_combined under a scheduler's array-size limit")
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--fl_strategies', nargs='+', default=None)
    p.add_argument('--kd_strategies', nargs='+', default=None)
    p.add_argument('--partition', default=None, choices=[None, 'iid', 'label_skew', 'dirichlet'])
    p.add_argument('--dirichlet_alpha', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--communication_rounds', type=int, default=None)
    p.add_argument('--local_epochs', type=int, default=None)
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--num_clients', type=int, default=None)
    p.add_argument('--num_point', type=int, default=None)
    p.add_argument('--no_combined', action='store_true', default=False)
    p.add_argument('--no_hard_labels', action='store_true', default=False)
    p.add_argument('--unlabeled', action='store_true', default=False)
    p.add_argument('--skip_done', dest='skip_done', action='store_true', default=True)
    p.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    p.add_argument('--phases', nargs='+',
                   default=['p0_baseline', 'p1_fl', 'p2_kd', 'p3_combined'],
                   help='subset of phases to plan/run')
    return p.parse_args()


def cfg_from_args(args):
    c = dict(PRESETS[args.preset])
    if args.datasets:
        c['datasets'] = args.datasets
        # Scope the combined (FL+KD) phase to the same datasets, so that
        # `--datasets <one>` shrinks the p3_combined job array too.  This is what
        # lets you chunk the combined phase per-dataset to stay under a scheduler's
        # array-size limit (the full-grid combined array is 13*10*6*3 = 2340).
        c['combined_datasets'] = args.datasets
    if args.combined_datasets:
        c['combined_datasets'] = args.combined_datasets
    if args.seeds: c['seeds'] = args.seeds
    if args.fl_strategies: c['fl'] = args.fl_strategies
    if args.kd_strategies: c['kd'] = args.kd_strategies
    if args.partition: c['partition'] = args.partition
    for key in ['batch_size', 'communication_rounds', 'local_epochs', 'epoch',
                'num_clients', 'num_point']:
        v = getattr(args, key, None)
        if v is not None:
            c[key] = v
    c.setdefault('combined_datasets', c['datasets'])
    return c


# ---------------------------------------------------------------------------
# Path + command builders
# ---------------------------------------------------------------------------

def baseline_ckpt(root, ds, model, seed):
    return Path(root) / ds / 'classification' / f'{model}_s{seed}' / 'checkpoints' / 'best_model.pth'


def fl_ckpt(root, ds, model, strat, seed):
    tag = f'fl_{model}_{strat}_s{seed}'
    return Path(root) / ds / 'flkd' / 'federated' / 'classification' / tag / 'checkpoints' / 'best_model.pth'


def kd_ckpt(root, ds, model, kd, sm, seed, prefix=''):
    tag = f'kd_{model}_{prefix}{kd}_{sm}_s{seed}'
    return Path(root) / ds / 'flkd' / 'knowledge_distillation' / 'classification' / tag / 'checkpoints' / 'best_model.pth'


def _common(args):
    s = f'--data_root {args.data_root} --output_root {args.output_root} --num_workers {args.num_workers}'
    if args.no_amp:
        s += ' --no_amp'
    if args.min_class_samples != 1:
        s += f' --min_class_samples {args.min_class_samples}'
    if getattr(args, 'max_runtime_hours', 0) and args.max_runtime_hours > 0:
        s += f' --max_runtime_hours {args.max_runtime_hours}'
    return s


def _fl_strat_args(strat):
    if strat == 'fedprox':
        return '--mu 0.01'
    if strat == 'feddyn':
        return '--alpha 0.01'
    if strat == 'moon':
        return '--moon_mu 1.0 --moon_temperature 0.5'
    if strat == 'fedavgm':
        return '--server_lr 1.0 --server_momentum 0.9'
    if strat in {'fedadam', 'fedyogi', 'fedadagrad'}:
        return '--adaptive_server_lr 0.05 --server_beta1 0.9 --server_beta2 0.999'
    if strat == 'ditto':
        return '--ditto_lambda 0.1'
    return ''


def _kd_strat_args(args, kd):
    s = ''
    if kd in {'feature', 'attention', 'rkd', 'sp'}:
        s += ' --beta 0.5'
    if args.no_hard_labels:
        s += ' --use_hard_labels off'
    if args.unlabeled:
        s += ' --unlabeled'
    return s.strip()


def build_phases(args, cfg):
    c = cfg
    model, sm = args.model, args.student_model
    common = _common(args)
    phases = {k: [] for k in ['p0_baseline', 'p1_fl', 'p2_kd', 'p3_combined']}

    def done(ckpt):
        # Completion is marked by last_model.pth (written only at the end of
        # training / finalisation), NOT best_model.pth (which is written on every
        # improvement and therefore appears early).  Using best_model.pth here
        # would let the planner treat an interrupted, resumable run as finished
        # and not relaunch it.
        return args.skip_done and (Path(ckpt).parent / 'last_model.pth').exists()

    # p0: centralised baselines (teachers for KD)
    for ds in c['datasets']:
        for seed in c['seeds']:
            ck = baseline_ckpt(args.output_root, ds, model, seed)
            if done(ck):
                continue
            phases['p0_baseline'].append(
                f'python train_classification.py --model {model} --dataset {ds} --seed {seed} '
                f'--batch_size {c["batch_size"]} --epoch {c["epoch"]} --num_point {c["num_point"]} '
                f'--log_dir {model}_s{seed} {common}')

    # p1: federated runs
    for ds in c['datasets']:
        for strat in c['fl']:
            for seed in c['seeds']:
                ck = fl_ckpt(args.output_root, ds, model, strat, seed)
                if done(ck):
                    continue
                phases['p1_fl'].append(
                    f'python flkd/advanced_fl_train_classification.py --model {model} '
                    f'--fl_strategy {strat} --dataset {ds} --seed {seed} '
                    f'--batch_size {c["batch_size"]} --num_clients {c["num_clients"]} '
                    f'--local_epochs {c["local_epochs"]} --communication_rounds {c["communication_rounds"]} '
                    f'--num_point {c["num_point"]} --partition {c["partition"]} '
                    f'--dirichlet_alpha {args.dirichlet_alpha} --log_dir fl_{model}_{strat}_s{seed} '
                    f'{_fl_strat_args(strat)} {common}')

    # p2: KD from centralised teacher
    for ds in c['datasets']:
        for kd in c['kd']:
            for seed in c['seeds']:
                ck = kd_ckpt(args.output_root, ds, model, kd, sm, seed)
                if done(ck):
                    continue
                teacher = baseline_ckpt(args.output_root, ds, model, seed)
                phases['p2_kd'].append(
                    f'python flkd/advanced_kd_train_classification.py --teacher_model {model} '
                    f'--student_model {sm} --kd_strategy {kd} --dataset {ds} --seed {seed} '
                    f'--batch_size {c["batch_size"]} --epoch {c["epoch"]} --num_point {c["num_point"]} '
                    f'--teacher_path "{teacher}" --log_dir kd_{model}_{kd}_{sm}_s{seed} '
                    f'{_kd_strat_args(args, kd)} {common}')

    # p3: combined FL+KD (KD distilled from the p1 FL teacher checkpoints)
    if not args.no_combined:
        for ds in c.get('combined_datasets', c['datasets']):
            for fl in c.get('combined_fl', []):
                for kd in c.get('combined_kd', []):
                    for seed in c['seeds']:
                        ck = kd_ckpt(args.output_root, ds, model, kd, sm, seed, prefix=f'{fl}+')
                        if done(ck):
                            continue
                        teacher = fl_ckpt(args.output_root, ds, model, fl, seed)
                        phases['p3_combined'].append(
                            f'python flkd/advanced_kd_train_classification.py --teacher_model {model} '
                            f'--student_model {sm} --kd_strategy {kd} --dataset {ds} --seed {seed} '
                            f'--batch_size {c["batch_size"]} --epoch {c["epoch"]} --num_point {c["num_point"]} '
                            f'--teacher_path "{teacher}" --log_dir kd_{model}_{fl}+{kd}_{sm}_s{seed} '
                            f'{_kd_strat_args(args, kd)} {common}')

    return {k: v for k, v in phases.items() if k in args.phases}


# ---------------------------------------------------------------------------
# Execution backends
# ---------------------------------------------------------------------------

def write_plan(args, phases):
    mdir = Path(args.output_root) / '_manifest'
    mdir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, jobs in phases.items():
        f = mdir / f'{name}.txt'
        f.write_text('\n'.join(jobs) + ('\n' if jobs else ''))
        counts[name] = len(jobs)
    plan = {'phases': list(phases.keys()), 'counts': counts, 'manifest_dir': str(mdir)}
    (mdir / 'plan.json').write_text(json.dumps(plan, indent=2))
    # shell-sourceable env for an external scheduler script
    with open(mdir / 'plan.env', 'w') as fh:
        for name in phases:
            fh.write(f'{name.upper()}_N={counts[name]}\n')
            fh.write(f'{name.upper()}_FILE={mdir / (name + ".txt")}\n')
    total = sum(counts.values())
    print(json.dumps(plan, indent=2))
    print(f'[plan] {total} pending job(s) across {len(phases)} phase(s) -> {mdir}')
    return plan


def run_local(args, phases):
    for name in args.phases:
        for cmd in phases.get(name, []):
            print(f'\n[local:{name}] {cmd}')
            r = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
            if r.returncode != 0:
                print(f'[local] WARNING: job failed (rc={r.returncode}); continuing.')


def run_local_parallel(args, phases):
    """One persistent worker per GPU pulling jobs from a shared queue.

    Each GPU thread grabs the next job the instant it is free, so no GPU ever
    idles while work remains (a static index%%N assignment would let two jobs
    collide on one GPU while another sat idle).  Phases run in order to respect
    the baseline->KD and FL->combined dependencies.
    """
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip() != '']
    if not gpus:
        gpus = ['0']
    print(f'[local_parallel] GPUs: {gpus}')
    # Dependency levels: baseline & FL are independent (run together); KD needs the
    # baseline and combined needs the FL teachers, so they form the next level.
    levels = [['p0_baseline', 'p1_fl'], ['p2_kd', 'p3_combined']]
    for level in levels:
        jobs = [j for name in level if name in args.phases for j in phases.get(name, [])]
        if not jobs:
            continue
        print(f'[local_parallel] level {level}: {len(jobs)} job(s) over {len(gpus)} GPU(s)')
        q: Queue = Queue()
        for j in jobs:
            q.put(j)

        def worker(gpu):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            while True:
                try:
                    cmd = q.get_nowait()
                except Empty:
                    return
                print(f'[gpu {gpu}] {cmd}')
                subprocess.run(cmd, shell=True, cwd=BASE_DIR, env=env)

        threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def run_aggregate(args):
    cmd = (f'python visualizations/make_figures.py --output_root {args.output_root} '
           f'--datasets {" ".join(cfg_from_args(args)["datasets"])} '
           f'--model {args.model} --student {args.student_model} '
           f'--out {Path(args.output_root) / "figures"}')
    print(f'[aggregate] {cmd}')
    subprocess.run(cmd, shell=True, cwd=BASE_DIR)


def main():
    args = parse_args()
    if args.mode == 'aggregate':
        run_aggregate(args)
        return
    cfg = cfg_from_args(args)
    phases = build_phases(args, cfg)
    if args.mode == 'plan':
        write_plan(args, phases)
    elif args.mode == 'local':
        run_local(args, phases)
    elif args.mode == 'local_parallel':
        run_local_parallel(args, phases)


if __name__ == '__main__':
    sys.exit(main())
