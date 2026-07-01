"""Backward-compatible shim -> flkd/orchestrate.py.

The full grid orchestration now lives in ``flkd/orchestrate.py`` (parallel,
dataset-namespaced, auto-resuming).  This shim maps the historical CLI onto the
new orchestrator's ``local`` mode so existing commands keep working.  Prefer
``flkd/orchestrate.py`` directly.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser('run_all_strategies (compat shim)')
    ap.add_argument('--model', default='pointnet2_cls_ssg')
    ap.add_argument('--student_model', default='small_pointnet2')
    ap.add_argument('--dataset', default='modelnet40')
    ap.add_argument('--num_category', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=24)
    ap.add_argument('--num_point', type=int, default=1024)
    ap.add_argument('--num_clients', type=int, default=5)
    ap.add_argument('--local_epochs', type=int, default=5)
    ap.add_argument('--communication_rounds', type=int, default=20)
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--fl_strategies', nargs='+', default=None)
    ap.add_argument('--kd_strategies', nargs='+', default=None)
    ap.add_argument('--partition', default='label_skew')
    ap.add_argument('--dirichlet_alpha', type=float, default=0.5)
    ap.add_argument('--output_dir', default='outputs/results')   # accepted, ignored (compat)
    ap.add_argument('--output_root', default='outputs')
    ap.add_argument('--data_root', default='data')
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--combined_strategies', action='store_true', default=False)
    ap.add_argument('--no_hard_labels', action='store_true', default=False)
    ap.add_argument('--unlabeled', action='store_true', default=False)
    ap.add_argument('--skip_original', action='store_true', default=False)
    ap.add_argument('--skip_fl', action='store_true', default=False)
    ap.add_argument('--skip_kd', action='store_true', default=False)
    ap.add_argument('--skip_comparison', action='store_true', default=False)
    # swallow legacy/no-op server args
    for extra in ['server_lr', 'server_momentum', 'server_beta1', 'server_beta2']:
        ap.add_argument(f'--{extra}', default=None)
    args, _ = ap.parse_known_args()

    phases = []
    if not args.skip_original:
        phases.append('p0_baseline')
    if not args.skip_fl:
        phases.append('p1_fl')
    if not args.skip_kd:
        phases.append('p2_kd')
    if args.combined_strategies:
        phases.append('p3_combined')

    cmd = [sys.executable, str(BASE / 'flkd' / 'orchestrate.py'),
           '--mode', 'local_parallel', '--preset', 'standard',
           '--model', args.model, '--student_model', args.student_model,
           '--datasets', args.dataset, '--seeds', *map(str, args.seeds),
           '--partition', args.partition, '--dirichlet_alpha', str(args.dirichlet_alpha),
           '--batch_size', str(args.batch_size), '--num_point', str(args.num_point),
           '--num_clients', str(args.num_clients), '--local_epochs', str(args.local_epochs),
           '--communication_rounds', str(args.communication_rounds), '--epoch', str(args.epoch),
           '--output_root', args.output_root, '--data_root', args.data_root, '--gpus', args.gpus,
           '--phases', *phases]
    if args.fl_strategies:
        cmd += ['--fl_strategies', *args.fl_strategies]
    if args.kd_strategies:
        cmd += ['--kd_strategies', *args.kd_strategies]
    if not args.combined_strategies:
        cmd += ['--no_combined']
    if args.no_hard_labels:
        cmd += ['--no_hard_labels']
    if args.unlabeled:
        cmd += ['--unlabeled']

    print('[compat shim] ->', ' '.join(cmd))
    rc = subprocess.run(cmd, cwd=str(BASE)).returncode
    if not args.skip_comparison:
        subprocess.run([sys.executable, str(BASE / 'flkd' / 'orchestrate.py'),
                        '--mode', 'aggregate', '--preset', 'standard',
                        '--datasets', args.dataset, '--output_root', args.output_root], cwd=str(BASE))
    sys.exit(rc)


if __name__ == '__main__':
    main()
