"""Torch-free self-check: validates the data/orchestration/visualisation layers
without importing torch.  Runnable anywhere numpy+matplotlib+pandas are present
(used as the sandbox smoke test).  For the full torch tests use flkd/unit_test.py.

    python flkd/selfcheck_notorch.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE))
sys.path.append(str(BASE / 'data_utils'))

import pc_ops as P  # noqa: E402


def check_pc_ops():
    np.random.seed(0)
    x = np.random.randn(4, 100, 6).astype('float32')
    y = P.augment_batch_np(x)
    assert y.shape == x.shape and not np.shares_memory(y, x)
    assert P.sample_points(np.random.randn(10, 3).astype('float32'), 25).shape == (25, 3)
    assert P.coerce_channels(np.random.randn(5, 6).astype('float32'), False).shape == (5, 3)
    labels = np.array([0] * 10 + [1] * 4 + [2] * 1 + [3] * 6)
    tr, te = P.stratified_split(labels, 0.8, 42)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(labels)
    assert 2 in set(labels[tr]) and 2 not in set(labels[te])  # singleton -> train only
    keep, remap, kept = P.filter_and_relabel(labels, 4, min_k=2)
    assert kept == [0, 1, 3] and remap == {0: 0, 1: 1, 3: 2}
    print('[ok] pc_ops: augment/sample/coerce/stratified-split/filter')


def check_orchestrate_plan():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, str(BASE / 'flkd' / 'orchestrate.py'),
                            '--mode', 'plan', '--preset', 'smoke', '--output_root', d],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        plan = json.loads((Path(d) / '_manifest' / 'plan.json').read_text())
        assert plan['counts']['p1_fl'] >= 1 and plan['counts']['p2_kd'] >= 1
        # combined teacher path must reference an FL checkpoint
        comb = (Path(d) / '_manifest' / 'p3_combined.txt').read_text()
        assert 'federated' in comb and '+' in comb
    print('[ok] orchestrate: plan generates phased manifest with correct deps')


def check_visualization():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / 'out'
        model, sm = 'pointnet2_cls_ssg', 'small_pointnet2'
        for seed in (42, 7):
            run = root / 'modelnet10' / 'classification' / f'{model}_s{seed}'
            run.mkdir(parents=True)
            (run / 'metrics.jsonl').write_text(
                json.dumps({'split': 'summary', 'best_instance_acc': 0.9,
                            'best_class_acc': 0.87, 'model_size_mb': 5.65,
                            'inference_time_ms': 12.0, 'num_classes': 10}) + '\n')
            for strat in ('fedavg', 'fedprox'):
                fr = root / 'modelnet10' / 'flkd' / 'federated' / 'classification' / f'fl_{model}_{strat}_s{seed}'
                fr.mkdir(parents=True)
                lines = [json.dumps({'split': 'eval', 'epoch': e, 'instance_acc': 0.5 + 0.02 * e,
                                     'class_acc': 0.45 + 0.02 * e, 'round_seconds': 30,
                                     'comm_mb': 56}) for e in range(1, 11)]
                lines.append(json.dumps({'split': 'summary', 'best_instance_acc': 0.7,
                                         'best_class_acc': 0.66, 'model_size_mb': 5.65,
                                         'comm_mb_per_round': 56, 'inference_time_ms': 12.0}))
                (fr / 'metrics.jsonl').write_text('\n'.join(lines) + '\n')
        figs = Path(d) / 'figs'
        r = subprocess.run([sys.executable, str(BASE / 'visualizations' / 'make_figures.py'),
                            '--output_root', str(root), '--datasets', 'modelnet10',
                            '--out', str(figs)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        pngs = list(figs.glob('*.png')); pdfs = list(figs.glob('*.pdf'))
        assert pngs and pdfs and len(pngs) == len(pdfs), 'expected matching PNG+PDF figures'
        assert (figs / 'data' / 'agg.csv').exists()
    print(f'[ok] visualization: {len(pngs)} figures as PNG+PDF + aggregate CSV')


def check_py_compile():
    targets = ['flkd/engine.py', 'flkd/fl_utils.py', 'flkd/kd_utils.py',
               'flkd/advanced_fl_train_classification.py', 'flkd/advanced_kd_train_classification.py',
               'flkd/orchestrate.py', 'flkd/run_all_strategies.py', 'flkd/unit_test.py',
               'train_classification.py', 'data_utils/dataset_factory.py', 'data_utils/pc_ops.py',
               'visualizations/flkd_results.py', 'visualizations/viz_style.py',
               'visualizations/make_figures.py', 'experiment.py']
    r = subprocess.run([sys.executable, '-m', 'py_compile'] + [str(BASE / t) for t in targets],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print(f'[ok] py_compile: {len(targets)} core modules compile')


def main():
    print('Torch-free self-check...')
    check_py_compile()
    check_pc_ops()
    check_orchestrate_plan()
    check_visualization()
    print('\nALL TORCH-FREE CHECKS PASSED')


if __name__ == '__main__':
    main()
