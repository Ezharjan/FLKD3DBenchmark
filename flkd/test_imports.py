"""Quick dependency / API check for the cg environment.

    python flkd/test_imports.py
"""
import importlib
import os
import sys

# Make this runnable as `python flkd/test_imports.py` from the repo root: put the
# project root (which contains the `flkd` package) on sys.path. Running a script
# by path only puts its own folder (flkd/) on sys.path, not the root, so without
# this `from flkd import ...` fails with ModuleNotFoundError: No module named 'flkd'.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED = ['numpy', 'torch', 'matplotlib', 'pandas', 'scipy', 'tqdm', 'h5py']
FL_FUNCS = ['split_dataset', 'fedavg_aggregate', 'fedmedian_aggregate', 'fedavgm_update',
            'fedopt_update', 'fedadam_update', 'fednova_aggregate', 'fedprox_client_update',
            'scaffold_client_update', 'feddyn_client_update', 'moon_client_update',
            'ditto_client_update', 'bn_param_names', 'broadcast_non_bn']
KD_CLASSES = ['VanillaDistillationLoss', 'FeatureDistillationLoss', 'AttentionDistillationLoss',
              'MultiTeacherDistillationLoss', 'SelfDistillationLoss', 'LogitMSEDistillationLoss',
              'CosineDistillationLoss', 'CRDDistillationLoss', 'DKDDistillationLoss',
              'RKDDistillationLoss', 'SPDistillationLoss', 'SmallPointNetCls',
              'SmallPointNet2ClsSsg', 'DGCNNClsStudent']
ENGINE = ['setup_perf', 'autocast_ctx', 'build_loader', 'augment_points', 'evaluate',
          'get_rng_state', 'set_rng_state', 'atomic_save', 'safe_load']


def main():
    bad = 0
    for mod in REQUIRED:
        try:
            importlib.import_module(mod); print(f'  [ok] {mod}')
        except Exception as e:
            print(f'  [MISSING] {mod}: {e}'); bad += 1
    from flkd import fl_utils, kd_utils, engine
    for name in FL_FUNCS:
        assert hasattr(fl_utils, name), f'fl_utils missing {name}'
    for name in KD_CLASSES:
        assert hasattr(kd_utils, name), f'kd_utils missing {name}'
    for name in ENGINE:
        assert hasattr(engine, name), f'engine missing {name}'
    print(f'  [ok] fl_utils exposes {len(FL_FUNCS)} FL symbols')
    print(f'  [ok] kd_utils exposes {len(KD_CLASSES)} KD symbols')
    print(f'  [ok] engine exposes {len(ENGINE)} symbols')
    import torch
    print(f'  CUDA available: {torch.cuda.is_available()}'
          + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))
    print('FAILED' if bad else 'ALL IMPORTS OK')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
