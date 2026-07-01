"""Smoke test entry point (run inside the `cg` env).

Runs the torch-free self-check, then the full torch unit/smoke suite
(synthetic data, CPU, ~1 min).  No dataset files required.

    conda activate cg
    python flkd/smoke_test.py
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)


def main():
    print('=== 1/2 torch-free self-check ===')
    from flkd import selfcheck_notorch
    selfcheck_notorch.main()
    print('\n=== 2/2 torch unit/smoke suite ===')
    try:
        import torch  # noqa: F401
    except ImportError:
        print('torch not installed in this environment; skipping torch suite.')
        return
    from flkd import unit_test
    unit_test.main()
    print('\nSMOKE TEST COMPLETE.')


if __name__ == '__main__':
    main()
