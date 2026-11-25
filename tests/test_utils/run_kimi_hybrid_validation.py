# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run focused four-GPU Kimi integration checks in the training container."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == sys.argv[1], (actual, sys.argv[1])
    import emerging_optimizers
    import torch
    import transformer_engine

    assert torch.cuda.device_count() == 4, torch.cuda.device_count()
    print(
        f'[KIMI HYBRID TEST SOURCE] sha={actual} torch={torch.__version__} '
        f'TE={transformer_engine.__version__} emerging={emerging_optimizers.__file__}',
        flush=True,
    )
    # The repository's sessionfinish hook normalizes pytest exit 5 to success.
    # Require real passing test reports on every rank, including dependency-skip cases.
    worker = '''import os, sys, pytest
class Reports:
    passed = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' and report.passed:
            self.passed += 1
reports = Reports()
code = pytest.main(sys.argv[1:], plugins=[reports])
print(f'[KIMI TEST RESULT] rank={os.environ["RANK"]} passed={reports.passed} exit={code}', flush=True)
sys.exit(int(code) if code else (0 if reports.passed else 1))
'''
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1'}
    scopes = [
        'tests/unit_tests/optimizer/test_hybrid_muon_dispatch.py',
        'tests/unit_tests/optimizer/test_optimizer_config.py',
        'tests/unit_tests/optimizer/test_layer_sharded_e2e_parity.py',
        'tests/unit_tests/test_optimizer_state_offloading.py',
    ]
    for scope in scopes:
        print(f'[KIMI TEST SCOPE] {scope}', flush=True)
        subprocess.run(
            [
                sys.executable,
                '-m',
                'torch.distributed.run',
                '--standalone',
                '--nproc-per-node=4',
                '--no-python',
                sys.executable,
                '-c',
                worker,
                '-q',
                '-o',
                'addopts=',
                scope,
            ],
            env=env,
            check=True,
        )
    print('[KIMI HYBRID VALIDATION PASSED]', flush=True)


if __name__ == '__main__':
    main()
