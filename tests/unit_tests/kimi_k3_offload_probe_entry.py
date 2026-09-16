# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Launcher-agnostic pytest entry for the Kimi-K3 chunked-offload parity probes.

Runs the probe test files under an srun-native Megatron launch (one process per GPU,
Megatron CLI arguments on ``sys.argv`` are ignored) or under ``torchrun``. Rank and
world-size variables fall back to the Slurm task variables so the vendor test
utilities (``tests/unit_tests/test_utilities.py``) can rendezvous.

Env:
    KIMI_PROBE_TEST          pytest target (default: tests/unit_tests/test_kimi_offload_trainstep.py)
    KIMI_PROBE_PYTEST_EXTRA  extra pytest arguments, whitespace separated
"""

import os
import sys

os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", os.environ["RANK"]))
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29511")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import pytest  # noqa: E402  (after env setup on purpose)


def main() -> int:
    test_target = os.environ.get(
        "KIMI_PROBE_TEST", "tests/unit_tests/test_kimi_offload_trainstep.py"
    )
    extra = os.environ.get("KIMI_PROBE_PYTEST_EXTRA", "").split()
    print(
        f"[probe-entry] rank={os.environ['RANK']} world={os.environ['WORLD_SIZE']} "
        f"local_rank={os.environ['LOCAL_RANK']} target={test_target} extra={extra}",
        flush=True,
    )
    sys.argv = ["pytest"]
    return int(
        pytest.main(
            [test_target, "-s", "-rA", "--durations=0", "-p", "no:cacheprovider", *extra]
        )
    )


if __name__ == "__main__":
    sys.exit(main())
