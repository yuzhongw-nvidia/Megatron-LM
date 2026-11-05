# TEMPORARY diagnostic (uncommitted): install the producer-stage dump when
# MCORE_KDA_CUTE_RAW_DUMP=1.  pytest auto-loads this conftest in every (per-rank)
# worker process, so the monkeypatch lands on both CP2 ranks before the test runs.
# Inert unless the env var is set.  Delete after the matches_single_rank probe.
import os

if os.environ.get("MCORE_KDA_CUTE_RAW_DUMP") == "1":
    import importlib.util
    import pathlib

    _p = (
        pathlib.Path(__file__).resolve().parents[4]
        / "worklogs"
        / "kda-gdr-kernel-fusion"
        / "producer_stage_dump.py"
    )
    _spec = importlib.util.spec_from_file_location("producer_stage_dump", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.install()
