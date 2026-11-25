# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Write the per-node launch script of a Kimi-K3 GTP recipe to standard output.

The recipe YAML has the schema ``DEPENDENCIES`` / ``ENV_VARS`` / ``ARGS`` of this directory's
other recipes: ``ENV_VARS`` are exported, ``ARGS`` become command-line flags (lists become
several values, ``true`` becomes a flag, ``false`` and ``null`` are omitted).

Differences from the launcher in ../README.md, all three part of the recorded measurement:

* the entry point is ``kimi_k3_gtp_entry.py`` (wraps ``pretrain_hybrid.py``);
* the repository root is put on ``PYTHONPATH`` (the wrapper lives in a subdirectory);
* ``hybridep_overlay_env.sh patched`` is sourced first, so that HybridEP compiles its kernels
  from a patched copy of its JIT header (``--hybridep-overlay installed`` keeps the installed
  package and only prints its digest, ``none`` skips the step).

Usage, inside the container with the repository root as working directory:

    python examples/moe_recipes/kimi_k3/gtp_launch/render_launcher.py RECIPE.yaml > run.sh
    NNODES=64 NODE_RANK=<0..63> MASTER_ADDR=<host of node rank 0> bash run.sh [extra flags]

Extra flags are appended to the training command line (the last occurrence of a flag wins).
"""

import argparse
import os
import shlex

import yaml

HERE = "examples/moe_recipes/kimi_k3/gtp_launch"


def recipe_flags(args):
    """Command-line words of a recipe's ``ARGS`` mapping."""
    words = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        words.append("--" + key.replace("_", "-"))
        if value is not True:
            values = value if isinstance(value, list) else [value]
            words.extend(os.path.expandvars(str(item)) for item in values)
    return words


def render(recipe, entry, overlay):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key, value in recipe.get("ENV_VARS", {}).items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append('export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"')
    if overlay != "none":
        lines.append(f". {HERE}/hybridep_overlay_env.sh {overlay}")
    launcher = (
        'exec python -m torch.distributed.run --nnodes="${NNODES:?}" '
        '--nproc-per-node="${GPUS_PER_NODE:-4}" --node-rank="${NODE_RANK:?}" '
        '--master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29500}" '
    )
    lines.append(
        launcher + shlex.quote(entry) + " " + shlex.join(recipe_flags(recipe["ARGS"])) + ' "$@"'
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", help="recipe YAML (DEPENDENCIES / ENV_VARS / ARGS)")
    parser.add_argument(
        "--entry",
        default=f"{HERE}/kimi_k3_gtp_entry.py",
        help="training entry point, relative to the repository root",
    )
    parser.add_argument(
        "--hybridep-overlay",
        choices=("patched", "installed", "none"),
        default="patched",
        help="HybridEP JIT header: patched copy (default), installed package, or no step",
    )
    options = parser.parse_args()
    with open(options.recipe) as stream:
        recipe = yaml.safe_load(stream)
    print(render(recipe, options.entry, options.hybridep_overlay), end="")


if __name__ == "__main__":
    main()
