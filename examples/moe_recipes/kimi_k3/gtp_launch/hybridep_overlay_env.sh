# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Source this file (do not execute it) once per node, inside the training container, right
# before the Megatron entry point (the launcher written by render_launcher.py does it):
#
#   . examples/moe_recipes/kimi_k3/gtp_launch/hybridep_overlay_env.sh installed
#   . examples/moe_recipes/kimi_k3/gtp_launch/hybridep_overlay_env.sh patched
#
# "patched" copies the container's deep_ep package to a per-task directory under /tmp, patches
# its JIT kernel header with hybridep_combine_prob_tma_patch.py and prepends the copy to
# PYTHONPATH, so the training process imports it and HybridEP compiles the patched kernels.
# "installed" changes nothing. Both modes print which deep_ep package and header digest the
# training process will use; any failure returns non-zero so the launch stops.

_hybridep_overlay_env() {
  local mode="${1:-installed}"
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
  local rank_tag="${SLURM_JOB_ID:-nojob}_${SLURM_PROCID:-${PMIX_RANK:-0}}"

  if [ "${mode}" = "patched" ]; then
    local pkg_dir
    pkg_dir="$(python -c 'import deep_ep, os; print(os.path.dirname(os.path.abspath(deep_ep.__file__)))' 2>/dev/null | tail -1)"
    if [ -z "${pkg_dir}" ] || [ ! -f "${pkg_dir}/backend/hybrid_ep_backend.cuh" ]; then
      echo "[HYBRIDEP OVERLAY] cannot locate the installed deep_ep package (${pkg_dir})" >&2
      return 1
    fi
    local overlay="/tmp/hybridep_overlay_${rank_tag}"
    rm -rf "${overlay}" && mkdir -p "${overlay}" || return 1
    cp -r "${pkg_dir}" "${overlay}/deep_ep" || return 1
    python "${script_dir}/hybridep_combine_prob_tma_patch.py" \
      "${overlay}/deep_ep/backend/hybrid_ep_backend.cuh" >/dev/null || return 1
    export PYTHONPATH="${overlay}${PYTHONPATH:+:${PYTHONPATH}}"
  elif [ "${mode}" != "installed" ]; then
    echo "[HYBRIDEP OVERLAY] unknown mode: ${mode}" >&2
    return 1
  fi

  HYBRIDEP_OVERLAY_MODE="${mode}" HYBRIDEP_OVERLAY_RANK="${rank_tag}" python - <<'PY' || return 1
import hashlib, os
import deep_ep
pkg = os.path.dirname(os.path.abspath(deep_ep.__file__))
with open(os.path.join(pkg, "backend", "hybrid_ep_backend.cuh"), "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print(f"[HYBRIDEP OVERLAY] mode={os.environ['HYBRIDEP_OVERLAY_MODE']} rank={os.environ['HYBRIDEP_OVERLAY_RANK']} "
      f"deep_ep={pkg} header_sha256={digest}", flush=True)
PY
}

_hybridep_overlay_env "$@"
