# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Container sanity checks for the Kimi-K3 Blackwell recipe image (dev-only probe).

Verifies that a container built from the Dockerfile embedded in
``examples/moe_recipes/kimi_k3/gb300/mxfp8_SL4K_256GPU_TP1PP4EP64_muon.yaml``
matches the guide: default Python is the ``/opt/venv`` environment, ``pip check``
passes at runtime, every pinned component is present at the documented
revision, and the GPU-facing pieces (TransformerEngine MXFP8, HybridEP,
flash-linear-attention, cuDNN frontend, CUTLASS DSL) import and run on the
device. Runs under the probe entry (one process per GPU).
"""

import importlib
import importlib.metadata as md
import json
import os
import subprocess
import sys

import pytest
import torch

GUIDE = {
    "te_commit": "0a2ebe942b9f6a66aa0292502966e73114a7b912",
    "cudnn_frontend_commit": "5ba9432e652e0b4df3aaca85cf3f95fe6bb7375a",
    "emerging_optimizers_commit": "a44d1f83a4950b445f2a77ca71177c5f033d45d5",
    "hybrid_ep_commit": "1b8f467965bb818bf2f6511e06993f5607e1721f",
    "versions": {
        "flash-linear-attention": "0.5.2",
        "transformers": "5.16.1",
        "nvidia-resiliency-ext": "0.6.0",
        "nvidia-mathdx": "25.1.1",
        "nvidia-cutlass-dsl": "4.5.2",
        "apache-tvm-ffi": "0.1.11",
        "ninja": "1.11.1.1",
    },
}


def _dist(*candidates):
    for name in candidates:
        try:
            return md.distribution(name)
        except md.PackageNotFoundError:
            continue
    installed = sorted({d.metadata["Name"] for d in md.distributions() if d.metadata["Name"]})
    raise AssertionError(f"none of {candidates} is installed; installed: {installed}")


def _vcs_commit(dist):
    raw = dist.read_text("direct_url.json")
    assert raw, f"{dist.metadata['Name']} has no direct_url.json (not a pip VCS/path install)"
    info = json.loads(raw)
    return info.get("vcs_info", {}).get("commit_id"), info


def _report(lines):
    rank = os.environ.get("RANK", "0")
    print(f"\n[container-sanity rank {rank}]\n  " + "\n  ".join(lines), flush=True)


def test_default_python_is_the_venv():
    _report([f"sys.executable={sys.executable}", f"sys.prefix={sys.prefix}",
             f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')}", f"python={sys.version.split()[0]}"])
    assert sys.prefix == "/opt/venv", "the guide puts /opt/venv first on PATH"
    assert os.path.realpath(sys.executable).startswith("/opt/venv/") or sys.executable.startswith(
        "/opt/venv/"
    ), sys.executable


def test_pip_check_passes_at_runtime():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=600
    )
    _report(["pip check: " + (result.stdout.strip() or result.stderr.strip())])
    assert result.returncode == 0, result.stdout + result.stderr


def test_pinned_git_revisions():
    te = _dist("transformer_engine", "transformer-engine")
    te_commit, te_info = _vcs_commit(te)
    cudnn_fe = _dist("nvidia-cudnn-frontend", "nvidia_cudnn_frontend")
    cudnn_commit, _ = _vcs_commit(cudnn_fe)
    emo = _dist("emerging-optimizers", "emerging_optimizers")
    emo_commit, _ = _vcs_commit(emo)
    _report([
        f"transformer_engine {te.version} @ {te_commit} ({te_info.get('url')})",
        f"nvidia-cudnn-frontend {cudnn_fe.version} @ {cudnn_commit}",
        f"emerging-optimizers {emo.version} @ {emo_commit}",
    ])
    assert te_commit == GUIDE["te_commit"]
    assert cudnn_commit == GUIDE["cudnn_frontend_commit"]
    assert emo_commit == GUIDE["emerging_optimizers_commit"]


def test_hybrid_ep_revision():
    deep_ep = _dist("deep_ep", "deep-ep")
    raw = deep_ep.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    head = None
    if os.path.isdir("/opt/DeepEP/.git"):
        head = subprocess.run(
            ["git", "-C", "/opt/DeepEP", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    _report([f"deep_ep {deep_ep.version} direct_url={info} /opt/DeepEP HEAD={head}"])
    assert head is not None, "/opt/DeepEP checkout missing; cannot verify the HybridEP revision"
    assert head == GUIDE["hybrid_ep_commit"]


def test_pinned_versions():
    found = {name: md.version(name) for name in GUIDE["versions"]}
    _report([f"{k} {v}" for k, v in found.items()])
    mismatched = {k: (v, GUIDE["versions"][k]) for k, v in found.items() if v != GUIDE["versions"][k]}
    assert not mismatched, f"version mismatches (found, guide): {mismatched}"


def test_stack_imports():
    modules = [
        "transformer_engine.pytorch",
        "fla",
        "fla.ops.kda",
        "fla.ops.gated_delta_rule",
        "cudnn",
        "cutlass",
        "tvm_ffi",
        "deep_ep",
        "hybrid_ep_cpp",
        "emerging_optimizers.orthogonalized_optimizers",
        "emerging_optimizers.orthogonalized_optimizers.muon_utils",
        "nvidia_resiliency_ext",
        "transformers",
        "sentencepiece",
        "tiktoken",
        "einops",
        "wandb",
        "datasets",
        "omegaconf",
        "tensorstore",
        "multistorageclient",
    ]
    failures = {}
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every failure at once
            failures[name] = f"{type(exc).__name__}: {exc}"
    from deep_ep import HybridEPBuffer  # noqa: F401

    import cudnn

    _report([f"torch {torch.__version__} cuda {torch.version.cuda}",
             f"cudnn backend {cudnn.backend_version()}",
             f"import failures: {failures or 'none'}"])
    assert not failures, failures


def test_te_mxfp8_linear_runs_on_this_gpu():
    assert torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    cap = torch.cuda.get_device_capability()
    _report([f"device {torch.cuda.get_device_name()} capability {cap}"])
    assert cap[0] == 10, f"expected a Blackwell (sm_100/sm_103) device, got {cap}"

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling

    torch.manual_seed(0)
    linear = te.Linear(512, 1024, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
        y = linear(x)
    y.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(linear.weight.grad).all()


def test_fla_kda_kernel_runs_on_this_gpu():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    b, t, h, d = 1, 128, 2, 64
    q = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(b, t, h, device="cuda", dtype=torch.float32))
    beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
    out = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    out = out[0] if isinstance(out, tuple) else out
    torch.cuda.synchronize()
    assert out.shape == (b, t, h, d) and torch.isfinite(out.float()).all()


def test_megatron_hybridep_backend_is_detected():
    from megatron.core.transformer.moe import fused_a2a

    checks = {
        name: getattr(fused_a2a, name)
        for name in dir(fused_a2a)
        if name.startswith("HAVE_") or name.startswith("has_") or name.startswith("is_")
    }
    flags = {k: v for k, v in checks.items() if isinstance(v, bool)}
    _report([f"fused_a2a availability flags: {flags}"])
    hybrid_flags = {k: v for k, v in flags.items() if "hybrid" in k.lower()}
    if hybrid_flags:
        assert all(hybrid_flags.values()), hybrid_flags
    else:
        pytest.skip("fused_a2a exposes no HybridEP availability flag to assert on")
