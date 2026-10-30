# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the cuDNN Kimi Delta Attention kernel backend (``kda_kernel_backend="cudnn"``).

The CPU-only tests cover the configuration knob, the kernel binding and the argument validation
of ``chunk_kda_cudnn``.  The GPU tests compare the cuDNN kernels with FLA's ``chunk_kda`` at the
kernel, module and chunkwise context-parallel level.  Agreement is measured as relative L2
(``||cudnn - fla|| / ||fla||``); the bounds widen the values observed on Blackwell (output ~6e-3,
dq/dk/dv/dbeta ~5-7e-3, dg ~9e-3, dA_log/ddt_bias up to ~1.3e-2) for headroom.
"""

import importlib
import math
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
from megatron.core.ssm.gated_delta_net import kda as kda_module
from megatron.core.ssm.gated_delta_net.cudnn_kda import (
    HAVE_CUDNN_KDA,
    chunk_kda_cudnn,
    cudnn_kda_unavailable_reason,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net.test_kda import (
    _KDA_INPUT_GRAD_ATOL,
    _build_kda,
    _make_config,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params

try:
    from fla.ops.kda import chunk_kda as fla_chunk_kda
except ImportError:  # pragma: no cover - environment dependent
    fla_chunk_kda = None

os.environ.update({"NCCL_NVLS_ENABLE": "0"})

# The cuDNN KDA kernels exist for Blackwell data-center GPUs only.  GB200 CI selection is
# marker-driven (tests/unit_tests/run_ci_test.sh) and the other platforms do not exclude the
# marker, so the CPU-only tests below still run everywhere.
pytestmark = pytest.mark.launch_on_gb200

_HEAD_DIM = 128  # Kimi's K == V head dimension; FLA's own CP backward is noisy at 64-dim heads.
_SAFE_GATE_LOWER_BOUND = -5.0
_HAVE_BLACKWELL_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

# Relative-L2 bounds, ||cudnn - fla|| / ||fla||.
_OUTPUT_TOL = 2e-2
_STATE_TOL = 2e-2
_INPUT_GRAD_TOL = 3e-2  # dq, dk, dv, dbeta and every parameter gradient downstream of them
_GATE_GRAD_TOL = 5e-2  # dg and the decay-logit projections
# A_log / dt_bias gradients are per-head reductions over every token and channel of mixed-sign
# bf16 terms; at the test sizes (a few heads, a few hundred tokens) the cancellation leaves a
# small sum whose relative error between two kernel implementations fluctuates run to run
# (observed up to ~1.3e-1 while every other tensor stays within a few 1e-2).
_GATE_PARAM_GRAD_TOL = 2e-1
_KERNEL_GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
_KERNEL_GRAD_TOLS = {
    "dq": _INPUT_GRAD_TOL,
    "dk": _INPUT_GRAD_TOL,
    "dv": _INPUT_GRAD_TOL,
    "dbeta": _INPUT_GRAD_TOL,
    "dg": _GATE_GRAD_TOL,
    "dA_log": _GATE_PARAM_GRAD_TOL,
    "ddt_bias": _GATE_PARAM_GRAD_TOL,
}

_GPU_PARITY_MARKS = [
    pytest.mark.internal,
    pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed."),
    pytest.mark.skipif(
        not HAVE_CUDNN_KDA,
        reason=f"cuDNN KDA backend unavailable: {cudnn_kda_unavailable_reason()}",
    ),
    pytest.mark.skipif(
        not _HAVE_BLACKWELL_GPU, reason="cuDNN KDA kernels need a Blackwell (SM100+) GPU."
    ),
]
_CP2_MARKS = _GPU_PARITY_MARKS + [
    pytest.mark.skipif(
        Utils.world_size < 2, reason="chunkwise CP2 parity needs at least two distributed ranks."
    )
]


def _apply_marks(marks):
    """Stack a list of pytest marks on one test function."""

    def decorate(test):
        for mark in reversed(marks):
            test = mark(test)
        return test

    return decorate


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    norm = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return (diff / norm).item()


def _assert_relative_l2(actual: torch.Tensor, expected: torch.Tensor, tol: float, name: str):
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), f"{name}: the cuDNN result has non-finite values"
    relative_l2 = _relative_l2(actual, expected)
    assert relative_l2 <= tol, f"{name}: relative L2 {relative_l2:.3e} > {tol:.1e}"


def _param_grad_tol(name: str) -> float:
    """Gradients derived from dg (gate parameters, decay-logit projections) get looser bounds."""
    if name in ("A_log", "dt_bias"):
        return _GATE_PARAM_GRAD_TOL
    if name.startswith(("f_proj.", "f_a_proj.", "f_b_proj.")):
        return _GATE_GRAD_TOL
    return _INPUT_GRAD_TOL


def _make_backend_config(
    backend: str,
    *,
    cp_size: int = 1,
    linear_cp_mode: str = "headwise",
    f_lora_rank: int | None = None,
    gate_lora_rank: int | None = None,
    **overrides,
):
    """``test_kda._make_config`` with Kimi's 128-dim heads and the requested kernel backend.

    ``replace`` carries the resolved ``linear_cp_layout`` ("zigzag") over as an explicit setting,
    which chunkwise CP rejects, so the contiguous layout is pinned whenever chunkwise is requested
    (as ``test_kda_fusion`` does).
    """
    if linear_cp_mode == "chunkwise":
        overrides.setdefault("linear_cp_layout", "contiguous")
    return replace(
        _make_config(
            cp_size=cp_size,
            linear_cp_mode=linear_cp_mode,
            f_lora_rank=f_lora_rank,
            gate_lora_rank=gate_lora_rank,
        ),
        linear_key_head_dim=_HEAD_DIM,
        linear_value_head_dim=_HEAD_DIM,
        kda_kernel_backend=backend,
        **overrides,
    )


def _make_kernel_inputs(batch: int, seq_len: int, heads: int, head_dim: int, *, seed: int, device):
    """Operands as ``KimiDeltaAttention`` hands them to ``self.gated_delta_rule``.

    q/k are L2-normalised bf16, v is bf16, g holds the raw bf16 gate logits, beta the fp32
    post-sigmoid weights, ``A_log`` is fp32 ``[H]`` from the module's ``A_init_range`` and
    ``dt_bias`` fp32 ``[H * D]`` in inverse-softplus space like ``_reset_dt_bias``.  Everything is
    drawn on the host from one generator so both backends see identical bits.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    shape = (batch, seq_len, heads, head_dim)
    dt_min, dt_max = 1e-3, 1e-1
    dt = torch.exp(
        torch.rand(heads * head_dim, generator=generator) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)
    inputs = {
        "q": F.normalize(randn(*shape), dim=-1).to(torch.bfloat16),
        "k": F.normalize(randn(*shape), dim=-1).to(torch.bfloat16),
        "v": randn(*shape).to(torch.bfloat16),
        "g": randn(*shape).to(torch.bfloat16),
        "beta": torch.sigmoid(randn(batch, seq_len, heads)),
        "A_log": torch.log(torch.empty(heads).uniform_(1.0, 16.0, generator=generator)),
        "dt_bias": dt + torch.log(-torch.expm1(-dt)),
    }
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _make_grad_output(shape, *, seed: int, device):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).to(device)


def _run_kernel(kernel, inputs, grad_output, **kwargs):
    """Forward + backward of a ``chunk_kda``-compatible kernel on fresh leaf copies of inputs."""
    leaves = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    output, final_state = kernel(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=_SAFE_GATE_LOWER_BOUND,
        **kwargs,
    )
    grads = torch.autograd.grad(
        output, list(leaves.values()), grad_outputs=grad_output.to(output.dtype)
    )
    return output.detach(), final_state, dict(zip(_KERNEL_GRAD_NAMES, grads))


def _compare_kernels(inputs, grad_output, **kwargs):
    """Run FLA and cuDNN on ``inputs`` and check the output and the seven gradients agree."""
    fla_output, fla_state, fla_grads = _run_kernel(fla_chunk_kda, inputs, grad_output, **kwargs)
    cudnn_output, cudnn_state, cudnn_grads = _run_kernel(
        chunk_kda_cudnn, inputs, grad_output, **kwargs
    )
    assert cudnn_output.dtype == fla_output.dtype
    _assert_relative_l2(cudnn_output, fla_output, _OUTPUT_TOL, "output")
    for name, tol in _KERNEL_GRAD_TOLS.items():
        _assert_relative_l2(cudnn_grads[name], fla_grads[name], tol, name)
    return (fla_output, fla_state), (cudnn_output, cudnn_state)


def _compare_module_forward_backward(
    reference, candidate, hidden_states, make_packed_seq_params=None, cp_group=None
):
    """Feed the FLA (reference) and cuDNN (candidate) modules the same input.

    Compares the output, the input gradient and every parameter gradient.
    ``make_packed_seq_params`` builds fresh THD metadata per call because the module's CP layout
    conversion mutates it.  Under CP the parameter gradients are summed over ``cp_group`` first,
    as the model's gradient reduction would do.
    """
    reference_input = hidden_states.detach().clone().requires_grad_(True)
    candidate_input = hidden_states.detach().clone().requires_grad_(True)
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)

    def run(module, module_input):
        packed_seq_params = make_packed_seq_params() if make_packed_seq_params is not None else None
        output, bias = module(module_input, None, packed_seq_params=packed_seq_params)
        assert bias is None
        grad_output = torch.linspace(
            -0.1, 0.1, output.numel(), device=output.device, dtype=torch.float32
        ).reshape(output.shape)
        (output.float() * grad_output).sum().backward()
        if cp_group is not None:
            torch.distributed.barrier()
        return output.detach()

    reference_output = run(reference, reference_input)
    candidate_output = run(candidate, candidate_input)

    _assert_relative_l2(candidate_output, reference_output, _OUTPUT_TOL, "output")
    _assert_relative_l2(candidate_input.grad, reference_input.grad, _INPUT_GRAD_TOL, "input grad")
    reference_params = dict(reference.named_parameters())
    candidate_params = dict(candidate.named_parameters())
    assert candidate_params.keys() == reference_params.keys()
    for name, reference_param in reference_params.items():
        reference_grad, candidate_grad = reference_param.grad, candidate_params[name].grad
        assert (candidate_grad is None) == (reference_grad is None), name
        if reference_grad is None:
            continue
        reference_grad = reference_grad.detach().float().clone()
        candidate_grad = candidate_grad.detach().float().clone()
        if cp_group is not None:
            torch.distributed.all_reduce(reference_grad, group=cp_group)
            torch.distributed.all_reduce(candidate_grad, group=cp_group)
        _assert_relative_l2(candidate_grad, reference_grad, _param_grad_tol(name), f"grad[{name}]")


def _zigzag_shard_sbhd(tensor: torch.Tensor, cp_group) -> torch.Tensor:
    """This rank's zigzag shard of a global ``[S, B, H]`` tensor.

    Same slicing as ``_test_parallel_attention_correctness``; the KDA module converts it to the
    contiguous chunkwise layout itself.
    """
    cp_size, cp_rank = cp_group.size(), torch.distributed.get_rank(cp_group)
    chunks = tensor.view(2 * cp_size, -1, *tensor.shape[1:])
    index = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=tensor.device)
    return chunks.index_select(0, index).reshape(-1, *tensor.shape[1:])


def _zigzag_shard_thd(tensor: torch.Tensor, cu_seqlens: list[int], cp_group) -> torch.Tensor:
    """This rank's zigzag shard of a packed ``[T, 1, H]`` batch with global ``cu_seqlens``."""
    batch = {
        "hidden_states": tensor,
        "cu_seqlens_padded": torch.tensor(cu_seqlens, device=tensor.device),
    }
    get_cp_slice_for_thd(batch, cp_group, keys=("hidden_states",), cp_partition_mode="zigzag")
    return batch["hidden_states"]


# --------------------------------------------------------------------------------------------
# CPU-only: package import, configuration knob, kernel binding, argument validation.
# --------------------------------------------------------------------------------------------


def test_cudnn_kda_package_imports_without_cudnn_frontend():
    """The package loads with or without cudnn-frontend and reports availability consistently."""
    package = importlib.import_module("megatron.core.ssm.gated_delta_net.cudnn_kda")
    assert package.chunk_kda_cudnn is chunk_kda_cudnn
    assert callable(chunk_kda_cudnn)
    assert isinstance(HAVE_CUDNN_KDA, bool)
    reason = cudnn_kda_unavailable_reason()
    if HAVE_CUDNN_KDA:
        assert reason is None
    else:
        assert isinstance(reason, str) and reason
    assert {"HAVE_CUDNN_KDA", "chunk_kda_cudnn", "cudnn_kda_unavailable_reason"} <= set(
        package.__all__
    )


def test_kda_kernel_backend_defaults_to_fla():
    assert _make_config().kda_kernel_backend == "fla"
    assert replace(_make_config(), kda_kernel_backend="cudnn").kda_kernel_backend == "cudnn"


def test_kda_kernel_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="kda_kernel_backend must be 'fla' or 'cudnn'"):
        replace(_make_config(), kda_kernel_backend="triton")


def test_kda_cudnn_backend_rejects_deterministic_mode():
    with pytest.raises(ValueError, match="kda_kernel_backend='cudnn' is not deterministic"):
        replace(_make_config(), kda_kernel_backend="cudnn", deterministic_mode=True)
    # The FLA backend keeps supporting deterministic mode.
    config = replace(_make_config(), kda_kernel_backend="fla", deterministic_mode=True)
    assert config.deterministic_mode


def test_select_kda_kernel_binds_backend():
    """``_select_kda_kernel`` reads only the config, so a module shell suffices (no GPU)."""
    fla_shell = SimpleNamespace(config=replace(_make_config(), kda_kernel_backend="fla"))
    assert KimiDeltaAttention._select_kda_kernel(fla_shell) is kda_module.chunk_kda

    cudnn_shell = SimpleNamespace(config=replace(_make_config(), kda_kernel_backend="cudnn"))
    if not HAVE_CUDNN_KDA:
        with pytest.raises(ImportError, match="cudnn-frontend"):
            KimiDeltaAttention._select_kda_kernel(cudnn_shell)
    else:
        from megatron.core.ssm.gated_delta_net.cudnn_kda.ops import device_unavailable_reason

        if device_unavailable_reason() is None:
            assert KimiDeltaAttention._select_kda_kernel(cudnn_shell) is chunk_kda_cudnn
        else:
            # cudnn-frontend is installed but this GPU / CuTe DSL cannot run its KDA engines.
            with pytest.raises(RuntimeError, match="cannot run here"):
                KimiDeltaAttention._select_kda_kernel(cudnn_shell)

    # The binding mirrors TransformerConfig's deterministic-mode rejection for hybrid stacks.
    det_shell = SimpleNamespace(
        config=SimpleNamespace(kda_kernel_backend="cudnn", deterministic_mode=True)
    )
    with pytest.raises(ValueError, match="deterministic"):
        KimiDeltaAttention._select_kda_kernel(det_shell)

    # TransformerConfig rejects unknown backends; the binding guards against a bypassed config.
    bogus_shell = SimpleNamespace(config=SimpleNamespace(kda_kernel_backend="triton"))
    with pytest.raises(ValueError, match="Unsupported kda_kernel_backend"):
        KimiDeltaAttention._select_kda_kernel(bogus_shell)


def test_chunk_kda_cudnn_validates_arguments_on_cpu():
    """Argument validation precedes any cudnn-frontend use, so it is exercised on CPU tensors."""
    batch, seq_len, heads, head_dim = 2, 8, 2, 16
    q = torch.zeros(batch, seq_len, heads, head_dim, dtype=torch.bfloat16)
    beta = torch.full((batch, seq_len, heads), 0.5)
    gate_kwargs = dict(
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=_SAFE_GATE_LOWER_BOUND,
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(heads * head_dim),
    )

    with pytest.raises(NotImplementedError, match="unsupported arguments"):
        chunk_kda_cudnn(q, q, q, q, beta, return_intermediate_states=True, **gate_kwargs)
    with pytest.raises(ValueError, match=r"\[B, T, H, D\]"):
        chunk_kda_cudnn(q[0], q[0], q[0], q[0], beta[0], **gate_kwargs)
    with pytest.raises(TypeError, match="bf16/fp16"):
        chunk_kda_cudnn(q.float(), q.float(), q.float(), q.float(), beta, **gate_kwargs)
    with pytest.raises(ValueError, match="B == 1"):
        cu_seqlens = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32)
        chunk_kda_cudnn(q, q, q, q, beta, cu_seqlens=cu_seqlens, **gate_kwargs)
    if not HAVE_CUDNN_KDA:
        with pytest.raises(ImportError, match="cuDNN KDA backend unavailable"):
            chunk_kda_cudnn(q, q, q, q, beta, **gate_kwargs)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="builds the module on a CUDA device inside an NCCL process group.",
)
@pytest.mark.skipif(
    HAVE_CUDNN_KDA,
    reason="cudnn-frontend with KDA operators is installed; ImportError path unreachable",
)
def test_kda_module_requires_cudnn_frontend_for_cudnn_backend():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        with pytest.raises(ImportError, match="cudnn-frontend"):
            _build_kda(replace(_make_config(), kda_kernel_backend="cudnn"))
    finally:
        Utils.destroy_model_parallel()


# --------------------------------------------------------------------------------------------
# GPU: kernel-level parity with FLA (no CP).
# --------------------------------------------------------------------------------------------


class TestChunkKdaCudnnKernelParity:
    """``chunk_kda_cudnn`` against ``fla.ops.kda.chunk_kda`` on identical operands."""

    pytestmark = _GPU_PARITY_MARKS

    def test_sbhd_forward_backward_matches_fla(self):
        device = torch.cuda.current_device()
        inputs = _make_kernel_inputs(2, 512, 4, _HEAD_DIM, seed=1234, device=device)
        grad_output = _make_grad_output(inputs["v"].shape, seed=4321, device=device)
        _compare_kernels(
            inputs,
            grad_output,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            cp_context=None,
        )

    def test_thd_forward_backward_matches_fla(self):
        device = torch.cuda.current_device()
        cu_seqlens = torch.tensor([0, 300, 812, 1024], dtype=torch.int32, device=device)
        inputs = _make_kernel_inputs(1, 1024, 4, _HEAD_DIM, seed=2345, device=device)
        grad_output = _make_grad_output(inputs["v"].shape, seed=5432, device=device)
        _compare_kernels(
            inputs,
            grad_output,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            cp_context=None,
        )

    def test_final_state_matches_fla(self):
        device = torch.cuda.current_device()
        batch, heads = 2, 4
        inputs = _make_kernel_inputs(batch, 256, heads, _HEAD_DIM, seed=3456, device=device)
        grad_output = _make_grad_output(inputs["v"].shape, seed=6543, device=device)
        (_, fla_state), (_, cudnn_state) = _compare_kernels(
            inputs,
            grad_output,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=None,
            cp_context=None,
        )
        assert fla_state is not None and cudnn_state is not None
        # FLA layout [N, H, K, V] in fp32.
        assert tuple(fla_state.shape) == (batch, heads, _HEAD_DIM, _HEAD_DIM)
        assert cudnn_state.dtype == torch.float32
        _assert_relative_l2(cudnn_state, fla_state, _STATE_TOL, "final_state")

    def test_initial_state_passthrough_matches_fla(self):
        device = torch.cuda.current_device()
        batch, heads = 2, 4
        inputs = _make_kernel_inputs(batch, 256, heads, _HEAD_DIM, seed=4567, device=device)
        grad_output = _make_grad_output(inputs["v"].shape, seed=7654, device=device)
        initial_state = torch.randn(
            batch, heads, _HEAD_DIM, _HEAD_DIM, generator=torch.Generator().manual_seed(8765)
        ).to(device)
        common = dict(output_final_state=False, cu_seqlens=None, cp_context=None)

        fla_output, _, _ = _run_kernel(
            fla_chunk_kda, inputs, grad_output, initial_state=initial_state.clone(), **common
        )
        cudnn_output, _, _ = _run_kernel(
            chunk_kda_cudnn, inputs, grad_output, initial_state=initial_state.clone(), **common
        )
        _assert_relative_l2(cudnn_output, fla_output, _OUTPUT_TOL, "output with initial_state")

        # The state must visibly shape the output, otherwise the comparison above says nothing.
        fla_no_state, _, _ = _run_kernel(
            fla_chunk_kda, inputs, grad_output, initial_state=None, **common
        )
        assert _relative_l2(fla_no_state, fla_output) > _OUTPUT_TOL


# --------------------------------------------------------------------------------------------
# GPU: module-level parity with the FLA backend (CP1).
# --------------------------------------------------------------------------------------------


class TestKdaCudnnModuleParity:
    """``KimiDeltaAttention`` with ``kda_kernel_backend="cudnn"`` against the FLA backend."""

    pytestmark = _GPU_PARITY_MARKS

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_pair(**config_kwargs):
        reference = _build_kda(_make_backend_config("fla", **config_kwargs))
        candidate = _build_kda(_make_backend_config("cudnn", **config_kwargs))
        candidate.load_state_dict(reference.state_dict())
        assert reference.gated_delta_rule is kda_module.chunk_kda
        assert candidate.gated_delta_rule is chunk_kda_cudnn
        return reference, candidate

    @pytest.mark.parametrize(
        ("f_lora_rank", "gate_lora_rank"),
        [(None, None), (8, 12)],
        ids=["legacy-fused", "kimi-low-rank"],
    )
    def test_sbhd_forward_backward_matches_fla(self, f_lora_rank, gate_lora_rank):
        reference, candidate = self._build_pair(
            f_lora_rank=f_lora_rank, gate_lora_rank=gate_lora_rank
        )
        torch.manual_seed(42)
        hidden_states = torch.randn(
            (512, 2, reference.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        _compare_module_forward_backward(reference, candidate, hidden_states)

    def test_thd_forward_backward_matches_fla(self):
        reference, candidate = self._build_pair(f_lora_rank=8, gate_lora_rank=12)
        torch.manual_seed(42)
        hidden_states = torch.randn(
            (512, 1, reference.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        _compare_module_forward_backward(
            reference,
            candidate,
            hidden_states,
            make_packed_seq_params=lambda: make_test_packed_seq_params(
                cu_seqlens=[0, 96, 245, 512]
            ),
        )


# --------------------------------------------------------------------------------------------
# GPU, two ranks: chunkwise context parallelism.
# --------------------------------------------------------------------------------------------


class TestKdaCudnnChunkwiseCP:
    """cuDNN against the FLA backend inside one chunkwise CP2 group.

    Both modules hold identical (replicated, TP=1) weights and receive this rank's zigzag shard
    of the same global sequence; the module converts the shard to the contiguous chunkwise
    layout itself, as it does under ``_test_parallel_attention_correctness``.  After that
    conversion the single SBHD sequence and the 384-token THD document both cross the rank
    boundary, so cuDNN's per-span summary exchange is exercised in forward and backward.
    """

    pytestmark = _CP2_MARKS

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        yield
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_pair():
        config_kwargs = dict(
            cp_size=2, linear_cp_mode="chunkwise", f_lora_rank=8, gate_lora_rank=12
        )
        reference = _build_kda(_make_backend_config("fla", **config_kwargs))
        candidate = _build_kda(_make_backend_config("cudnn", **config_kwargs))
        candidate.load_state_dict(reference.state_dict())
        assert candidate.config.linear_cp_layout == "contiguous"
        assert candidate.gated_delta_rule is chunk_kda_cudnn
        return reference, candidate

    def test_sbhd_matches_fla_backend(self):
        reference, candidate = self._build_pair()
        cp_group = parallel_state.get_context_parallel_group()
        torch.manual_seed(2026)
        # SBHD chunkwise CP requires micro_batch_size == 1.
        global_hidden_states = torch.randn(
            (256, 1, reference.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        hidden_states = _zigzag_shard_sbhd(global_hidden_states, cp_group)
        _compare_module_forward_backward(reference, candidate, hidden_states, cp_group=cp_group)

    def test_thd_matches_fla_backend(self):
        reference, candidate = self._build_pair()
        cp_group = parallel_state.get_context_parallel_group()
        # Per-sequence lengths divisible by 2 * cp_size.  The first document (384 tokens) spans
        # both 256-token ranks after the contiguous re-partition; sizes stay large enough that
        # the per-head gate-parameter gradients (a sum over all tokens and channels) are not
        # dominated by bf16 cancellation noise, which at 96 + 32 tokens exceeded the tolerance.
        cu_seqlens = [0, 384, 512]
        torch.manual_seed(2026)
        global_hidden_states = torch.randn(
            (cu_seqlens[-1], 1, reference.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        hidden_states = _zigzag_shard_thd(global_hidden_states, cu_seqlens, cp_group)
        _compare_module_forward_backward(
            reference,
            candidate,
            hidden_states,
            make_packed_seq_params=lambda: make_test_packed_seq_params(cu_seqlens=cu_seqlens),
            cp_group=cp_group,
        )


@_apply_marks(_CP2_MARKS)
@pytest.mark.parametrize(
    ("sequence_packing", "micro_batch_size"), [(True, 3), (False, 1)], ids=["thd", "sbhd-mbs1"]
)
def test_cudnn_chunkwise_cp_matches_single_rank(
    tmp_path_dist_ckpt, sequence_packing, micro_batch_size, run_config
):
    """CP2 chunkwise cuDNN against the CP1 cuDNN reference (``test_kda_fusion``'s recipe).

    SBHD chunkwise CP requires ``micro_batch_size == 1``.  The THD case packs three 128-token
    sequences so that, after the contiguous re-partition (192 tokens per rank), the middle
    sequence spans both ranks and the cross-rank summary / fold path of ``cp.py`` is exercised
    (two sequences would give each rank one whole sequence and only zero summaries).
    """
    config = _make_backend_config(
        "cudnn", linear_cp_mode="chunkwise", f_lora_rank=8, gate_lora_rank=12
    )
    _test_parallel_attention_correctness(
        transformer_config=config,
        transformer_layer_spec=hybrid_stack_spec.submodules.kda_layer,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=1e-2,
        rtol=1e-2,
        input_grad_atol=_KDA_INPUT_GRAD_ATOL,
        input_grad_rtol=1e-2,
        cosine_similarity_threshold=0.9999,
        compare_param_grads=True,
        tp=1,
        cp=2,
        seed=42,
        sequence_length=128,
        micro_batch_size=micro_batch_size,
        sequence_packing=sequence_packing,
    )
