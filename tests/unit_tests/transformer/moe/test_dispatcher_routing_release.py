# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The flex dispatcher must drop its cached router probabilities once combine has read them.

``token_probs`` / ``dispatched_probs`` carry the router's autograd graph back to the layer input.
Left on the manager until the next forward, they keep that graph alive across the optimizer step,
and under MoE recompute the checkpointed layer input plus its gradient with it: one
``[tokens, hidden]`` pair per MoE layer. Pure-CPU tests; no process group is needed.
"""

import torch

from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher, _HybridEPManager


def _manager_with_probs():
    manager = _HybridEPManager.__new__(_HybridEPManager)
    logits = torch.randn(6, 4, requires_grad=True)
    manager.token_probs = torch.sigmoid(logits)  # carries a grad_fn back to ``logits``
    manager.dispatched_probs = manager.token_probs[:, :2].contiguous()
    manager.tokens_per_expert = torch.tensor([3, 3])
    return manager, logits


def test_release_router_probs_drops_graph_carrying_tensors():
    manager, _ = _manager_with_probs()
    assert manager.token_probs.grad_fn is not None

    manager.release_router_probs()

    assert manager.token_probs is None
    assert manager.dispatched_probs is None
    # Static routing metadata survives: CUDA graph replay skips setup_metadata.
    assert manager.tokens_per_expert is not None


def test_release_router_probs_tolerates_unset_attributes():
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.release_router_probs()
    assert manager.token_probs is None and manager.dispatched_probs is None


def test_flex_combine_postprocess_releases_router_probs():
    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher._comm_manager, _ = _manager_with_probs()
    dispatcher.shared_experts = None
    dispatcher.hidden_shape = torch.Size([2, 3, 8])

    output = dispatcher.combine_postprocess(torch.zeros(6, 8))

    assert output.shape == dispatcher.hidden_shape
    assert dispatcher._comm_manager.token_probs is None
    assert dispatcher._comm_manager.dispatched_probs is None
