# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Switch a running engine between prefill and decode all2all backends.

A prefill-role engine and a decode-role engine of the same model differ in
launch-time configuration, not in weights. The expensive difference is the MoE
all2all backend: prefill wants ``deepep_high_throughput`` (bulk dispatch,
Standard activation format), decode wants ``deepep_low_latency`` (per-rank
batched dispatch, BatchedExperts format).

Running one engine per role costs a second copy of the weights, which for a
large MoE does not fit: 49 GiB/rank of weights needs 98 GiB on an 80 GiB card.
Restarting the engine to change roles costs a full load -- measured at ~358s
for GLM-5.3 on H100.

This module changes the role in place. Weights are never read, written, moved
or re-shuffled; only the communication path around them is rebuilt.

What actually differs between the two backends
----------------------------------------------
Three things, and only the first is obvious:

1. The all2all manager on the EP group's device communicator
   (``DeepEPHTAll2AllManager`` vs ``DeepEPLLAll2AllManager``). Both allocate a
   symmetric NVSHMEM buffer. Measured on H100: HT 1264 MiB, LL 3226 MiB, and
   the two coexist on one GPU without interfering -- which is what makes
   keeping both alive affordable.

2. The per-layer ``prepare_finalize``, rebuilt from the manager.

3. The per-layer *experts*, which are NOT the same class. HT reports
   ``FusedMoEActivationFormat.Standard`` and LL reports ``BatchedExperts``, and
   ``use_batched_activation_format`` is derived straight from
   ``use_deepep_ll_kernels``. So the switch selects e.g. ``DeepGemmExperts``
   one way and ``BatchedDeepGemmExperts`` the other.

Point 3 is why this cannot reuse ``eep_reconfigure.make_eep_staged_quant_method``:
that path deliberately reuses ``source_experts.__class__`` and asserts the
batched format is unchanged, because elastic EP changes the world size and not
the backend.

Why the weights survive a change of experts class
-------------------------------------------------
Because the batched and standard variants of one backend share a weight layout.
``convert_to_fp8_moe_kernel_format`` maps ``DEEPGEMM`` and ``BATCHED_DEEPGEMM``
to the same ``prepare_fp8_moe_layer_for_deepgemm`` call, and ``TRITON`` /
``BATCHED_TRITON`` and ``VLLM_CUTLASS`` / ``BATCHED_VLLM_CUTLASS`` alike fall
through to no conversion at all. The pairs differ in how tokens arrive, not in
how weights are stored. Verified per quant method by
``rebuild_moe_kernel``, which re-selects the experts class and re-runs only the
kernel construction -- never ``convert_to_*_kernel_format``.

Hidden size is a real constraint, checked not assumed
-----------------------------------------------------
The two backends round layer hidden size up differently: HT to a 512-byte
transfer atom, LL to a fixed list of supported sizes. Rounding happens once, at
load, for the backend the engine started with, so a model whose hidden size
needs *different* padding for the two backends cannot switch -- its parameters
have the wrong trailing dimension for the other side. ``check_switchable``
rejects that case with the two sizes named, rather than letting it surface as a
shape error mid-forward.

GLM-5.3 (hidden_size 6144) needs no padding for either: 6144 is in LL's
supported list, and 6144 x 2 bytes is a multiple of 512 for HT. That is a
property of the model, not a general guarantee.

Usage
-----
Driven from a worker extension so it runs in the process that owns the model::

    class RoleSwitcher:
        def switch_role(self, backend: str) -> int:
            from vllm.model_executor.layers.fused_moe.pd_role_switch import (
                switch_all2all_backend,
            )
            return switch_all2all_backend(self.model_runner.model, backend)

    vllm serve ... --worker-extension-cls ext.RoleSwitcher
    curl -X POST .../collective_rpc \\
         -d '{"method":"switch_role","args":["deepep_low_latency"]}'

Every rank must switch, and no request may be in flight while it happens: the
dispatch path is torn down and rebuilt, so a forward pass spanning the switch
would use half of each backend.
"""

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

logger = init_logger(__name__)

# Only these two are role-switchable today. Both are DeepEP, so they share a
# manager constructor signature and a symmetric-buffer model; the flashinfer
# and mori backends differ enough that claiming support without measuring it
# would be guessing.
HIGH_THROUGHPUT = "deepep_high_throughput"
LOW_LATENCY = "deepep_low_latency"
SWITCHABLE_BACKENDS = (HIGH_THROUGHPUT, LOW_LATENCY)


class RoleSwitchError(RuntimeError):
    """The engine cannot switch backends. Raised before anything is mutated."""


def _rounded_hidden_size(hidden_size: int, dtype: torch.dtype, backend: str) -> int:
    """Hidden size ``backend`` would have padded this layer to at load time."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        DeepEPHTPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        DeepEPLLPrepareAndFinalize,
    )

    if backend == HIGH_THROUGHPUT:
        return DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, dtype
        )
    return DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(hidden_size)


def _moe_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Every MoE layer in ``model``, identified by shape rather than class.

    Duck-typed on purpose: the attributes below are what the switch actually
    needs, and matching on them keeps this working across the several layer
    classes that carry a ``FusedMoEConfig``.
    """
    layers = []
    for module in model.modules():
        moe_config = getattr(module, "moe_config", None)
        if not isinstance(moe_config, FusedMoEConfig):
            continue
        if not hasattr(module, "_quant_method"):
            continue
        layers.append(module)
    return layers


def check_switchable(model: torch.nn.Module, backend: str) -> list[torch.nn.Module]:
    """Validate the switch and return the layers it would touch.

    Raises before any mutation. A half-switched model has no way back short of
    a restart, so every reason to refuse is collected here rather than
    discovered layer by layer.
    """
    if backend not in SWITCHABLE_BACKENDS:
        raise RoleSwitchError(
            f"Cannot switch to {backend!r}; supported: {list(SWITCHABLE_BACKENDS)}."
        )

    layers = _moe_layers(model)
    if not layers:
        raise RoleSwitchError(
            "No MoE layers found. Role switching only applies to MoE models "
            "using an all2all backend."
        )

    current = layers[0].moe_config.moe_parallel_config.all2all_backend
    if current not in SWITCHABLE_BACKENDS:
        raise RoleSwitchError(
            f"Engine started on all2all backend {current!r}, which is not "
            f"role-switchable. Start it on one of {list(SWITCHABLE_BACKENDS)}."
        )

    for layer in layers:
        moe = layer.moe_config
        unpadded = getattr(moe, "hidden_dim_unpadded", None) or moe.hidden_dim
        here = _rounded_hidden_size(unpadded, moe.in_dtype, current)
        there = _rounded_hidden_size(unpadded, moe.in_dtype, backend)
        if here != there:
            raise RoleSwitchError(
                f"Hidden size padding differs between backends: {current} "
                f"pads {unpadded} to {here}, {backend} pads it to {there}. "
                f"Weights were laid out for {here} at load time, so this model "
                f"cannot switch roles in place."
            )

        quant_method = layer._quant_method
        if not hasattr(quant_method, "rebuild_moe_kernel"):
            raise RoleSwitchError(
                f"{type(quant_method).__name__} does not implement "
                f"rebuild_moe_kernel(), so its MoE kernel cannot be rebuilt "
                f"for a different activation format."
            )

    return layers


def _swap_all2all_manager(backend: str):
    """Point the EP group's device communicator at ``backend``'s manager.

    The outgoing manager is returned, still alive and still holding its
    symmetric buffer. Destroying it would make switching back cost a fresh
    NVSHMEM allocation and rendezvous; keeping it costs ~1.3 GiB (HT) or
    ~3.2 GiB (LL), which is the whole point of preferring this over a restart.
    """
    from vllm.distributed.device_communicators.all2all import (
        DeepEPHTAll2AllManager,
        DeepEPLLAll2AllManager,
    )

    device_communicator = get_ep_group().device_communicator
    assert device_communicator is not None

    previous = device_communicator.all2all_manager
    cached = getattr(device_communicator, "_role_switch_managers", None)
    if cached is None:
        cached = {}
        device_communicator._role_switch_managers = cached
    # The manager the engine booted with was never put in the cache by us.
    cached.setdefault(device_communicator.all2all_backend, previous)

    manager = cached.get(backend)
    if manager is None:
        manager_cls = (
            DeepEPHTAll2AllManager
            if backend == HIGH_THROUGHPUT
            else DeepEPLLAll2AllManager
        )
        # All2AllManagerBase keeps the group it was constructed with, so the
        # manager already running is the source for the one being built. It
        # only decides self.internode, but reading it back beats passing None:
        # on a multi-node pool cpu_group and tcp_store_group disagree, and the
        # new manager must reach the same verdict as the old one.
        tcp_store_group = getattr(previous, "tcp_store_group", None)
        manager = manager_cls(device_communicator.cpu_group, tcp_store_group)
        cached[backend] = manager
        logger.info("role switch: built %s", manager_cls.__name__)

    device_communicator.all2all_manager = manager
    device_communicator.all2all_backend = backend
    return previous


def switch_all2all_backend(model: torch.nn.Module, backend: str) -> int:
    """Move this engine to ``backend``. Returns the number of layers switched.

    Call on every rank, with no requests in flight.
    """
    layers = check_switchable(model, backend)
    current = layers[0].moe_config.moe_parallel_config.all2all_backend
    if current == backend:
        logger.info("role switch: already on %s, nothing to do", backend)
        return 0

    _swap_all2all_manager(backend)

    # Flip the engine-wide config too, so anything constructed after this point
    # agrees with the layers that were just rebuilt.
    # _or_none: get_current_vllm_config() raises outside a
    # set_current_vllm_config() context, and the switch is driven from a
    # worker RPC that has no reason to be inside one.
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        vllm_config.parallel_config.all2all_backend = backend

    switched = 0
    for layer in layers:
        # One assignment drives the whole derivation: use_deepep_ht_kernels,
        # use_deepep_ll_kernels and use_batched_activation_format are all
        # read-only properties over this field. Mutated in place because the
        # same config object is shared by the layer, its routed experts and
        # its quant method.
        layer.moe_config.moe_parallel_config.all2all_backend = backend
        layer._quant_method.rebuild_moe_kernel(layer)
        switched += 1

    logger.info(
        "role switch: %s -> %s across %d MoE layers, weights untouched",
        current,
        backend,
        switched,
    )
    return switched
