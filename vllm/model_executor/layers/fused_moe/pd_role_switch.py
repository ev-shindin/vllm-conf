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

import os

import torch

from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
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


def ll_max_tokens_cap() -> int:
    """Largest per-rank dispatch bound DeepEP LL will accept.

    Two constraints, both of which fail as bare assertions deep inside a
    dispatch rather than at construction:

    * ``nvshmem_qp_depth >= (n + 1) * 2``, and the depth is read once from
      NVSHMEM_QP_DEPTH (default 1024) when deep_ep builds its first buffer,
      so it is fixed at engine boot and cannot be raised by a later switch.
    * ``(num_ranks * n) % 4 == 0``, for TMA. Keeping n itself a multiple of
      4 satisfies this whatever num_ranks turns out to be, so the rule does
      not need to know the EP world size.

    Read from the same environment variable deep_ep reads, so the two agree
    by construction rather than by a constant copied here that could drift.
    """
    depth = int(os.environ.get("NVSHMEM_QP_DEPTH", "1024"))
    return ((depth // 2 - 1) // 4) * 4


def _scheduler_token_budget(config) -> int | None:
    """The scheduler's per-step token budget, from an explicit config.

    Takes the config rather than reaching for the ambient one: this is
    called from check_switchable, which runs before the caller establishes
    a set_current_vllm_config context, so the ambient lookup returns None
    and the check quietly passes everything.
    """
    if config is None or config.scheduler_config is None:
        return None
    return config.scheduler_config.max_num_batched_tokens


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


def check_switchable(
    model: torch.nn.Module,
    backend: str,
    max_num_tokens: int | None = None,
    config=None,
) -> list[torch.nn.Module]:
    """Validate the switch and return the layers it would touch.

    Raises before any mutation. A half-switched model has no way back short of
    a restart, so every reason to refuse is collected here rather than
    discovered layer by layer.
    """
    if backend == LOW_LATENCY and max_num_tokens is not None:
        cap = ll_max_tokens_cap()
        scheduled = _scheduler_token_budget(config)
        if scheduled is not None and max_num_tokens < scheduled:
            raise RoleSwitchError(
                f"max_num_tokens={max_num_tokens} is below the scheduler's "
                f"budget of {scheduled}, so the scheduler could dispatch more "
                f"tokens than the buffer holds. Lowering it here is not "
                f"enough: v1 Scheduler caches max_num_scheduled_tokens at "
                f"init and runs in EngineCore, not in this worker. Lower "
                f"max_num_batched_tokens for the decode role there first."
            )
        if max_num_tokens % 4 != 0:
            raise RoleSwitchError(
                f"max_num_tokens={max_num_tokens} must be a multiple of 4: "
                f"DeepEP LL dispatch asserts (num_ranks * n) % 4 == 0 for TMA."
            )
        if max_num_tokens > cap:
            needed = (max_num_tokens + 1) * 2
            raise RoleSwitchError(
                f"max_num_tokens={max_num_tokens} exceeds the DeepEP LL cap "
                f"of {cap}. Beyond it every dispatch fails an assert inside "
                f"deep_ep. Set NVSHMEM_QP_DEPTH>={needed} in the environment "
                f"BEFORE the engine starts -- deep_ep reads it once, when it "
                f"builds its first buffer -- or lower max_num_batched_tokens. "
                f"Clamping here is not an option: a bound below what the "
                f"scheduler may dispatch is what the assert is protecting "
                f"against."
            )

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


def _swap_all2all_manager(backend: str, keep_previous: bool):
    """Point the EP group's device communicator at ``backend``'s manager.

    The outgoing manager is returned so the caller can destroy it once every
    layer has been rebuilt against the new one. Keeping it makes switching
    back free of a fresh NVSHMEM rendezvous, but the buffer scales with
    tokens x hidden x experts -- ~3.2 GiB on two GPUs, ~13 GiB at a large
    MoE's shape -- so holding it is only worth it when it is small.
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

    if not keep_previous:
        # Do not leave a manager in the cache that the caller is about to
        # destroy; a later switch back must build a fresh one.
        cached.pop(device_communicator.all2all_backend, None)

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


def switch_all2all_backend(
    model: torch.nn.Module,
    backend: str,
    vllm_config=None,
    max_num_tokens: int | None = None,
    keep_previous: bool = False,
) -> int:
    """Move this engine to ``backend``. Returns the number of layers switched.

    Call on every rank, with no requests in flight.

    ``vllm_config`` is required when there is no ambient config, which is the
    normal case here: the switch is driven from a worker RPC, long after
    model init left its ``set_current_vllm_config`` context. Parts of the
    rebuild path still call ``get_current_vllm_config()`` unconditionally
    (all2all_utils reads scheduler_config for some backends), so the rebuild
    runs inside a restored context rather than hoping none of them fire.

    ``max_num_tokens`` is the per-rank dispatch bound for the new role, and
    moving it is not optional when switching to low latency: DeepEP LL sizes
    its queue pairs from it and asserts
    ``nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2`` on
    every dispatch. Leaving an engine's prefill-sized budget in place builds
    a buffer too small for the value it is then handed, and the first forward
    pass after the switch dies inside deep_ep with a bare AssertionError.
    Defaults to the scheduler's own token budget and restores the launch
    value for HT.

    ``keep_previous`` retains the outgoing manager so a switch back needs no
    fresh NVSHMEM rendezvous. Off by default: the buffer scales with
    tokens x hidden x experts, so on a large MoE it would pin more memory
    than the faster switch back is worth.
    """
    config = vllm_config or get_current_vllm_config_or_none()
    if config is None:
        raise RoleSwitchError(
            "No VllmConfig available. Pass vllm_config=: the MoE kernel "
            "rebuild reads it, and a worker RPC has no ambient config."
        )

    layers = check_switchable(model, backend, max_num_tokens, config)
    current = layers[0].moe_config.moe_parallel_config.all2all_backend
    if current == backend:
        logger.info("role switch: already on %s, nothing to do", backend)
        return 0

    if max_num_tokens is None:
        if backend == LOW_LATENCY:
            # max_num_batched_tokens is the scheduler's cap on tokens per
            # step, and speculative decoding is charged against it, so it
            # already covers MTP where max_num_seqs would not. Rounded up,
            # never down: a buffer smaller than what the scheduler may
            # dispatch is the failure this whole path kept hitting.
            budget = config.scheduler_config.max_num_batched_tokens
            max_num_tokens = ((budget + 3) // 4) * 4
        else:
            max_num_tokens = getattr(
                layers[0], "_role_switch_boot_max_num_tokens", None
            )

    with set_current_vllm_config(config, check_compile=False):
        # Ask every layer whether it could rebuild, before touching anything.
        # The weight-layout rules live in the quant methods, so only they can
        # answer -- and a refusal discovered mid-rebuild would leave the
        # manager swapped and some layers already switched.
        _precheck_rebuild(layers, backend)
        previous = _swap_all2all_manager(backend, keep_previous)
        # Flip the engine-wide config too, so anything constructed after this
        # point agrees with the layers that were just rebuilt.
        config.parallel_config.all2all_backend = backend
        switched = _rebuild_layers(layers, backend, max_num_tokens)
        if not keep_previous and previous is not None:
            # Only now is nothing holding one of its handles.
            previous.destroy()
            logger.info(
                "role switch: destroyed %s", type(previous).__name__
            )

    logger.info(
        "role switch: %s -> %s across %d MoE layers, weights untouched",
        current,
        backend,
        switched,
    )
    return switched


def _precheck_rebuild(layers: list, backend: str) -> None:
    """Dry-run every layer against ``backend``, restoring config either way.

    The backend has to be flipped for the selection to see it, so it is
    flipped and put back. Nothing else is touched: dry_run makes the quant
    method select and check without assigning.
    """
    for layer in layers:
        parallel_config = layer.moe_config.moe_parallel_config
        saved = parallel_config.all2all_backend
        try:
            parallel_config.all2all_backend = backend
            layer._quant_method.rebuild_moe_kernel(layer, dry_run=True)
        except RoleSwitchError:
            raise
        except Exception as exc:
            raise RoleSwitchError(
                f"{type(layer._quant_method).__name__} cannot rebuild for "
                f"{backend}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            parallel_config.all2all_backend = saved


def _rebuild_layers(
    layers: list, backend: str, max_num_tokens: int | None
) -> int:
    switched = 0
    for layer in layers:
        moe = layer.moe_config
        if not hasattr(layer, "_role_switch_boot_max_num_tokens"):
            layer._role_switch_boot_max_num_tokens = moe.max_num_tokens
        if max_num_tokens is not None:
            # Read by maybe_make_prepare_finalize when it sizes the new
            # handle, so it has to move before the rebuild, not after.
            moe.max_num_tokens = max_num_tokens
        # One assignment drives the whole derivation: use_deepep_ht_kernels,
        # use_deepep_ll_kernels and use_batched_activation_format are all
        # read-only properties over this field. Mutated in place because the
        # same config object is shared by the layer, its routed experts and
        # its quant method.
        layer.moe_config.moe_parallel_config.all2all_backend = backend
        layer._quant_method.rebuild_moe_kernel(layer)
        switched += 1
    return switched
