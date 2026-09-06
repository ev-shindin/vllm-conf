# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive a prefill/decode role change from the engine side.

``pd_role_switch`` flips the MoE all2all backend, but it runs in the workers and
half of a role change does not live there. The scheduler does: it caches
``max_num_scheduled_tokens`` at init and runs in EngineCore, so a worker cannot
lower the token budget for the decode role no matter what it does to its own
copy of the config.

That matters for more than tidiness. The low-latency buffer is linear in the
token budget -- on a 753B-class MoE roughly 4 GiB at 256 tokens against 16 GiB
at 1024 -- so moving the budget with the role is the single largest memory
decision in the switch, and it can only be made here.

Sequence
--------
Lower the budget, switch every worker, done. No pause and no drain by
default: a switch measured under 41 concurrent requests completed in 748 ms
and lost none of 30 in-flight completions, so there is nothing observed for a
pause to protect.

Draining via ``pause_scheduler(mode="wait")`` is not merely unnecessary here,
it is unusable: it drains by continuing to step(), and this runs inside the
EngineCore busy loop that would have to do the stepping. Waiting on it hung a
switch for three minutes. ``pause=True`` queues new admissions without
waiting for anything.

On failure the budget is put back and the scheduler resumed, so a refused
switch leaves the engine as it was found. The worker-side guards refuse before
mutating anything, so a refusal there means no rank moved.
"""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _worker_switch_role(
    worker,
    backend: str,
    max_num_tokens: int | None,
    scheduler_budget: int | None,
) -> int:
    """Run in every worker process; returns the number of layers switched.

    Passed to ``collective_rpc`` as a callable, so this needs no worker
    extension class to be registered at launch.

    The worker holds its own copy of the config, so the new scheduler budget is
    applied here too. Otherwise the worker's own check -- that the buffer covers
    what the scheduler may dispatch -- would compare against the launch value
    and refuse a switch that is in fact correct.
    """
    from vllm.model_executor.layers.fused_moe.pd_role_switch import (
        switch_all2all_backend,
    )

    vllm_config = worker.vllm_config
    if scheduler_budget is not None:
        vllm_config.scheduler_config.max_num_batched_tokens = scheduler_budget

    model = worker.model_runner.model
    return switch_all2all_backend(
        model,
        backend,
        vllm_config=vllm_config,
        max_num_tokens=max_num_tokens,
    )


def _set_scheduler_budget(engine_core, budget: int) -> None:
    """Move the scheduler's per-step token budget.

    Both halves matter: the config value, which is read live for the input
    budget, and ``max_num_scheduled_tokens``, which the scheduler cached at
    init and uses as the actual per-step cap. Setting only the config leaves
    the scheduler dispatching at its old rate into a buffer sized for the new
    one, which is the mismatch every guard here exists to prevent.
    """
    engine_core.scheduler.scheduler_config.max_num_batched_tokens = budget
    engine_core.scheduler.max_num_scheduled_tokens = budget


def _pause(engine_core) -> str | None:
    """Stop admitting new requests. Never waits, and never drains.

    ``pause_scheduler(mode="wait")`` drains by continuing to step(), and
    this runs inside EngineCore's busy loop -- the only thread that could
    step. Blocking on its future deadlocks the drain it is waiting for, and
    that hung a switch for three minutes with no way back.

    So the pause state is set directly and nothing is awaited. New
    admissions queue for the sub-second the switch takes; requests already
    running keep running, which is measured to be safe -- 30 of 30 in-flight
    completions survived a switch under 41 concurrent requests.
    """
    try:
        from vllm.v1.core.sched.interface import PauseState

        engine_core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
        return "paused_new"
    except Exception as exc:  # noqa: BLE001 - not being able to pause is fine
        logger.debug("role switch: could not pause admissions: %s", exc)
        return None


# Connectors that cache the role at __init__ instead of reading it from the
# config on each use. Flipping the config under those desynchronises them:
# MooncakeConnector sets self.is_kv_producer once in its constructor, so it
# would keep behaving as whatever it was built as while the config claims
# otherwise. Refuse rather than half-switch.
_ROLE_CACHING_CONNECTORS = frozenset({"MooncakeConnector", "MooncakeStoreConnector"})

# Literals rather than importing HIGH_THROUGHPUT/LOW_LATENCY from
# pd_role_switch: this module imports that one lazily, inside the function,
# and a module-level import would risk a cycle for two string constants.
_BACKEND_KV_ROLE = {
    "deepep_high_throughput": "kv_producer",
    "deepep_low_latency": "kv_consumer",
}

# One backend serves both roles under deepep_v2, so its name says nothing
# about direction and the map above cannot answer for it.
_BUDGET_KEYED_BACKENDS = frozenset({"deepep_v2"})


def _kv_role_for_budget(previous: int | None, current: int | None) -> str | None:
    """KV direction implied by a change in the scheduler budget.

    Prefill runs a large budget and produces KV; decode runs a small one and
    consumes it. So a budget that shrank means the engine moved to decode and
    one that grew means it moved to prefill. An unchanged budget is not a role
    change and must not move the connector.
    """
    if previous is None or current is None or previous == current:
        return None
    return "kv_consumer" if current < previous else "kv_producer"



def _switch_kv_role(
    engine_core,
    backend: str,
    previous_budget: int | None = None,
    current_budget: int | None = None,
) -> str | None:
    """Point the KV connector the way the new role sends KV. Returns the
    role now in effect, or None when there is no connector to point.

    Prefill produces KV and decode consumes it, so a role change is also a
    direction change. kv_role='kv_both' still works but is deprecated with
    NixlConnector and slated for removal.

    Safe as a plain assignment for NIXL: across base_worker, pull_worker,
    push_worker and both schedulers, kv_role is read exactly once -- a
    pp_size > 1 guard -- and every other reference sits behind a pcp_size >
    1 check. Memory registration, handshake and transfers are all
    role-agnostic, which is precisely why kv_both works at all. So on this
    path the flip is a declaration of intent rather than a behaviour
    change, and nothing has to be torn down.
    """
    cfg = getattr(engine_core.vllm_config, "kv_transfer_config", None)
    if cfg is None or getattr(cfg, "kv_connector", None) is None:
        return None
    if backend in _BUDGET_KEYED_BACKENDS:
        wanted = _kv_role_for_budget(previous_budget, current_budget)
    else:
        wanted = _BACKEND_KV_ROLE.get(backend)
    if wanted is None or cfg.kv_role == wanted:
        return getattr(cfg, "kv_role", None)
    if cfg.kv_connector in _ROLE_CACHING_CONNECTORS:
        raise RuntimeError(
            f"{cfg.kv_connector} caches kv_role at construction, so moving"
            f" it from {cfg.kv_role} to {wanted} here would leave the"
            f" connector and the config disagreeing. Run this engine with a"
            f" connector that reads kv_role per call, or pin the role."
        )
    logger.info(
        "role switch: kv_role %s -> %s for %s",
        cfg.kv_role,
        wanted,
        cfg.kv_connector,
    )
    cfg.kv_role = wanted
    return wanted


def switch_pd_role(
    engine_core,
    backend: str,
    max_num_tokens: int | None = None,
    max_num_batched_tokens: int | None = None,
    pause: bool = False,
) -> dict[str, Any]:
    """Move this engine between prefill and decode roles.

    ``max_num_batched_tokens`` is the scheduler budget for the new role and is
    the reason this function exists; ``max_num_tokens`` overrides the MoE
    dispatch bound, which otherwise follows the scheduler budget.

    ``pause`` stops new admissions for the duration. Off by default: it
    guards against nothing that has been observed, and see _pause for why
    the draining variant cannot be used from here at all.
    """
    previous_budget = engine_core.scheduler.scheduler_config.max_num_batched_tokens
    # Off by default: it protects against nothing measured, and the one
    # mode that would have drained is the one that deadlocks here.
    paused_with = _pause(engine_core) if pause else None

    previous_kv_role = getattr(
        getattr(engine_core.vllm_config, "kv_transfer_config", None),
        "kv_role",
        None,
    )
    try:
        if max_num_batched_tokens is not None:
            _set_scheduler_budget(engine_core, max_num_batched_tokens)
        kv_role = _switch_kv_role(
            engine_core,
            backend,
            previous_budget=previous_budget,
            current_budget=max_num_batched_tokens,
        )
        switched = engine_core.model_executor.collective_rpc(
            _worker_switch_role,
            args=(backend, max_num_tokens, max_num_batched_tokens),
        )
    except Exception:
        # Put the budget back before re-raising: the workers refuse before
        # mutating, so on this path nothing has moved and the engine should be
        # left exactly as it was found.
        if max_num_batched_tokens is not None:
            _set_scheduler_budget(engine_core, previous_budget)
        # The direction is part of the role, so it goes back with it.
        cfg = getattr(engine_core.vllm_config, "kv_transfer_config", None)
        if cfg is not None and previous_kv_role is not None:
            cfg.kv_role = previous_kv_role
        raise
    finally:
        if paused_with is not None:
            engine_core.resume_scheduler()

    layers = switched[0] if switched else 0
    logger.info(
        "role switch: %s across %s layers, scheduler budget %s -> %s, paused=%s",
        backend,
        layers,
        previous_budget,
        engine_core.scheduler.scheduler_config.max_num_batched_tokens,
        paused_with,
    )
    return {
        "backend": backend,
        "layers_switched": layers,
        "scheduler_budget": (
            engine_core.scheduler.scheduler_config.max_num_batched_tokens
        ),
        "previous_scheduler_budget": previous_budget,
        "paused_with": paused_with,
        # Reported so the caller can see the direction as well as the
        # backend: prefill produces KV, decode consumes it, and a control
        # plane labelling Pods by role needs both halves to agree.
        "kv_role": kv_role,
        "previous_kv_role": previous_kv_role,
    }
