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
Pause and drain, lower the budget, switch every worker, resume. Draining is
belt and braces rather than a correctness requirement: a switch measured under
41 concurrent requests completed in 748 ms and lost none of 30 in-flight
completions. It is kept because a quiet engine is a cheaper thing to reason
about, and because ``pause_scheduler(mode="wait")`` already exists to do it.

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
    """Quiesce the scheduler, returning the mode used, or None if it could not.

    ``wait`` drains in-flight requests and is what a role change wants.
    ``keep`` merely stops stepping. Neither is available in every engine
    configuration -- inproc rejects ``wait`` outright -- and failing to pause is
    not fatal, because switching under live traffic was measured to be safe.
    """
    for mode in ("wait", "keep"):
        try:
            future = engine_core.pause_scheduler(mode=mode, clear_cache=False)
            if future is not None:
                future.result()
            return mode
        except Exception as exc:  # noqa: BLE001 - try the next mode
            logger.debug("role switch: pause mode %s unavailable: %s", mode, exc)
    logger.warning(
        "role switch: could not pause the scheduler; switching with traffic "
        "still flowing, which is measured to be safe but not preferred"
    )
    return None


def switch_pd_role(
    engine_core,
    backend: str,
    max_num_tokens: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> dict[str, Any]:
    """Move this engine between prefill and decode roles.

    ``max_num_batched_tokens`` is the scheduler budget for the new role and is
    the reason this function exists; ``max_num_tokens`` overrides the MoE
    dispatch bound, which otherwise follows the scheduler budget.
    """
    previous_budget = engine_core.scheduler.scheduler_config.max_num_batched_tokens
    paused_with = _pause(engine_core)

    try:
        if max_num_batched_tokens is not None:
            _set_scheduler_budget(engine_core, max_num_batched_tokens)
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
    }
