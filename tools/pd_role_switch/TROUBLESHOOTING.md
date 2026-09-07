# Troubleshooting P/D role switching

Failure modes seen while bringing `deepep_v2` up, with the cause for each. If a
run does not reproduce the numbers in [README.md](README.md), start here.

## Symptom to cause

| symptom | cause | fix |
| --- | --- | --- |
| `Segfault encountered`, Python-only frames, during startup | NCCL newer than 2.30 | pin `nvidia-nccl-cu12==2.30.7` |
| `cudaErrorIllegalAddress` in `csrc/elastic/buffer.hpp` | deep_ep built against a different NCCL than it runs on | rebuild deep_ep against the pinned version |
| `ZeroDivisionError` at `deep_ep/buffers/elastic.py:809` | hybrid mode enabled on 2+ nodes | `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=0` |
| illegal address in `launch_engram_fetch` | hybrid mode disabled on a single node | `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1` |
| `assert expert_start_loc.shape[0] == num_experts` | deep_gemm still selected | `VLLM_USE_DEEP_GEMM=0` **and** drop `--moe-backend` |
| `ranks_switched: 0` with a fast, correct-looking response | the switch was a no-op | check the requested budget differs from the current one |
| engine appears hung, pod still `Running` | the workers died; the wrapper outlives them | grep `hit an exception`, `Engine core initialization failed` |

## The NCCL pin has to be exact

`>=2.30.4` resolves to the newest wheel. 2.31.2 segfaults inside
`ncclCommQueryProperties`.

`vllm/distributed/device_communicators/pynccl_wrapper.py` hand-mirrors NCCL's
`ncclCommProperties` as a ctypes `Structure`, and its own comment says
"NCCL 2.30+". The struct appears in **no public `nccl.h`** — only the symbol is
exported from `libnccl.so` — so it is mirrored from NCCL's internal headers and
is free to change between minor versions with no public API change.
(`nccl.h` grew from 844 to 972 lines between 2.30.7 and 2.31.2.)

The failure is memory corruption, not a clean API error, so it is
nondeterministic: in one run 3 of 8 ranks returned from the call normally and
the other 5 died. A single clean boot does not clear this.

`has_deep_ep_v2()` accepts any NCCL `>= 2.30.4` against that fixed mirror, so
newer releases will keep being accepted. A defensive upper bound would turn the
segfault into a clean "deepep_v2 unavailable".

## deep_ep must match its NCCL

deep_ep's compiled `_C` tracks the NCCL it was built against. The 2.0.0+local
build shipped in `vllm/vllm-openai:v0.28.0` was built against 2.29.7. Observed:

- shipped build on NCCL 2.30.7: created 600 buffers, then `cudaErrorIllegalAddress`
- rebuilt against 2.31.2, run on 2.30.7: could not complete a single buffer
- rebuilt against 2.30.7, run on 2.30.7: works

## Hybrid mode inverts with topology

`VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE` selects which rank set GIN spans:

```c
num_gin_ranks = allow_hybrid_mode ? num_scaleout_ranks : num_ranks;
```

**Single node needs 1.** With 0, `num_gin_ranks = num_ranks` and DeepEP drives
GIN/RDMA between ranks that are all NVLink-local, giving an illegal address in
`launch_engram_fetch`.

**Two or more nodes need 0.** With 1, `get_theoretical_num_sms` raises
`ZeroDivisionError`: upstream auto-detects `rdma_gbs` only `if num_rdma_ranks > 1`
but divides by it under `if num_scaleout_ranks > 1`, and hybrid mode makes those
two counts diverge. On one node the `and` short-circuits, so the bug is
unreachable there — it only appears internode, which is where hybrid mode was
supposed to help.

Also note vLLM annotates `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE: bool = True` in
`envs.py` while its lambda defaults to `"0"`. The implementation wins.

## deep_gemm is forced from two places

`--moe-backend deep_gemm` and `VLLM_USE_DEEP_GEMM=1`. Clearing one leaves the
other in effect.

`deepep_v2` allocates its contiguous layout with worst-case padding —
`oracle/fp8.py` says so explicitly — and the deep_gemm path derives
`num_experts` from the padded tensor, so it trips
`assert expert_start_loc.shape[0] == num_experts`. Left alone, vLLM's oracle
picks a padding-tolerant expert backend (FlashInfer TRT-LLM on capability 100,
FlashInfer CUTLASS for EP on capability 90).

## A no-op switch looks like a successful one

Before `_role_budget_unchanged()` existed, every `deepep_v2` switch hit the
pre-existing guard:

```python
if current == backend:
    logger.info("role switch: already on %s, nothing to do", backend)
    return 0
```

That guard predates `deepep_v2`, where the backend name was the whole identity
of a role. Under v2 one backend serves both roles and the role is the token
budget, so it matched on every call. The run reported 16/16 ranks, 373 ms
switches and matching answers, with `ranks_switched: 0, layers_switched: 0` —
the scheduler budget moved but no layer was rebuilt and no buffer was built.

Always check `ranks_switched` and `layers_switched`. Timing and output do not
distinguish a working switch from a skipped one.

## Retention is not automatic

vLLM's all2all handle cache is a `WeakValueDictionary`. Its only strong
referents are the layers' prepare/finalize objects, so once the layers are
rebuilt the outgoing buffer is collected and its memory released — which is
exactly what invalidates captured CUDA graphs.

`VLLM_PD_KEEP_PREVIOUS=1` takes a strong reference before the rebuild. Without
it, "the manager is cached" reuses nothing.

## "Unknown vLLM environment variable detected: VLLM_PD_KEEP_PREVIOUS"

Harmless, and it appears once per API server process, so a DP=16 run prints it
sixteen times. vLLM warns about any `VLLM_*` variable missing from `vllm/envs.py`,
and this one is deliberately not declared there — `pd_role_switch.py` reads it
straight from `os.environ`, so the injection does not have to patch `envs.py`.

Do not read the warning as "retention is off". Check the memory instead: with
retention working, both buffers are resident (a few hundred MiB above the
one-buffer figure) and the switch back onto the parked buffer allocates **0 MiB**.
A rebuild would show neither.

## Every switch returns in a few ms with no ranks

```
### -> decode(128): / in 6 ms
FAIL  ranks=0/8 layers=0 -- the switch did nothing
```

The engine boots, serves correct answers, and every switch is instant. In the
engine log:

```
INFO: "POST /switch_pd_role HTTP/1.1" 404 Not Found
```

`/switch_pd_role` is attached by `register_vllm_dev_api_routers`, which the
server calls **only when `VLLM_SERVER_DEV_MODE=1`**
(`vllm/entrypoints/launchers/api_server/routers.py:34`). Without it the route
does not exist. `run_switch_test.sh` exports it; a hand-rolled launch must too.

Note what this looks like if you only watch timing and output: a very fast
switch on an engine that still answers perfectly. It is the reason the harness
checks `ranks_switched` and `layers_switched` and fails on zero.

Importing `vllm.v1.engine.pd_role` successfully does **not** mean the route is
served — the module can be present while the router is never attached.

## Diagnosing on Kubernetes

- A crash presents as a hang. The wrapper shell outlives the engine, so the pod
  stays `Running` and holds its GPUs long after the workers died.
- The kubelet rotates the container log. A crash that spams stack traces can
  push the root cause out of `kubectl logs` entirely; read the engine's own
  `tee`d copy inside the pod.
- `kubectl exec` needs a running pod, `kubectl logs` works on a terminated one.
  Snapshot logs while the pod lives if you intend to read them after it dies.
