# Troubleshooting P/D role switching

Failure modes seen while bringing `deepep_v2` up, with the cause for each. If a
run does not reproduce the numbers in [README.md](README.md), start here.

## Symptom to cause

| symptom | cause | fix |
| --- | --- | --- |
| `Segfault encountered`, Python-only frames, during startup | vLLM's `ncclCommProperties` mirror older than the NCCL runtime | fixed upstream by the version clamp; on an older vLLM, pin NCCL to the mirror's layout |
| `cudaErrorIllegalAddress` in `csrc/elastic/buffer.hpp` | deep_ep built against a different NCCL than it runs on | rebuild deep_ep against the installed NCCL |
| `missing link library: ... nccl=none`, on an image that has one | `nvidia` is a namespace package, so `nvidia.__file__` is `None` | use `nvidia.__path__`; fixed in `build_deep_ep.sh` |
| ptxas `Feature 'elect' requires .target sm_90 or higher` | built for the image's whole `TORCH_CUDA_ARCH_LIST` | DeepEP is sm_90-only; set `DEEP_EP_ARCH` |
| `ZeroDivisionError` at `deep_ep/buffers/elastic.py:809` | hybrid mode enabled on 2+ nodes | `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=0` |
| illegal address in `launch_engram_fetch` | hybrid mode disabled on a single node | `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1` |
| `assert expert_start_loc.shape[0] == num_experts` | deep_gemm still selected | `VLLM_USE_DEEP_GEMM=0` **and** drop `--moe-backend` |
| `ranks_switched: 0` with a fast, correct-looking response | the switch was a no-op | check the requested budget differs from the current one |
| engine appears hung, pod still `Running` | the workers died; the wrapper outlives them | grep `hit an exception`, `Engine core initialization failed` |

## NCCL needs no pin; the CUDA major does need checking

`vllm/distributed/device_communicators/pynccl_wrapper.py` mirrors NCCL's
`ncclCommProperties` as a ctypes `Structure`. The struct appears in **no public
`nccl.h`** — only the symbol is exported from `libnccl.so` — so it is mirrored
from NCCL's internal headers and changes between minor versions with no public
API change. (`nccl.h` grew from 844 to 972 lines between 2.30.7 and 2.31.2.)

`ncclCommQueryProperties` fills the fields gated by the version the **caller**
declares in `props.version`, not by `props.size`. A mirror written for one
layout, told the runtime is newer, is therefore written past its end — memory
corruption rather than a clean API error, and nondeterministic with it: in one
run 3 of 8 ranks returned from the call normally and the other 5 died.

The mirror now carries the v2.31.2 layout and clamps what it declares:

```python
props.version = min(nccl.ncclGetRawVersion(), NCCL_COMM_PROPERTIES_LAYOUT_VERSION)
```

so the runtime fills only fields the mirror has, in either direction, and no
particular NCCL version is required. To use fields from a newer NCCL, extend the
layout and bump the constant.

The CUDA major still has to be right. Both `vllm/vllm-openai:v0.28.0` and the
nightly named in [README.md](README.md) are CUDA 13 and carry
`nvidia-nccl-cu13` 2.30.7, so installing an `nvidia-nccl-cu12` wheel puts a
second NCCL beside the first instead of replacing it, and leaves the
`libnccl.so.2` search to pick between them. `build_deep_ep.sh` reads which wheel
is installed rather than naming one.

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
