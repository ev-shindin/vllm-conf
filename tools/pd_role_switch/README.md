# Reproducing P/D role switching on a live engine

One engine that switches between the prefill and decode MoE configuration
without reloading weights, including **across nodes with CUDA graphs intact**.

Internode is the interesting case. Under NVSHMEM the high-throughput and
low-latency buffers cannot coexist — `nvshmem::init` is called with different
team parameters for each, so the old buffer must be destroyed before the new one
is built. A destroy strands every captured CUDA graph, which leaves internode
eager-only, and eager was measured at **4.9x slower decode** (32403 ms vs
6573 ms for the same 4096 tokens). `deepep_v2` replaces NVSHMEM with NCCL
symmetric memory, two `ElasticBuffer`s *do* coexist, and the graphs survive.

## What has actually been measured

Measured on 2x8 H200 (139.8 GiB), GLM-5.2-FP8, EP=16, `gpu-memory-utilization`
0.90, vLLM v0.28.0. Numbers below are from real runs, not estimates.

| result | value |
| --- | --- |
| switch, building a new buffer | 1464–1587 ms |
| switch, reusing a parked buffer | 378–387 ms, **0 MiB allocated** |
| switch under load | 4279 ms build / 387 ms reuse, **24/24 in-flight requests completed, 0 failed** |
| repeatability | 2/2 extra cycles, 16/16 ranks both directions, matching output |
| CUDA graphs | captured (FULL decode + PIECEWISE mixed), 2.93 GiB |
| retained second buffer @128 | **+286 MiB** |
| retained second buffer @512 | **+902 MiB** |
| implied buffer size | **~1.60 MiB/token + ~81 MiB fixed** |
| NVSHMEM low-latency buffer, for contrast | 4.28 GiB (~12.2 MiB/token) |

Intranode (EP=8, single node) switches in 277–629 ms and has always kept its
graphs; it does not need any of this.

**Not yet measured:** decode tokens/s after a switch, compared against the same
engine with `--enforce-eager`. Until that exists, "the 4.9x penalty is
recovered" is an inference from the graphs being captured, not a measurement.
Buffer scaling with EP *width* is also unmeasured — every number above is EP=16.

## Prerequisites

- 2 nodes, 8 GPUs each, InfiniBand with GIN (GPU-Initiated Networking)
  capability. Check it reports non-zero: the engine logs `gin_type` during
  startup; 0 means the NICs or drivers cannot do GIN and `deepep_v2` will refuse.
- vLLM v0.28.0 with this branch applied (`feat/pd-role-switch`).
- The model weights on node-local disk if possible. Two pods pulling 753 GB
  over one NFS share took ~1355 s against ~104 s from local disk.

## Step 1 — build deep_ep against a pinned NCCL

```bash
tools/pd_role_switch/build_deep_ep.sh          # writes a wheel to ./dist
pip install --no-deps --force-reinstall dist/deep_ep-*.whl
```

**Pin NCCL to 2.30.7. Do not use a range.** `>=2.30.4` resolves to the newest
wheel, and 2.31.2 segfaults during startup: vLLM's `ncclCommProperties` in
`vllm/distributed/device_communicators/pynccl_wrapper.py` is hand-mirrored from
NCCL's *internal* headers (the struct is not in any public `nccl.h`, only the
symbol is exported), and it is written for the 2.30 layout. The failure is
nondeterministic — 3 of 8 ranks survived the query in one run — because it is
memory corruption rather than a clean API error.

deep_ep must be built against the **same** NCCL it runs on. A wheel built
against 2.31.2 and run on 2.30.7 fails with `cudaErrorIllegalAddress` inside
`csrc/elastic/buffer.hpp`.

## Step 2 — launch

```bash
tools/pd_role_switch/run_switch_test.sh --nodes 2 --model zai-org/GLM-5.2-FP8
```

The four settings that matter, and why each is needed:

| setting | value | if you get it wrong |
| --- | --- | --- |
| `nvidia-nccl-cu12` | `==2.30.7` | segfault in `ncclCommQueryProperties` |
| deep_ep build | against 2.30.7 | `cudaErrorIllegalAddress` in `buffer.hpp` |
| `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE` | **0 internode, 1 single-node** | see below |
| `VLLM_USE_DEEP_GEMM=0` **and** no `--moe-backend` | | `assert expert_start_loc.shape[0] == num_experts` |

**The hybrid setting inverts with topology.** On a single node it must be `1`:
otherwise `num_gin_ranks = num_ranks` and DeepEP drives GIN/RDMA between ranks
that are all NVLink-local, giving an illegal address. On 2+ nodes it must be
`0`: with `1` you hit `ZeroDivisionError` at `elastic.py:809`, because upstream
auto-detects `rdma_gbs` only `if num_rdma_ranks > 1` while dividing by it under
`if num_scaleout_ranks > 1`, and hybrid mode makes those two counts diverge.
That path is unreachable on one node, where the `and` short-circuits.

Note vLLM annotates `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE: bool = True` in
`envs.py` while its lambda defaults to `"0"`. The implementation wins.

deep_gemm is forced from **two** places — the `--moe-backend` flag and the
`VLLM_USE_DEEP_GEMM` env var. Clearing only one is not enough. `deepep_v2` pads
its contiguous layout, and the deep_gemm path derives `num_experts` from the
padded tensor, so it trips an assert. Left alone, vLLM's oracle picks a
padding-tolerant expert backend.

## Step 3 — switch, and check the right thing

```bash
curl -s -X POST localhost:8000/switch_pd_role \
  -H 'Content-Type: application/json' \
  -d '{"backend":"deepep_v2","max_num_tokens":128,"max_num_batched_tokens":128}'
```

**Check `ranks_switched` and `layers_switched`, not the timing or the output.**
A switch that does nothing still returns quickly and still answers correctly.
Before `_role_budget_unchanged()` existed, every `deepep_v2` switch
short-circuited on the pre-existing "already on this backend" guard: the run
reported 16/16 ranks, 373 ms, matching answers — and `ranks_switched: 0,
layers_switched: 0`. Nothing had been rebuilt.

A good switch on this configuration looks like:

```json
{"backend":"deepep_v2","ranks_switched":16,"ranks_total":16,"layers_switched":75}
```

Set `VLLM_PD_KEEP_PREVIOUS=1` to park the outgoing buffer. This is what keeps
captured CUDA graphs valid, and it is not automatic: vLLM's all2all handle cache
is a `WeakValueDictionary`, so the outgoing buffer is collected as soon as the
layers let go of it unless something holds a strong reference.

## Interpreting a failure

| symptom | cause |
| --- | --- |
| `Segfault encountered`, Python-only frames | NCCL version — pin 2.30.7 |
| `cudaErrorIllegalAddress` in `buffer.hpp` | deep_ep built against a different NCCL |
| `ZeroDivisionError` at `elastic.py:809` | hybrid mode on, internode |
| illegal address in `launch_engram_fetch` | hybrid mode off, single node |
| `assert expert_start_loc.shape[0] == num_experts` | deep_gemm still selected |
| `ranks_switched: 0` | the switch was a no-op; check the budget actually differs |

A crash can also present as a hang: the wrapper outlives the engine, so the
process stays up after the workers die. Grep for `hit an exception` and
`Engine core initialization failed`, not just for segfaults.
