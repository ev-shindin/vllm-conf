# P/D role switching on a live engine

One engine that moves between the prefill and decode MoE configuration in
**under 400 ms**, without reloading weights, across nodes, with CUDA graphs
intact.

A replica no longer has to be a prefill replica or a decode replica. It can be
whichever the current load needs, so a fleet is provisioned for total demand
rather than for peak-prefill plus peak-decode separately.

## Results

Measured on 2 x 8 H200 (139.8 GiB), GLM-5.2-FP8, EP=16, vLLM v0.28.0,
`gpu-memory-utilization` 0.90, `all2all-backend deepep_v2`.

### Switch latency

| | idle | with 24 requests in flight |
| --- | --- | --- |
| into a role held before (buffer parked) | **378–387 ms** | **387 ms** |
| into a role built fresh | 1464–1587 ms | 4279 ms |

Switching back onto a parked buffer costs the same under load as idle: the
extra time in the fresh-build case is draining in-flight work, not buffer
construction. **24 of 24 in-flight requests completed, 0 failed**, and output
was identical before and after every round trip.

Intranode (EP=8, one node) switches in **277–629 ms**.

Repeatability: 2 of 2 extra cycles switched all 16 ranks in both directions
with matching output.

### Memory, against dedicated prefill and decode replicas

Per GPU. A switching engine keeps both role buffers resident; a dedicated
replica keeps only its own.

| | per GPU | vs switching engine |
| --- | --- | --- |
| dedicated **prefill** replica | 126257 MiB | — |
| dedicated **decode** replica | 123177 MiB (projected) | — |
| **switching engine, both buffers resident** | **126543 MiB** (measured) | **+0.20%** vs prefill, **+2.35%** vs decode |

**Dual-role capability costs at most 2.35% of the card.** The retained buffer
is 286 MiB, 1.7% of the 16.5 GiB left free after boot.

That is affordable because the buffer itself is small:

| | per token | at 2048 tokens |
| --- | --- | --- |
| `deepep_v2` ElasticBuffer | **1.60 MiB** (+81 MiB fixed) | 3.29 GiB (projected) |
| NVSHMEM low-latency buffer | 12.2 MiB | 24.4 GiB |

Measured at two budgets: 128 tokens costs 286 MiB, 512 tokens costs 902 MiB.
Switching back onto a parked buffer allocates **0 MiB** — the reuse path is
free, which is what makes the 378 ms round trip possible.

`reserved_MiB` is unchanged across every sample: the buffer lives outside
PyTorch's pool, so `torch.cuda.empty_cache()` will not recover it and
torch-side memory reporting will not show it.

### Decode throughput: what the graphs are worth

Same topology, same benchmark placement (decode role after the switch), 16
concurrent requests of 256 tokens each:

| | decode of 4096 tokens |
| --- | --- |
| CUDA graphs retained across the switch | **10327 ms** |
| same engine with `--enforce-eager` | 45233 ms |
| **penalty for losing the graphs** | **4.38x** |

That is the payoff. Internode under NVSHMEM is eager-only, so it pays this 4.38x
on every decode; deepep_v2 keeps the graphs and does not.

Token totals are derived from the request shape (16 x 256 with `ignore_eos`),
not read back from the responses.

### Serving performance against dedicated replicas

The switch is cheap, but a switchable engine is not configured the way a
dedicated one is, and that costs something in steady-state serving. It is
measured here rather than assumed.

`deepep_v2` pads its contiguous layout and `deep_gemm` derives the expert count
from the padded tensor (`deep_gemm_utils.py:300`), so a switchable engine cannot
use it and the oracle picks `FLASHINFER_CUTLASS`. A dedicated replica is free to
take `DEEPGEMM` for prefill or `BATCHED_DEEPGEMM` for decode — the batched
variant being tuned for the decode activation format.

One 8 × H200 node per arm, GLM-5.2-FP8, EP=8, 16 concurrent, prompts unique per
request and per repeat, two warm-up passes discarded, four measured.

| Engine | MoE kernel | Decode throughput |
| --- | --- | --- |
| dedicated decode replica | `BATCHED_DEEPGEMM` | **471.9 tok/s** (465–481) |
| ours, decode role | `FLASHINFER_CUTLASS` | 407.1 tok/s (405–409) |
| dedicated prefill replica | `DEEPGEMM` | 99.3 tok/s (98–100) |
| ours, prefill role | `FLASHINFER_CUTLASS` | **417.3 tok/s** (407–445) |

**Decode costs 13.7%** against a dedicated decode replica. In exchange the
engine never collapses off-role: a dedicated prefill replica manages 99 tok/s of
decode work, ours 417 — **4.2×** — and ours holds ~410 tok/s in either
configuration while the specialists swing between 99 and 472.

Prefill belongs in latency, not output-token throughput. Time to the first
streamed chunk carrying text, 1500-word prompts:

| Prefill role | one request at a time | 16 concurrent | p90 at 16 |
| --- | --- | --- | --- |
| dedicated prefill replica | 533.6 ms (525–538) | **973.6 ms** | **1360 ms** |
| ours | **364.8 ms** (360–369) | 1311.2 ms | 1915 ms |

**This reverses with load.** Unloaded, ours reaches the first token 1.46×
sooner; at 16 concurrent prompts it is 1.35× slower with a worse tail. The
general-purpose kernel serves one sequence well; the batched kernel scales
better across a batch. Ours is the faster engine for an interactive,
low-concurrency prefill path and the slower one for a saturated prefill fleet.

### Why internode is the interesting case

Under NVSHMEM the high-throughput and low-latency buffers cannot coexist —
`nvshmem::init` takes different team parameters for each — so the outgoing
buffer must be destroyed before the incoming one is built. A destroy strands
every captured CUDA graph, leaving internode eager-only, and eager measured
**4.9x slower decode** (32403 ms against 6573 ms for the same 4096 tokens).

`deepep_v2` uses NCCL symmetric memory instead. Two `ElasticBuffer`s coexist,
the graphs survive, and a switch becomes a buffer swap rather than a teardown.

## Running it

### 1. Build deep_ep against a pinned NCCL

```bash
tools/pd_role_switch/build_deep_ep.sh
pip install --no-deps --force-reinstall dist/deep_ep-*.whl
cd /tmp && python3 -c "import deep_ep; print(hasattr(deep_ep, 'ElasticBuffer'))"
```

The version pin is exact and deep_ep must be built against the same NCCL it
runs on. Both matter; see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### 1b. Put this branch into a stock image

Only needed if you are running the published `vllm/vllm-openai:v0.28.0` image
rather than an environment built from this branch. It was previously left
unstated, and there is no way to guess it correctly.

**Do not copy the changed files in.** This branch is based on vLLM `main`, which
has drifted past v0.28.0, so the nine MODIFIED files carry newer upstream code
with them — `vllm/v1/engine/core.py` imports `resolve_kv_cache_layout`, which
does not exist in v0.28.0, and every engine dies at init with an `ImportError`.

Copy the four ADDED files whole, and PATCH the nine modified ones at their
anchors:

```bash
BASE=$(git merge-base main HEAD)
SP=$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')

# 4 added files: self-contained, safe to copy
git archive HEAD $(git diff --diff-filter=A --name-only $BASE..HEAD -- vllm/) | tar -x -C "$SP"

# 9 modified files: apply as a patch, and refuse anything that does not apply
git diff $BASE..HEAD -- $(git diff --diff-filter=M --name-only $BASE..HEAD -- vllm/) > /tmp/edits.patch
( cd "$SP" && patch -p1 --forward --batch < /tmp/edits.patch )
[ "$(find "$SP/vllm" -name '*.rej' | wc -l)" = "0" ] || { echo "patch did not apply cleanly"; exit 1; }

python3 -c "import vllm.v1.engine.pd_role; print('pd_role OK')"
```

Verified against v0.28.0: 13 hunks, 0 rejects. Check `pd_role` imports before
launching — without it `/switch_pd_role` does not exist and the test below will
fail at the first switch rather than at startup.

### 2. Launch

Run on every node. Node 0 serves the API, the rest are headless followers.

```bash
# node 0
NODE_RANK=0 MASTER_ADDR=10.0.0.1 tools/pd_role_switch/run_switch_test.sh
# node 1
NODE_RANK=1 MASTER_ADDR=10.0.0.1 tools/pd_role_switch/run_switch_test.sh
```

Single node works too: `NODES=1 tools/pd_role_switch/run_switch_test.sh`.

The script sets the required environment itself, including the one setting that
depends on topology, and prints a PASS or FAIL per switch.

### 3. Switch a running engine

```bash
curl -s -X POST localhost:8000/switch_pd_role \
  -H 'Content-Type: application/json' \
  -d '{"backend":"deepep_v2","max_num_tokens":128,"max_num_batched_tokens":128}'
```

```json
{"backend":"deepep_v2","ranks_switched":16,"ranks_total":16,"layers_switched":75}
```

`ranks_switched` and `layers_switched` are the result to check. A switch that
did nothing also returns quickly and still answers correctly, so timing and
output alone will not tell you it worked.

`/switch_pd_role` is a development route: it is attached by
`register_vllm_dev_api_routers`, which the server calls only when
**`VLLM_SERVER_DEV_MODE=1`** is set (`launchers/api_server/routers.py:34`). The
script exports it. Launching without it gives an engine that serves normally
while every switch returns `404` — which shows up as a 6 ms switch with no ranks
rather than as a missing route.

Set `VLLM_PD_KEEP_PREVIOUS=1` (the script does) to park the outgoing buffer.
That is what keeps captured CUDA graphs valid and makes the return switch cost
378 ms and 0 MiB instead of a full rebuild.

## KV connector direction

A role change is also a KV direction change: prefill produces KV, decode
consumes it. When a KV connector is configured, the switch moves
`kv_transfer_config.kv_role` with the role -- `kv_producer` when the budget
grows, `kv_consumer` when it shrinks -- so `kv_both`, which is deprecated for
NixlConnector, is not required.

The response reports it:

```json
{"backend":"deepep_v2","ranks_switched":16,"kv_role":"kv_consumer",
 "previous_kv_role":"kv_producer"}
```

Connectors that read `kv_role` once at construction (`MooncakeConnector`,
`MooncakeStoreConnector`) are refused rather than left disagreeing with the
config. NIXL reads it per call, so the flip is a declaration of intent and
nothing is torn down.

Verified on hardware: EP=8 with a NixlConnector attached, all 8 ranks moved
`kv_producer -> kv_consumer` on the switch to the smaller budget and back again
on the return, with identical output across the round trip.

```
role switch: kv_role kv_producer -> kv_consumer for NixlConnector   (x8 ranks)
role switch: kv_role kv_consumer -> kv_producer for NixlConnector   (x8 ranks)
```

That exercises the direction flip and shows NixlConnector tolerates a live
`kv_role` change. It does not exercise an actual KV transfer between a prefill
and a decode replica, which needs two engines and a disaggregated setup.

### KV does move between two engines (measured separately)

A 1P1D pair on stock v0.28.0, `NixlConnector`, upstream's
`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`:

```
vllm:nixl_bytes_transferred_count        1
vllm:nixl_bytes_transferred_sum          3670016      (3.5 MiB)
vllm:nixl_xfer_time_seconds_sum          0.008462
vllm:nixl_num_failed_transfers_total     0
```

Booked on the **decode** engine — the connector pulls, so the puller records the
transfer and the prefiller reading zero is correct, not a miss.

**Assert on bytes, never on output.** A decode engine that receives no KV
silently recomputes the prefix and answers correctly, so an output-equality
check passes a completely broken transfer. Match the metric name up to `{`: a
bare prefix match also catches Prometheus's `_created` series, whose value is an
epoch timestamp, which once reported `failed=1788776689` for zero real failures.

Still open: this pair had not switched roles. A transfer **across** a role
change, with the direction reversed, is the one claim on this branch that
hardware has not yet answered.

## Integrating with llm-d: the switch must also move the pod label

**In an llm-d deployment the engine switch alone does nothing useful, and is
actively harmful.** llm-d decides which endpoints are prefill and which are
decode from a **pod label**, not from anything the engine reports:

```
llm-d.ai/guide=<pool>      # InferencePool selector -- unchanged by a switch
llm-d.ai/role=prefill      # what the endpoint picker filters on
llm-d.ai/model=<model>
```

The InferencePool selects on `llm-d.ai/guide`, and the endpoint picker's
`prefill-filter` / `decode-filter` split that set by `llm-d.ai/role`. That label
is written by the Deployment template and is static for the pod's lifetime.
Nothing in this branch touches it (`grep -r 'llm-d.ai/role' vllm/` returns
nothing).

So after `POST /switch_pd_role` on an llm-d-managed pod:

- the engine is reconfigured (budget, buffer, KV direction), but
- the router still believes it holds the old role, so
- an engine now running a 128-token decode budget keeps receiving prefill
  traffic with multi-thousand-token prompts.

The fleet's effective P:D ratio — the thing the feature exists to change — does
not move at all.

**What a working integration needs.** The switch has to patch its own pod's
`llm-d.ai/role`, which means a ServiceAccount with `patch` on pods, the pod's own
name and namespace via the downward API, and a flag so non-llm-d deployments are
unaffected. Membership itself is safe: the pool selector is `llm-d.ai/guide`,
which does not change, so the pod is never ejected from the pool.

**Ordering is the subtle part, and neither obvious order is correct:**

- relabel first, and the router stops sending prefill work while the engine is
  still in the prefill configuration;
- switch the engine first, and the router keeps sending prefill work to an
  engine that has already dropped to a decode budget.

Either way there is a window in which routing and configuration disagree. The
sequence has to be drain, relabel, wait for the picker to observe the change,
then switch — and the picker's observation is an informer update, so the wait is
small but not zero.

**Status: identified, not built.** This is the gap between "one engine can change
role in 378 ms" and "a fleet can change its P:D ratio in 378 ms".

## Requirements

- 8 GPUs per node; InfiniBand with GIN (GPU-Initiated Networking) for multi-node
- vLLM v0.28.0 with this branch
- NCCL 2.30.7, and deep_ep built against it
- Model weights on node-local disk if available: 753 GB over one NFS share took
  ~1355 s for two nodes against ~104 s from local disk

## Scope of these numbers

Measured at EP=16 on H200. Not yet measured: how the buffer scales with EP
*width*. Every number here is EP=16, and a DeepEP buffer holds receive space
related to rank count, so EP=32 is a projection rather than a measurement.

The switch latencies were taken on kermit with node-local weights; the
throughput pair was taken on fozzie, where weights come off a shared PVC. Both
halves of that pair ran in identical conditions, so the 4.38x ratio stands, but
absolute switch timings differ between the two clusters.

Rows marked "projected" are derived from the two measured budget points, not
observed directly.

The serving-performance and TTFT tables are EP=8 on kermit, single node, and are
separate runs from the switch latencies above.

### Three ways these benchmarks lied, all of which looked like clean results

Anyone re-running this will meet them, so they are named rather than fixed
silently:

- **An unconverged warm-up curve.** Decode throughput climbs for several passes
  before flattening (444 → 607 → 641 tok/s on one arm). Comparing two arms at
  different points on that curve produced a 34% gap where the settled figure is
  13.7%. Discard warm-up passes and check the trend has flattened.
- **The prefix cache.** Re-sending one identical long prompt makes every pass
  after the first a cache hit — 16 × 1800 tokens "prefilled" in 298 ms, which is
  not physically possible. Prompts must be unique per request *and* per repeat.
- **TTFT taken from the HTTP response headers.** vLLM returns `200` and headers
  immediately on a streaming request, so `curl -w %{time_starttransfer}` reports
  ~4 ms for a 2000-token prefill. Parse the stream and take the first chunk that
  actually carries text. A plausibility floor is worth keeping: a sub-30 ms TTFT
  on a prompt this size is a broken measurement, not a fast engine.

A related tell: if mean, p50, p90 and max come back identical, the samples have
collapsed to n=1 — check the harness before believing the number.
