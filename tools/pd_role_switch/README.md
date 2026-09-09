# P/D role switching on a live engine

One engine that moves between the prefill and decode MoE configuration in
**under a second** — and in **178 ms with requests in flight** — without
reloading weights, across nodes, with CUDA graphs intact.

A replica does not have to be a prefill replica or a decode replica. It can be
whichever the current load needs, so a fleet is provisioned for total demand
rather than for peak-prefill plus peak-decode separately.

## Results

Measured on 2 x 8 H200 (139.8 GiB), GLM-5.2-FP8, EP=16, vLLM v0.28.0,
`gpu-memory-utilization` 0.90, `all2all-backend deepep_v2`.

### Switch latency

| | idle | with 24 requests in flight |
| --- | --- | --- |
| into a role held before (buffer parked) | **797–799 ms** | 178 ms (`keep`) / 7227 ms (`wait`) |
| into a role built fresh | 1483–1548 ms | — |

**24 of 24 in-flight requests completed, 0 failed**, and output was identical
before and after every round trip.

Intranode (EP=8, one node) switches in **277–629 ms**.

Repeatability: 2 of 2 extra cycles switched all 16 ranks in both directions
with matching output.

### `VLLM_PD_PAUSE_MODE` decides the cost of switching under load

The default `wait` drains in-flight requests before switching; `keep` freezes
and resumes them instead. Same node pair, EP=16:

| | fresh build | reuse parked buffer | switch under load |
| --- | --- | --- | --- |
| `wait` (default) | 1548 ms | 798 ms | **7227 ms** |
| `keep` | 1483 ms | 797 ms | **178 ms** |

One millisecond apart on an idle switch, **40x apart under load**: the drain is
the entire cost of switching with requests in flight, and none of the cost of
switching idle. A fleet re-roles while serving traffic, so this is the setting
that decides what a switch actually costs.

**`keep` is not a free win.** It completed 24 of 24 in-flight requests in this
run, the same as `wait`, but `wait` is the default because it guarantees no
request is mid-step across the switch — and one green run does not establish
that freezing is equivalent. Measure your own workload before changing it.

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
free, which is what makes a sub-second round trip possible.

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

Only needed if you are running a published image rather than an environment
built from this branch.

**Use a base image whose vLLM contains "Fast Start" (#54921, 2026-09-04).**
The injection patches nine existing files at their anchors, and one of those
anchors is `_init_moe_kernel`, which vLLM gained in that commit. On an older
base the hunk rejects.

The nightlies are tagged by the exact commit they were built from, which is what
makes one verifiable as a base:

```
vllm/vllm-openai:nightly-385dce36bcee42309924a5ece951a96db3dce7f2
```

That commit sits two behind this branch's merge base, and both are in the
mooncake connector and its tests -- none of the thirteen files here. The
injection applies to it at 13 hunks, 0 rejects, no offsets.

Release images do not work at the time of writing: `v0.29.0` was cut from
`98dff2a81` on 2026-09-08 and still predates Fast Start, as does `v0.28.0`. For
`v0.28.0` specifically there is the tag `pd-role-switch-v0.28.0` -- this work as
it stood before the branch moved onto newer upstream, and where every
measurement below was taken.

**Do not copy the changed files in.** The nine MODIFIED files carry upstream
code with them, newer than any image whose vLLM predates this branch's base, and
an engine that starts with a mismatched pair of them dies at init with an
`ImportError` rather than at the point of the switch.

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

Thirteen hunks. The `.rej` check above is the one that matters: seven of the
nine patched files change under upstream regularly, so a base image that drifts
from this branch's own base fails here rather than at runtime. Check `pd_role`
imports before launching — without it `/switch_pd_role` does not exist and the
test below will fail at the first switch rather than at startup.

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
~800 ms and 0 MiB instead of a full rebuild.

## KV connector direction

A role change is also a KV direction change: prefill produces KV, decode
consumes it. When a KV connector is configured, the switch moves
`kv_transfer_config.kv_role` with the role -- `kv_producer` when the budget
grows, `kv_consumer` when it shrinks.

**An engine deployed as `kv_both` is left alone.** `kv_both` already covers both
directions, which is exactly what a switchable engine needs, and it is what
**llm-d sets on both its prefill and its decode replicas** — so this is the
common case in a real deployment, not a corner. Overwriting it would also be
one-way: the budget mapping only ever answers `kv_producer` or `kv_consumer`, so
a round trip could never restore `kv_both` and the engine would drift
permanently away from its deployed configuration on the first switch.

`tools/pd_role_switch/test_kv_role.py` covers these rules and needs no GPU or
vLLM runtime:

```bash
python3 tools/pd_role_switch/test_kv_role.py
```

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

### KV still moves after both engines change role

The above pair had not switched. This one did — GLM-5.2-FP8, one 8-GPU node per
replica, EP=8 each, `NixlConnector`, two phases against the same two engines:

```
phase 1  prefill -> decode      decode node   1 transfer  3452160 B  0 failed
switch both engines             8/8 ranks each way
phase 2  roles reversed         prefill node  1 transfer  3452160 B  0 failed
```

In phase 2 the transfer is booked on the node that used to be the prefiller,
because it is the decoder now and the connector pulls. Answers were correct in
both phases — which is not the assertion, for the reason above.

**Read the caveat with the result.** Phase 2 worked because the proxy was
re-pointed at the new roles. Under llm-d nothing does that; see the next
section, which is the real remaining gap.

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

**This belongs to the autoscaler, not to the engine.** `/switch_pd_role` is a
mechanism: it changes one engine and knows nothing about Kubernetes. Deciding
*when* to change the ratio, choosing which replica to convert, and moving the
label are policy and orchestration, and they belong to the controller that
already watches the metrics and holds RBAC on pods. Having the engine patch its
own label would put Kubernetes write credentials inside every inference pod for
no good reason.

So the division is:

| | owns |
| --- | --- |
| engine (this branch) | `POST /switch_pd_role` — reconfigure in place, report `ranks_switched` |
| autoscaler | the ratio decision, replica selection, label, drain, ordering |

**The sequence the controller runs**, which also removes the ordering hazard —
the trick is that a converting replica should belong to *neither* role for the
duration, rather than being handed straight from one to the other:

1. take the replica out of its current role (change or remove `llm-d.ai/role`)
   so the endpoint picker stops selecting it — it stays in the pool, because the
   pool selector is `llm-d.ai/guide` and that does not change;
2. let in-flight requests drain (the engine survives a switch under load, so
   this is for tidiness rather than safety);
3. `POST /switch_pd_role` and check `ranks_switched` equals the rank count —
   a switch that did nothing also returns quickly and still answers correctly;
4. set `llm-d.ai/role` to the new role; the picker's informer observes it and
   traffic resumes in the new direction.

Between steps 1 and 4 the replica serves nothing, which is the cost of doing it
safely — bounded by the switch itself, a few hundred milliseconds, rather than
the minutes a replacement replica would take.

**And the standard llm-d guide cannot host this at all.** `llm-d.ai/role` is part
of each role Deployment's `spec.selector`, which Kubernetes will not let you
change. Relabelling a pod therefore does not move it between roles — it removes
it from its owner, which promptly replaces it with a cold one. A switchable
deployment has to keep the role out of every selector.

The deployment shape that works — one Deployment, role as a mutable pod label,
a full GLM-5.2 manifest and the four-step handover — is a guide in the
autoscaler repository: **docs/guides/pd-role-switch/**. See
[WELL-LIT-PATH-LLM-D.md](WELL-LIT-PATH-LLM-D.md) for the pointer and what it
covers.

**Status: the engine half is built and measured. The controller half is not.**
That is the gap between "one engine can change role in under a second" and "a
fleet can change its P:D ratio in under a second".

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

The switch latencies are EP=16 across two nodes; the serving-performance and
TTFT tables are EP=8 on a single node, and are separate runs.

The reuse-switch figure is stable across hardware: two independent node pairs
and both pause modes all produce ~798 ms.

### Measuring this correctly

Three properties of this workload will produce clean-looking but meaningless
numbers if the harness ignores them.

- **Throughput needs a warm-up.** Decode throughput climbs for several passes
  before flattening (444 → 607 → 641 tok/s on one arm). Comparing two arms at
  different points on that curve overstates the difference by more than 2x.
  Discard warm-up passes and check the trend has flattened.
- **Prompts must be unique per request *and* per repeat.** Re-sending one long
  prompt makes every pass after the first a prefix-cache hit — 16 × 1800 tokens
  "prefilled" in 298 ms, which is not physically possible.
- **TTFT must come from the token stream, not the HTTP response.** vLLM returns
  `200` and headers immediately on a streaming request, so
  `curl -w %{time_starttransfer}` reports ~4 ms for a 2000-token prefill. Parse
  the stream and take the first chunk that carries text. Keep a plausibility
  floor: a sub-30 ms TTFT on a prompt this size is a broken measurement, not a
  fast engine.

A related tell: if mean, p50, p90 and max come back identical, the samples have
collapsed to n=1 — check the harness before believing the number.

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
