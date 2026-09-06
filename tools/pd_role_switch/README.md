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

**Not yet verified on hardware.** Every measurement in this document ran with no
KV connector attached, so the direction flip is implemented and reviewed but
unexercised. Treat it as untested until a run with a connector confirms it.

## Requirements

- 8 GPUs per node; InfiniBand with GIN (GPU-Initiated Networking) for multi-node
- vLLM v0.28.0 with this branch
- NCCL 2.30.7, and deep_ep built against it
- Model weights on node-local disk if available: 753 GB over one NFS share took
  ~1355 s for two nodes against ~104 s from local disk

## Scope of these numbers

Measured at EP=16 on H200. Not yet measured
against an eager baseline (CUDA graphs are captured and in-flight requests
survive a switch, but tokens/s has not been compared), and how the buffer scales
with EP *width* — every number here is EP=16, and a DeepEP buffer holds receive
space related to rank count, so EP=32 is a projection rather than a measurement.

Rows marked "projected" are derived from the two measured budget points, not
observed directly.
