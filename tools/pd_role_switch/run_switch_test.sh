#!/bin/bash
# Boot an engine on deepep_v2, switch its role, and verify the switch was real.
#
# Run this on EVERY node. Node 0 serves the API and drives the test; the others
# are headless followers. Example, two nodes:
#
#   # on node 0
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 ./run_switch_test.sh
#   # on node 1
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 ./run_switch_test.sh
#
# Single node (EP=8) works too: NODES=1 ./run_switch_test.sh
#
# See README.md for why each setting is what it is. The short version: pin NCCL
# to 2.30.7, build deep_ep against that same version, set hybrid mode to 0
# internode and 1 single-node, and keep deep_gemm out of the way.
set -uo pipefail

MODEL=${MODEL:-zai-org/GLM-5.2-FP8}
NODES=${NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PORT=${PORT:-8000}
PREFILL_TOKENS=${PREFILL_TOKENS:-2048}
DECODE_TOKENS=${DECODE_TOKENS:-128}
GPU_UTIL=${GPU_UTIL:-0.90}
EP=$(( NODES * GPUS_PER_NODE ))

# Hybrid mode INVERTS with topology; see README. Single node needs 1 (keep
# GIN/RDMA away from NVLink-local ranks), 2+ nodes need 0 (upstream DeepEP
# divides by an rdma_gbs it only auto-detects when num_rdma_ranks > 1, and
# hybrid mode makes that count diverge from num_scaleout_ranks).
if [ "$NODES" -gt 1 ]; then HYBRID=${HYBRID:-0}; else HYBRID=${HYBRID:-1}; fi

export VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=$HYBRID
# deep_gemm is forced from two places and BOTH must be cleared: this env var and
# the --moe-backend flag (simply omitted below). deepep_v2 pads its contiguous
# layout and the deep_gemm path derives num_experts from the padded tensor.
export VLLM_USE_DEEP_GEMM=0
# Park the outgoing buffer so captured CUDA graphs stay valid across a switch.
# Not automatic: the all2all handle cache is a WeakValueDictionary, so without a
# strong reference the buffer is collected as soon as the layers let go.
export VLLM_PD_KEEP_PREVIOUS=${VLLM_PD_KEEP_PREVIOUS:-1}
export VLLM_PD_ALLOW_CUDAGRAPHS=${VLLM_PD_ALLOW_CUDAGRAPHS:-1}

echo "### EP=${EP} (${NODES} x ${GPUS_PER_NODE}), node_rank=${NODE_RANK}, hybrid=${HYBRID}"
python3 -c "import importlib.metadata as m; print('### nccl:', m.version('nvidia-nccl-cu12'))" 2>/dev/null
python3 -c "import deep_ep, importlib.metadata as m; print('### deep_ep:', m.version('deep_ep'), 'ElasticBuffer:', hasattr(deep_ep,'ElasticBuffer'))" 2>/dev/null

if [ "$NODE_RANK" = "0" ]; then
  ROLEFLAGS="--port ${PORT}"
else
  # --data-parallel-start-rank goes ONLY on the followers: arg_utils infers
  # hybrid load balancing from "start-rank set AND not headless", and the leader
  # defaults to rank 0 already.
  ROLEFLAGS="--headless --data-parallel-start-rank $(( NODE_RANK * GPUS_PER_NODE ))"
fi

DPFLAGS=""
if [ "$NODES" -gt 1 ]; then
  DPFLAGS="--data-parallel-address ${MASTER_ADDR} --data-parallel-rpc-port ${DP_RPC_PORT:-5555}"
fi

# NOTE: no --moe-backend. Left alone, vLLM's oracle picks an expert backend that
# tolerates deepep_v2's padding.
vllm serve "$MODEL" $ROLEFLAGS \
  --trust-remote-code \
  --block-size 64 \
  --kv-cache-dtype fp8 \
  --data-parallel-size "$EP" --data-parallel-size-local "$GPUS_PER_NODE" \
  $DPFLAGS \
  --enable-expert-parallel \
  --tensor-parallel-size 1 \
  --all2all-backend deepep_v2 \
  --max-num-seqs 128 --max-num-batched-tokens "$PREFILL_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 \
  --worker-extension-cls roleext2.RoleSwitcher 2>&1 | tee /tmp/engine.log &
ENGINE_PID=$!

# Followers serve no API; hold them open while rank 0 drives the test.
if [ "$NODE_RANK" != "0" ]; then
  wait $ENGINE_PID
  exit 0
fi

echo "### waiting for the API (up to 30 min; a cold multi-node load is slow)"
for i in $(seq 1 360); do
  curl -sf "localhost:${PORT}/health" >/dev/null 2>&1 && break
  # A crash presents as a hang: the wrapper outlives the engine. Fail fast on
  # the errors that actually occur rather than waiting out the timeout.
  if grep -qaE "hit an exception|Engine core initialization failed|Segfault encountered" /tmp/engine.log 2>/dev/null; then
    echo "### ENGINE FAILED -- first error:"
    grep -aE "ZeroDivisionError|cudaError|AssertionError|assert |RuntimeError" /tmp/engine.log | head -5
    exit 1
  fi
  sleep 5
done

switch () {  # $1 = token budget
  curl -s --max-time 300 -X POST "localhost:${PORT}/switch_pd_role" \
    -H 'Content-Type: application/json' \
    -d "{\"backend\":\"deepep_v2\",\"max_num_tokens\":$1,\"max_num_batched_tokens\":$1}"
}
ask () {
  curl -s --max-time 120 "localhost:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
    | grep -o '"text":"[^"]*"' | head -1
}

A0=$(ask); echo "### answer before: $A0"

T0=$(date +%s%N)
R1=$(switch "$DECODE_TOKENS")
T1=$(date +%s%N)
echo "### to decode (${DECODE_TOKENS}): $R1"
echo "### took $(( (T1 - T0) / 1000000 )) ms"
A1=$(ask); echo "### answer after: $A1"

T2=$(date +%s%N)
R2=$(switch "$PREFILL_TOKENS")
T3=$(date +%s%N)
echo "### back to prefill (${PREFILL_TOKENS}): $R2"
echo "### took $(( (T3 - T2) / 1000000 )) ms"
A2=$(ask); echo "### answer after: $A2"

# The verdict is ranks_switched and layers_switched, NOT the timing and NOT the
# output. A no-op switch returns fast and still answers correctly -- that is
# exactly how a broken build looked before this check existed.
echo "############ RESULT ############"
for r in "$R1" "$R2"; do
  rs=$(echo "$r" | grep -o '"ranks_switched":[0-9]*' | grep -o '[0-9]*')
  ls=$(echo "$r" | grep -o '"layers_switched":[0-9]*' | grep -o '[0-9]*')
  if [ "${rs:-0}" = "$EP" ] && [ "${ls:-0}" -gt 0 ]; then
    echo "PASS: ${rs}/${EP} ranks, ${ls} layers rebuilt"
  else
    echo "FAIL: ranks_switched=${rs:-0}/${EP} layers_switched=${ls:-0} -- the switch did nothing"
  fi
done
[ "$A0" = "$A2" ] && echo "PASS: output identical before and after a round trip" \
                  || echo "WARN: output differs across the round trip"

kill $ENGINE_PID 2>/dev/null
