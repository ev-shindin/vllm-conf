#!/bin/bash
# Reproduce the P/D role-switch results: timing, timing under load, and memory.
#
# Run on every node. Node 0 serves the API and drives the test, the rest are
# headless followers.
#
#   # node 0
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 ./run_switch_test.sh
#   # node 1
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 ./run_switch_test.sh
#
# Single node: NODES=1 ./run_switch_test.sh
#
# Prints a table of what it measured next to the published numbers, so a run
# either reproduces them or shows where it differs. If something fails, see
# TROUBLESHOOTING.md.
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
LOAD=${LOAD:-24}
EP=$(( NODES * GPUS_PER_NODE ))

# Hybrid mode inverts with topology: 1 on one node, 0 across nodes.
# TROUBLESHOOTING.md explains both failures.
if [ "$NODES" -gt 1 ]; then HYBRID=${HYBRID:-0}; else HYBRID=${HYBRID:-1}; fi
export VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=$HYBRID
# /switch_pd_role is attached by register_vllm_dev_api_routers, which the server
# only calls when this is set (entrypoints/launchers/api_server/routers.py:34).
# Without it the engine serves normally and every switch returns 404 -- which
# reads as a 6 ms switch with no ranks, not as a missing route.
export VLLM_SERVER_DEV_MODE=1
# deep_gemm is forced from two places; the flag is omitted below and this is
# the other one. deepep_v2 pads its layout and deep_gemm rejects the padding.
export VLLM_USE_DEEP_GEMM=0
# Park the outgoing buffer so captured CUDA graphs stay valid. Without this the
# handle cache (a WeakValueDictionary) lets it go and the return switch is a
# full rebuild instead of 378 ms.
export VLLM_PD_KEEP_PREVIOUS=${VLLM_PD_KEEP_PREVIOUS:-1}
# No VLLM_PD_ALLOW_CUDAGRAPHS here on purpose. check_switchable only consults
# the cudagraph refusal when the outgoing buffer is NOT retained, so keeping it
# is already sufficient -- setting the bypass as well would imply a test-only
# escape hatch is required for normal use.

mem_used () {  # driver-level MiB on GPU 0, same source as torch.cuda.mem_get_info.
               # GPU 0 only, and it counts every process on that device -- read
               # it on a node running nothing else.
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' '
}

echo "### EP=${EP} (${NODES} x ${GPUS_PER_NODE}), node_rank=${NODE_RANK}, hybrid=${HYBRID}"
python3 -c "import importlib.metadata as m; print('### nccl:', m.version('nvidia-nccl-cu12'))" 2>/dev/null
python3 -c "import deep_ep, importlib.metadata as m; print('### deep_ep:', m.version('deep_ep'), 'ElasticBuffer:', hasattr(deep_ep,'ElasticBuffer'))" 2>/dev/null

if [ "$NODE_RANK" = "0" ]; then
  ROLEFLAGS="--port ${PORT}"
else
  # start-rank goes only on followers: vLLM infers hybrid LB from
  # "start-rank set AND not headless", and the leader is rank 0 already.
  ROLEFLAGS="--headless --data-parallel-start-rank $(( NODE_RANK * GPUS_PER_NODE ))"
fi
DPFLAGS=""
if [ "$NODES" -gt 1 ]; then
  DPFLAGS="--data-parallel-address ${MASTER_ADDR} --data-parallel-rpc-port ${DP_RPC_PORT:-5555}"
fi

# No --moe-backend on purpose: vLLM's oracle then picks an expert backend that
# tolerates deepep_v2's padded layout.
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
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 2>&1 | tee /tmp/engine.log &
ENGINE_PID=$!

if [ "$NODE_RANK" != "0" ]; then
  wait $ENGINE_PID          # followers serve no API; hold them open
  exit 0
fi

echo "### waiting for the API (a cold multi-node load can take 20+ min)"
UP=0
for i in $(seq 1 480); do
  if curl -sf "localhost:${PORT}/health" >/dev/null 2>&1; then UP=1; break; fi
  # A crash presents as a hang here: this shell outlives the engine. Fail on
  # the errors that actually occur instead of waiting out the timeout.
  if grep -qaE "hit an exception|Engine core initialization failed|Segfault encountered" /tmp/engine.log 2>/dev/null; then
    echo "### ENGINE FAILED. First error:"
    grep -aE "ZeroDivisionError|cudaError|AssertionError|assert |RuntimeError" /tmp/engine.log | head -5
    echo "### see TROUBLESHOOTING.md"
    kill $ENGINE_PID 2>/dev/null; exit 1
  fi
  sleep 5
done
[ "$UP" = "1" ] || { echo "### API never came up"; kill $ENGINE_PID 2>/dev/null; exit 1; }

switch () {
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
ok_ranks () {  # $1 = response; echoes "ranks/layers"
  # A data-parallel response carries layers_switched once PER RANK, so an
  # unbounded grep returns "75\n75\n..." -- which then fails the integer test in
  # the verdict and reports a perfectly good 8/75 switch as FAIL. Take the first
  # of each so the verdict compares numbers rather than a multi-line string.
  local r l
  r=$(printf '%s' "$1" | grep -o '"ranks_switched":[0-9]*' | grep -o '[0-9]*' | head -1)
  l=$(printf '%s' "$1" | grep -o '"layers_switched":[0-9]*' | grep -o '[0-9]*' | head -1)
  printf '%s/%s' "${r:-0}" "${l:-0}"
}

A0=$(ask)
MEM_BOOT=$(mem_used)
echo "### booted. answer: $A0 ; GPU0 used ${MEM_BOOT} MiB"

# --- idle switches -----------------------------------------------------------
T0=$(date +%s%N); R1=$(switch "$DECODE_TOKENS"); T1=$(date +%s%N)
MS_BUILD=$(( (T1 - T0) / 1000000 )); MEM_BOTH=$(mem_used)
echo "### -> decode(${DECODE_TOKENS}): $(ok_ranks "$R1") in ${MS_BUILD} ms ; used ${MEM_BOTH} MiB"

T2=$(date +%s%N); R2=$(switch "$PREFILL_TOKENS"); T3=$(date +%s%N)
MS_REUSE=$(( (T3 - T2) / 1000000 )); MEM_BACK=$(mem_used)
echo "### -> prefill(${PREFILL_TOKENS}): $(ok_ranks "$R2") in ${MS_REUSE} ms ; used ${MEM_BACK} MiB"
A1=$(ask)

# --- switch with requests in flight -----------------------------------------
MS_LOAD=0; OKN=0
if [ "$LOAD" != "0" ]; then
  rm -rf /tmp/pdload; mkdir -p /tmp/pdload; PIDS=""
  for r in $(seq 1 "$LOAD"); do
    ( curl -s --max-time 600 "localhost:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Write a long story about a robot.\",\"max_tokens\":256,\"temperature\":0,\"ignore_eos\":true}" \
        -o "/tmp/pdload/r${r}.json" 2>/dev/null ) &
    PIDS="$PIDS $!"      # collect PIDs; a bare wait would also wait on the engine
  done
  sleep 2                # let them get in flight
  T4=$(date +%s%N); R3=$(switch "$DECODE_TOKENS"); T5=$(date +%s%N)
  MS_LOAD=$(( (T5 - T4) / 1000000 ))
  echo "### -> decode under load: $(ok_ranks "$R3") in ${MS_LOAD} ms"
  wait $PIDS 2>/dev/null
  OKN=$(grep -l '"text"' /tmp/pdload/*.json 2>/dev/null | wc -l)
  switch "$PREFILL_TOKENS" >/dev/null
fi

# --- results -----------------------------------------------------------------
R1RL=$(ok_ranks "$R1"); R2RL=$(ok_ranks "$R2")
echo
echo "############ RESULT ############"
# The published figures are EP=16 across two nodes. Comparing a single-node
# EP=8 run against them invites the wrong conclusion, so say which column you
# are being shown and only claim a reproduction when the topology matches.
if [ "$NODES" -gt 1 ]; then
  REF="published (2x8 H200, EP=16)"
  REF_BUILD="1464-1587 ms"; REF_REUSE="378-387 ms"; REF_LOAD="4279 ms"
  REF_ONE="126257 MiB"; REF_BOTH="126543 MiB"; REF_RETAIN="286 MiB (at 128 tokens)"
else
  REF="published (1x8 H200, EP=8)"
  REF_BUILD="n/a"; REF_REUSE="277-656 ms"; REF_LOAD="n/a"
  REF_ONE="n/a"; REF_BOTH="n/a"; REF_RETAIN="234 MiB (at 128 tokens)"
  echo "### NOTE: single node. The headline 378 ms figure is EP=16 on two nodes;"
  echo "### this column is the intranode EP=8 reference instead."
fi
printf '%-34s %-16s %s\n' "measurement" "this run" "$REF"
printf '%-34s %-16s %s\n' "switch, fresh build" "${MS_BUILD} ms" "$REF_BUILD"
printf '%-34s %-16s %s\n' "switch, reuse parked buffer" "${MS_REUSE} ms" "$REF_REUSE"
[ "$LOAD" != "0" ] && \
printf '%-34s %-16s %s\n' "switch under load" "${MS_LOAD} ms" "$REF_LOAD"
[ "$LOAD" != "0" ] && \
printf '%-34s %-16s %s\n' "in-flight requests completed" "${OKN}/${LOAD}" "24/24, 0 failed"
printf '%-34s %-16s %s\n' "GPU0 used, one buffer" "${MEM_BOOT} MiB" "$REF_ONE"
printf '%-34s %-16s %s\n' "GPU0 used, both buffers" "${MEM_BOTH} MiB" "$REF_BOTH"
printf '%-34s %-16s %s\n' "retained buffer cost" "$(( MEM_BOTH - MEM_BOOT )) MiB" "$REF_RETAIN"
printf '%-34s %-16s %s\n' "reuse switch allocation" "$(( MEM_BACK - MEM_BOTH )) MiB" "0 MiB"
echo
# ranks/layers is the verdict: a no-op switch also returns fast and answers
# correctly, so timing and output alone cannot tell you it worked.
FAIL=0
for rl in "$R1RL" "$R2RL"; do
  r=${rl%%/*}; l=${rl##*/}
  if [ "${r:-0}" = "$EP" ] && [ "${l:-0}" -gt 0 ] 2>/dev/null; then
    echo "PASS  ${r}/${EP} ranks, ${l} layers rebuilt"
  else
    echo "FAIL  ranks=${r:-0}/${EP} layers=${l:-0} -- the switch did nothing (TROUBLESHOOTING.md)"
    FAIL=1
  fi
done
[ "$A0" = "$A1" ] && echo "PASS  output identical across the round trip" \
                  || echo "WARN  output differs across the round trip"

kill $ENGINE_PID 2>/dev/null
exit $FAIL
