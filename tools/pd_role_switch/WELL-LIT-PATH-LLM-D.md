# Deploying this on llm-d

The deployment guide lives in the autoscaler repository, because deploying on
llm-d and driving the ratio are that side's concerns — this repository holds the
engine mechanism.

**→ `docs/guides/pd-role-switch/` in the workload-variant-autoscaler repo**

It covers:

- why a stock llm-d guide **cannot** host role switching: `llm-d.ai/role` is part
  of each role Deployment's immutable `spec.selector`, so relabelling a pod
  orphans it and forces a cold replacement — demonstrated both ways by
  `check-label-semantics.sh`, no GPUs needed;
- the deployment shape that works — one Deployment for all switchable replicas,
  role kept out of the selector so it is an ordinary mutable pod label;
- a complete GLM-5.2-FP8 manifest, including the settings that are not
  guessable (`VLLM_NIXL_SIDE_CHANNEL_HOST` from `status.podIP`, port 5600,
  `VLLM_SERVER_DEV_MODE=1`, `VLLM_USE_DEEP_GEMM=0`, booting at the prefill
  budget);
- the four-step handover — unlabel, drain, switch, relabel — and why three of
  those four steps belong to the autoscaler rather than to this engine.

What stays here: [README.md](README.md) for the mechanism, its measurements and
reproduction; [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for the failure modes;
[ARCHITECTURE.md](ARCHITECTURE.md) for what it costs and why it matters.
