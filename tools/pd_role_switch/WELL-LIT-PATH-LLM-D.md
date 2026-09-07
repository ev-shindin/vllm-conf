# Deploying GLM-5.2 on llm-d with role switching

How to run GLM-5.2-FP8 as an llm-d P/D deployment whose replicas can change role
in place, and why it cannot be done by adding a flag to the standard guide.

Everything below about llm-d's own shape was read off deployments running on a
production cluster, not from documentation.

---

## 1. How llm-d organises a P/D deployment

A guide is a Helm release that produces, per role, one Deployment of vLLM pods,
plus one endpoint picker and one InferencePool for the whole guide:

```
InferencePool  <guide>                       selector: llm-d.ai/guide=<guide>
                                             endpointPickerRef -> Service <guide>-epp:9002
Deployment     <guide>-nvidia-gpu-vllm-prefill
Deployment     <guide>-nvidia-gpu-vllm-decode
Deployment     <guide>-epp                   containers: epp + envoy-proxy sidecar
Gateway        llm-d-inference-gateway-istio
```

Every engine pod carries the same six labels:

```yaml
llm-d.ai/guide: <guide>             # InferencePool membership
llm-d.ai/role: prefill | decode     # how the endpoint picker splits the pool
llm-d.ai/model: <model>
llm-d.ai/engine-type: vllm
llm-d.ai/accelerator-vendor: nvidia
llm-d.ai/accelerator-variant: gpu
```

The pool selects on `llm-d.ai/guide`; the picker's `prefill-filter` and
`decode-filter` split that set by `llm-d.ai/role`. Routing is therefore decided
by a **label on the pod**, never by anything the engine reports.

Two details of the engine pod that matter and are easy to miss:

```yaml
- name: VLLM_NIXL_SIDE_CHANNEL_HOST      # must be the pod's own IP
  valueFrom: { fieldRef: { fieldPath: status.podIP } }
ports:
  - { containerPort: 8000, name: modelserver }
  - { containerPort: 5600, name: nixl }    # the KV side channel
```

and llm-d gives **both** roles `kv_role: kv_both`:

```
--kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_both",
                      "kv_connector_extra_config":{"kv_lease_duration":300}}
```

## 2. Why the standard shape cannot switch roles

`llm-d.ai/role` is part of each Deployment's **selector**:

```json
"selector": {"matchLabels": {"llm-d.ai/guide": "...", "llm-d.ai/role": "prefill", ...}}
```

A Deployment's `spec.selector` is immutable, and a Deployment owns exactly the
pods matching it. So relabelling a running pod from `prefill` to `decode` does
not move it between roles — it removes it from its owner, which immediately
creates a replacement. You get a new cold replica and an orphan, which is the
opposite of the point.

**So the switch cannot be bolted onto the standard guide.** The workload has to
be shaped so that the role label is *not* part of any selector.

## 3. The switchable shape: one Deployment, role as a mutable label

Put every switchable replica in a single Deployment and keep the role out of its
selector. The role then becomes an ordinary mutable pod label that a controller
can change, while the Deployment keeps maintaining N pods regardless.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: glm52-switchable
  labels:
    llm-d.ai/guide: glm52-switchable
spec:
  replicas: 4                      # total capacity; the RATIO is labels, not replicas
  selector:
    matchLabels:                   # NOTE: no llm-d.ai/role here. That is the point.
      llm-d.ai/guide: glm52-switchable
      llm-d.ai/model: GLM-5.2-FP8
      llm-d.ai/engine-type: vllm
  template:
    metadata:
      labels:
        llm-d.ai/guide: glm52-switchable
        llm-d.ai/model: GLM-5.2-FP8
        llm-d.ai/engine-type: vllm
        llm-d.ai/accelerator-vendor: nvidia
        llm-d.ai/accelerator-variant: gpu
        llm-d.ai/role: prefill     # starting role only; mutable from here on
    spec:
      containers:
        - name: modelserver
          image: <image with this branch, or stock v0.28.0 + README step 1b>
          command: ["vllm", "serve"]
          args:
            - zai-org/GLM-5.2-FP8
            - --trust-remote-code
            - --port=8000
            - --data-parallel-size=8
            - --data-parallel-size-local=8
            - --enable-expert-parallel
            - --tensor-parallel-size=1
            - --all2all-backend=deepep_v2
            - --block-size=64
            - --kv-cache-dtype=fp8
            - --max-model-len=8192
            - --gpu-memory-utilization=0.90
            - --max-num-seqs=128
            # Boot at the PREFILL budget. A switch may lower it but never raise
            # it past the launch value -- this figure also sizes the input
            # buffers, the compile range and the attention workspace at init.
            - --max-num-batched-tokens=2048
            - --kv-transfer-config
            - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
          env:
            - name: VLLM_NIXL_SIDE_CHANNEL_HOST
              valueFrom: { fieldRef: { fieldPath: status.podIP } }
            - { name: VLLM_SERVER_DEV_MODE, value: "1" }   # /switch_pd_role is a dev route
            - { name: VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE, value: "1" }  # 1 node/replica
            - { name: VLLM_USE_DEEP_GEMM, value: "0" }     # deepep_v2 pads; deep_gemm asserts
            - { name: VLLM_PD_KEEP_PREVIOUS, value: "1" }  # park the outgoing buffer
          ports:
            - { containerPort: 8000, name: modelserver }
            - { containerPort: 5600, name: nixl }
          resources:
            limits: { nvidia.com/gpu: "8", rdma/ib: "8", cpu: "64", memory: 1200Gi }
          startupProbe:                # GLM-5.2 is ~753 GB; allow a long load
            httpGet: { path: /v1/models, port: modelserver }
            failureThreshold: 120
            periodSeconds: 30
```

The InferencePool is unchanged from the standard guide — it selects on
`llm-d.ai/guide`, which never moves:

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata: { name: glm52-switchable }
spec:
  selector:
    matchLabels: { llm-d.ai/guide: glm52-switchable }
  targetPorts: [{ number: 8000 }]
  endpointPickerRef:
    kind: Service
    name: glm52-switchable-epp
    port: { number: 9002 }
    failureMode: FailOpen
```

Reuse the guide's EPP deployment and config as-is. It needs no changes: it
already filters on `llm-d.ai/role`, and it does not care who owns the pod.

### Consequences to accept

- **Boot every replica at the prefill budget.** `max_num_batched_tokens` sizes
  buffers at init, so lowering it later is safe and raising it past the launch
  value is not.
- **A rollout resets roles** to the template's starting value. The controller
  has to reassert the ratio after one.
- **The ratio is labels, not replica counts.** Scaling the Deployment changes
  total capacity; relabelling changes the split.

## 4. Changing a replica's role

Four steps. The engine call is one of them, and the other three are why this
belongs to a controller and not to the engine.

```bash
POD=<a pod of the deployment>

# 1. take it out of BOTH roles so the picker stops selecting it. Membership is
#    unaffected: the pool selects on llm-d.ai/guide, which does not change.
kubectl label pod "$POD" llm-d.ai/role-

# 2. let in-flight requests finish (the engine survives a switch under load,
#    so this is for tidiness, not safety)

# 3. switch the engine, and CHECK IT. A switch that did nothing also returns
#    quickly and still answers correctly.
kubectl exec "$POD" -- curl -s -X POST localhost:8000/switch_pd_role \
  -H 'Content-Type: application/json' \
  -d '{"backend":"deepep_v2","max_num_tokens":128,"max_num_batched_tokens":128}'
# expect ranks_switched == the rank count, layers_switched > 0

# 4. put it back in, as the new role
kubectl label pod "$POD" llm-d.ai/role=decode
```

Between steps 1 and 4 the replica serves nothing. That is the honest cost of
doing it safely: a few hundred milliseconds, against the minutes a replacement
replica would take.

**Do not have the engine relabel itself.** It would need Kubernetes write
credentials in every inference pod to mutate cluster state about itself. The
engine owns the mechanism; the controller owns the decision, the label and the
ordering.

## 5. What is proven and what is not

Proven on hardware (see [README.md](README.md)):

- the switch itself, 16/16 ranks at EP=16 across two nodes, output identical
- KV transfers between two GLM-5.2 engines and **still transfers after both
  change role**, in the reversed direction
- both role buffers resident for 286 MiB, and a return switch allocating 0 MiB

Not yet run: this deployment shape end to end under a live endpoint picker. The
pieces are each verified, the assembly is not. The controller in step 4 does not
exist yet either — today it is a manual `kubectl label`.
