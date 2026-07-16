# Planning & Pre-flight

Decide whether a training job can run on this cluster *before* submitting it. Covers the four MCP
planning tools: `check_compatibility`, `pre_flight`, `get_cluster_resources`, `estimate_resources`.

---

## 1. Compatibility check

Minimum versions (from the Kubeflow Trainer requirements):

| Component | Minimum |
|---|---|
| Kubernetes | 1.27 |
| `kubectl` | 1.31 (to read the v1alpha1 TrainJob CRD cleanly) |
| Kubeflow Trainer (controller) | 2.2.0 |
| Kubeflow Python SDK | 0.4.0 |

```bash
# Kubernetes version
kubectl version --output=yaml | grep gitVersion

# Trainer CRDs present?
kubectl get crd | grep -E 'trainjobs\.trainer|trainingruntimes\.trainer|clustertrainingruntimes\.trainer'
```

All three CRDs — `trainjobs`, `trainingruntimes`, `clustertrainingruntimes` — must exist. If the
`trainingruntimes` (namespaced) CRD is missing, only cluster-scoped runtimes are usable; that's
fine for most setups but namespace teams can't define their own.

---

## 2. Pre-flight (one-shot readiness)

A single pipeline that fails fast if anything needed to run a TrainJob is missing or unhealthy.
Each stage prints `OK` or the first problem it hits.

```bash
echo "== CRDs ==" ;
  kubectl get crd trainjobs.trainer.kubeflow.org >/dev/null 2>&1 && echo "OK trainjobs" || echo "MISSING trainjobs CRD"
echo "== Controller ==" ;
  kubectl get deploy -n kubeflow-system -l app.kubernetes.io/component=kubeflow-trainer -o name >/dev/null 2>&1 \
    && kubectl rollout status deploy/kubeflow-trainer-controller-manager -n kubeflow-system --timeout=30s \
    || echo "controller not found/ unhealthy"
echo "== JobSet ==" ;
  kubectl get deploy -n kubeflow-system -l app.kubernetes.io/component=jobset -o name >/dev/null 2>&1 \
    && echo "OK jobset" || echo "MISSING jobset (gang scheduling will fail)"
echo "== GPU operator / device plugin ==" ;
  kubectl get pods -A -l app=nvidia-device-plugin -o name | head -1 >/dev/null \
    && echo "OK nvidia-device-plugin" \
    || kubectl get pods -A -l app.kubernetes.io/component=gpu-operator -o name | head -1 >/dev/null \
       && echo "OK gpu-operator" || echo "no GPU plugin detected (CPU-only cluster?)"
echo "== Default runtimes ==" ;
  kubectl get clustertrainingruntime -o name | head -1 >/dev/null \
    && kubectl get clustertrainingruntime || echo "NO runtimes installed — pass --set runtimes.defaultEnabled=true on install"
```

Interpreting failures:

- **MISSING trainjobs CRD** — Trainer isn't installed. See the install steps in [trainjob-runtime-api.md](trainjob-runtime-api.md).
- **controller unhealthy** — `kubectl logs -n kubeflow-system deploy/kubeflow-trainer-controller-manager`.
- **MISSING jobset** — TrainJobs build a JobSet underneath; without the JobSet controller they'll
  never create pods. Install JobSet before proceeding.
- **no GPU plugin** — pods requesting `nvidia.com/gpu` will stay `FailedScheduling`. Either this is
  a CPU-only cluster (use `gloo`, not `nccl`) or the GPU operator isn't installed.
- **NO runtimes** — a TrainJob with no `runtimeRef` target has nothing to render against. Install
  the community defaults or author your own (see [builtin-runtimes.md](builtin-runtimes.md)).

---

## 3. Cluster resource inventory

### Total allocatable GPUs (and where they are)

```bash
# Per-node GPU count (allocatable), plus CPU and memory
kubectl get nodes -o custom-columns='\
NAME:.metadata.name,\
GPU:.status.allocatable.nvidia\.com/gpu,\
CPU:.status.allocatable.cpu,\
MEM:.status.allocatable.memory'
```

A blank `GPU` column means that node has no GPU (or no device plugin exposing it).

### GPUs already in use (requested by running pods)

`kubectl describe nodes` prints an `Allocated resources` block. To get it machine-readably, sum the
`nvidia.com/gpu` requests across non-completed pods:

```bash
kubectl get pods -A --field-selector=status.phase!=Succeeded,status.phase!=Failed -o json \
  | python3 -c "
import sys,json,collections
pods=json.load(sys.stdin)['items']
used=collections.defaultdict(int)
for p in pods:
    node=p['spec'].get('nodeName')
    for c in p['spec']['containers']:
        used[node]+=int(c['resources'].get('requests',{}).get('nvidia.com/gpu',0) or 0)
for n,u in sorted(used.items()):
    if u: print(f'{n}: {u} GPU requested')
"
```

Subtract "requested" from "allocatable" to get the GPUs actually free per node. Note this is
*requests*, not live utilization — a pod that requested a GPU but is idle still holds it.

### Live utilization (needs metrics-server)

```bash
kubectl top nodes    # CPU/memory only; GPUs have no top equivalent without DCGM exporter
```

For real GPU utilization (% SM, memory used) you need the DCGM exporter + Prometheus — out of scope
for `kubectl`, but worth knowing the raw `kubectl` path only tells you *allocation*, not *saturation*.

---

## 4. Resource estimation

Replaces the MCP `estimate_resources` tool. Two tables: how much GPU memory a model needs, and what
batch size that memory supports.

### GPU memory by model size

| Params | Full (bf16) | QLoRA (int4) | Example GPU |
|---|---|---|---|
| 1–3B | ~6–16 GB | ~3–6 GB | T4, RTX 3080 |
| 7–8B | ~24 GB | ~7 GB | A10, RTX 4090 |
| 13B | ~40 GB | ~12 GB | A100-40G |
| 70B | ~140 GB | ~40 GB | A100-80G ×2 |

"Full" = weights + optimizer state + activations for a single rank doing full fine-tuning in bf16.
"QLoRA" = 4-bit weights + small trainable LoRA adapters. Add ~20–30% for activations at larger
sequence lengths.

### Batch size by GPU memory

| GPU memory | Recommended `batch_size` |
|---|---|
| 8 GB | 1–2 |
| 16 GB | 2–4 |
| 24 GB | 4–8 |
| 40 GB+ | 8–16 |

These are fine-tuning figures (LoRA/QLoRA). Pre-training from scratch needs smaller batches and
more gradient accumulation. If you hit `OOMKilled`, halve the batch before reaching for a bigger GPU.

### Quick estimator (model params → memory)

For a rough bf16 single-GPU number: `mem_GB ≈ params_B × 4 × 1.2` (weights 2 bytes + Adam optimizer
state ~6 bytes + activations ~20%). For QLoRA: `mem_GB ≈ params_B × 0.7` (4-bit weights ~0.5 bytes +
small adapters + activations).

---

## 5. Does my model fit? (decision tree)

```
1. How many params?  →  look up GPU memory row above (use QLoRA column if quantizing)
        │
2. Pick GPU type from inventory (step 3). Free GPUs >= 1 with that much memory?
        │  no  → tell the user: cluster has no suitable free GPU. Options:
        │         reduce to QLoRA, shrink batch, wait for running jobs, or add nodes.
        │  yes → continue
        │
3. Multi-node or single-node?
        │  mem fits on 1 GPU   → num_nodes=1, gpu=1   (simplest; pick torchtune for LoRA)
        │  mem fits on N GPUs  → num_nodes=1, gpu=N   (DDP/FSDP; watch /dev/shm — see troubleshooting)
        │  needs >N/node       → num_nodes=M, gpu=N   (true multi-node; needs gang scheduling)
        │
4. CPU-only cluster (no GPU plugin)?
        → use gloo backend, NOT nccl. TorchTune runtimes are nccl-only — fall back to
          CustomTrainer with a gloo script. See submission-modes.md.
        │
5. Ready. Go to submission-modes.md.
```

### Common pitfalls at planning time

- **Confusing `Allocatable` with `Capacity`** — `Capacity` is the node's hardware; `Allocatable` is
  what's left after system daemons reserve some. Always plan against `Allocatable`.
- **Forgetting GPU time-slicing** — if the cluster time-slices GPUs, `nvidia.com/gpu` allocatable
  may exceed the physical count. A pod requesting `gpu=1` may land on a shared GPU; use
  `num_nodes=N, gpu=1` (not `gpu=N`) so `torch.cuda.device_count()` matches the request.
- **No GPU but requested `nvidia.com/gpu`** — immediate `FailedScheduling`. Catch this at step 2,
  not after submission.
