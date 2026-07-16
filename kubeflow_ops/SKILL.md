---
name: kubeflow-ops
description: The complete Kubeflow Trainer v2 skill — both authoring and operating. Covers TrainJob/TrainingRuntime/ClusterTrainingRuntime CRDs and YAML authoring, the Kubeflow Python SDK (TrainerClient, CustomTrainer, BuiltinTrainer, CustomTrainerContainer), per-framework distributed training (PyTorch DDP/FSDP, DeepSpeed, JAX, MLX, XGBoost, Megatron-Core, TorchTune, Flux), and migration from Training Operator v1 — PLUS the operational lifecycle on a live cluster: kubeconfig/prerequisites, cluster readiness & GPU inventory, model-fit resource estimation, the three submission modes, TrainJob status/logs/events monitoring, suspend/resume/cleanup, error-to-fix troubleshooting (OOMKilled, FailedScheduling, ImagePullBackOff, NCCL timeouts), platform fixes (OpenShift SCC, EKS/GKE, GPU tolerations, NCCL env), and environment awareness (Notebook GPU, PVC capacity, namespace egress). A dependency-free kubectl + SDK replacement for the kubeflow/mcp-server. Use when writing/reviewing TrainJob YAML, using the TrainerClient SDK, checking whether a cluster is ready / a model fits the GPUs, submitting/watching/debugging/cleaning up TrainJobs, diagnosing failures, applying platform workarounds, or deciding where a workload should run (Notebook vs TrainJob vs K8s Job).
version: 1.1.0
author: Terence Liu
license: Apache-2.0
tags: [Kubeflow, Kubernetes, Distributed Training, TrainJob, TrainingRuntime, PyTorch, DeepSpeed, JAX, MLX, XGBoost, Megatron-Core, TorchTune, LLM Fine-Tuning, MLOps, Operations, Troubleshooting, GPU, Resource Estimation, OpenShift, EKS, GKE, NCCL, Monitoring, Cluster Readiness, kubectl, Training Operator Migration]
dependencies: [kubectl (>= 1.31), kubeflow (Python SDK), helm, kubeconfig pointing at the target cluster]
---

# Kubeflow Trainer

## What it is

Kubeflow Trainer is a Kubernetes-native platform for distributed AI training and LLM fine-tuning.
It replaces the framework-specific CRDs from the legacy **Training Operator v1**
(`PyTorchJob`, `TFJob`, `MPIJob`, ...) with **one unified CRD**: `TrainJob`. Under the hood it
builds a [JobSet](https://jobset.sigs.k8s.io/) and integrates with Kueue/Volcano/KAI for gang
scheduling and an Arrow/DataFusion data cache for zero-copy GPU streaming.

This skill covers **both halves** of working with it — authoring jobs *and* operating them on a
live cluster — through direct `kubectl` + the Kubeflow Python SDK. It is a dependency-free
replacement for the [`kubeflow/mcp-server`](https://github.com/kubeflow/mcp-server): the MCP's 23
tools are reproduced here as `kubectl`/`TrainerClient` commands, with no extra process to run and
wider coverage (streaming logs, any CRD including Notebook/PVC/NetworkPolicy, no tool-layer
restrictions).

Three CRDs, three personas:

| CRD | Scope | Who writes it |
|---|---|---|
| `TrainJob` | namespace | AI practitioners (usually via the Python SDK, not raw YAML) |
| `TrainingRuntime` | namespace | Platform admins — reusable blueprint for one team/namespace |
| `ClusterTrainingRuntime` | cluster | Platform admins — reusable blueprint shared cluster-wide |

A `TrainJob` references a runtime via `runtimeRef` and supplies the training code/image, node count,
and resources. The runtime supplies the pod template, ML framework policy (`mlPolicy`), and
scheduling policy (`podGroupPolicy`).

## How this skill is organized

| If you want to… | Load this reference |
|---|---|
| Author/review TrainJob/TrainingRuntime YAML (MLPolicy, JobSet template, RuntimePatches, lifecycle fields) | [trainjob-runtime-api.md](references/trainjob-runtime-api.md) |
| Look up a `TrainerClient` method or dataclass field (`CustomTrainer`, `BuiltinTrainer`, `TorchTuneConfig`, backends) | [python-sdk.md](references/python-sdk.md) |
| Get a per-framework guide (PyTorch DDP/FSDP, DeepSpeed, JAX/TPU, MLX, XGBoost, Megatron-Core, TorchTune LoRA) | [builtin-runtimes.md](references/builtin-runtimes.md) |
| Migrate a v1 `PyTorchJob`/`TFJob`/`MPIJob` to v2 | [migration-from-v1.md](references/migration-from-v1.md) |
| Check cluster readiness / GPU inventory / whether a model fits | [planning-and-preflight.md](references/planning-and-preflight.md) |
| Submit (fine-tune / custom script / container image) with scheduling params | [submission-modes.md](references/submission-modes.md) |
| Monitor status/logs/events, suspend/resume, clean up, inspect controller | [monitoring-and-lifecycle.md](references/monitoring-and-lifecycle.md) |
| Diagnose a failure (OOM, scheduling, NCCL, image pull) | [troubleshooting.md](references/troubleshooting.md) |
| Apply platform workarounds (OpenShift SCC, tolerations, NCCL env) | [platform-fixes.md](references/platform-fixes.md) |
| Decide *where* a workload runs (Notebook GPU, PVC capacity, namespace egress) | [environment-awareness.md](references/environment-awareness.md) |

## Install

**Control plane** (Kubernetes >= 1.27, `kubectl` >= 1.31):

```bash
export VERSION=v2.1.0
helm install kubeflow-trainer oci://ghcr.io/kubeflow/charts/kubeflow-trainer \
    --namespace kubeflow-system \
    --create-namespace \
    --version ${VERSION#v} \
    --set runtimes.defaultEnabled=true   # also install the built-in ClusterTrainingRuntimes
```

Verify:

```bash
kubectl get pods -n kubeflow-system   # jobset-controller-manager + kubeflow-trainer-controller-manager
kubectl get clustertrainingruntime
```

**Python SDK**:

```bash
pip install -U kubeflow
```

Full install options (Kustomize, selective runtimes, data cache add-on):
[trainjob-runtime-api.md](references/trainjob-runtime-api.md).

## Core concepts

- **TrainJob** — the job instance. `spec.runtimeRef` (name/kind/apiGroup) + `spec.trainer`
  (image/command/numNodes/resources/env) + optional `spec.initializer` (dataset/model),
  `spec.suspend`, `spec.activeDeadlineSeconds`, `spec.runtimePatches`.
- **TrainingRuntime / ClusterTrainingRuntime** — the blueprint: `spec.mlPolicy` (PlainML/Torch/MPI)
  + `spec.podGroupPolicy` (gang scheduling) + `spec.template` (JobSet `replicatedJobs`: pod templates
  for `dataset-initializer`, `model-initializer`, `node`, `launcher`, ...).
- **CustomTrainer vs BuiltinTrainer** — `CustomTrainer(func=...)` gives full control (your own
  Python function). `BuiltinTrainer(config=TorchTuneConfig(...))` uses a pre-built fine-tuning script
  (currently TorchTune) — you configure hyperparameters, not code.
- **Framework label** — every runtime must carry `trainer.kubeflow.org/framework: <torch|deepspeed|
  jax|mlx|xgboost|torchtune>` or the SDK won't list it.
- **RuntimePatches** — override pod-level fields (nodeSelector, tolerations, volumes,
  serviceAccount) of a referenced runtime *without* mutating the shared runtime object. Multi-owner:
  each contributor writes to its own `manager`-keyed entry.

Full API and YAML detail: [trainjob-runtime-api.md](references/trainjob-runtime-api.md).

## Quick start — PyTorch in 3 steps

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def train_pytorch():
    import os, torch, torch.distributed as dist
    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)
    print(f"WORLD_SIZE={dist.get_world_size()} RANK={dist.get_rank()} "
          f"LOCAL_RANK={os.environ['LOCAL_RANK']}")
    # ... build model, DDP-wrap it, train, save on rank 0 ...
    dist.destroy_process_group()

# 1. Discover available runtimes.
for r in TrainerClient().list_runtimes():
    print(r.name)   # e.g. torch-distributed

# 2. Launch: 4 nodes x 1 GPU each.
job_id = TrainerClient().train(
    trainer=CustomTrainer(
        func=train_pytorch,
        num_nodes=4,
        resources_per_node={"cpu": 3, "memory": "16Gi", "gpu": 1},
    )
)

# 3. Watch it.
TrainerClient().wait_for_job_status(job_id)
for line in TrainerClient().get_job_logs(job_id, follow=True):
    print(line)
```

Everything the training function needs (imports included) must live **inside the function body** —
the SDK ships the function's source, not a closure, to every node.

## Operating on a live cluster

### Prerequisites

The only real prerequisite for operations is a working kubeconfig:

```bash
kubectl config current-context                                  # must print something
kubectl get crd trainjobs.trainer.kubeflow.org                  # Trainer installed
kubectl get pods -n kubeflow-system                             # controllers Running
```

**Identity = your kubeconfig.** There is no separate auth layer (unlike the MCP's bearer/JWT). Every
command runs as the current context's identity, subject to that identity's RBAC.

### The operational lifecycle

```
1. PLAN        Is the cluster ready? How much GPU is free? Does my model fit?
                 └─ planning-and-preflight.md
2. DISCOVER    What runtimes exist? What jobs are already running?
                 └─ monitoring-and-lifecycle.md
3. SUBMIT      fine-tune / script / image  (preview, then confirm)
                 └─ submission-modes.md
4. MONITOR     status, logs -f, events
                 └─ monitoring-and-lifecycle.md
5. DEBUG       match error → fix  (+ platform-fixes if needed)
                 └─ troubleshooting.md, platform-fixes.md
6. CLEAN UP    delete / suspend / resume
                 └─ monitoring-and-lifecycle.md
```

Before submitting anything, run the pre-flight check in [planning-and-preflight.md](references/planning-and-preflight.md).

### MCP tool → kubectl/SDK quick reference

The `kubeflow/mcp-server` exposes 23 tools. Here is what to run instead:

| MCP tool | Do this instead |
|---|---|
| `check_compatibility` / `pre_flight` | `kubectl get crd \| grep trainer` + readiness pipeline — [planning-and-preflight.md](references/planning-and-preflight.md) |
| `get_cluster_resources` | `kubectl describe nodes` → parse `Allocatable: nvidia.com/gpu` |
| `estimate_resources` | model-size → GPU-memory → batch-size table — [planning-and-preflight.md](references/planning-and-preflight.md) |
| `list_training_jobs` / `get_training_job` | `kubectl get trainjobs -A` / `kubectl describe trainjob <name>` |
| `list_runtimes` / `get_runtime` | `kubectl get clustertrainingruntime` |
| `get_training_logs` | `kubectl logs -f <pod>` (streams; the MCP's `follow=True` does **not**) |
| `get_training_events` | `kubectl get events --field-selector involvedObject.name=<name>` |
| `wait_for_training` | poll loop on `.status.conditions` — [monitoring-and-lifecycle.md](references/monitoring-and-lifecycle.md) |
| `fine_tune` / `run_custom_training` / `run_container_training` | `TrainerClient().train(trainer=BuiltinTrainer\|CustomTrainer\|CustomTrainerContainer(...))` — [submission-modes.md](references/submission-modes.md) |
| `delete_training_job` / `update_training_job` | `kubectl delete trainjob <name>` / `kubectl patch trainjob <name> -p '{"spec":{"suspend":true}}'` |
| `inspect_crd` / `inspect_controller` | `kubectl get crd ... -o yaml` / `kubectl logs -n kubeflow-system deploy/kubeflow-trainer-controller-manager` |
| `create/patch/delete_runtime` | `kubectl apply/patch/delete clustertrainingruntime` |

`health_check` and `get_server_logs` are about the MCP process itself — no equivalent, omitted.

## TrainerClient API (Kubeflow SDK)

| Method | Purpose |
|---|---|
| `list_runtimes() -> list[Runtime]` | Enumerate runtimes (namespaced preferred over cluster-scoped on clash) |
| `get_runtime(name) -> Runtime` | Fetch one runtime by name |
| `get_runtime_packages(runtime)` | Print installed packages (+ GPUs) for a runtime's node |
| `train(runtime=None, initializer=None, trainer=None, options=None) -> str` | Create a TrainJob; returns its generated name |
| `list_jobs(runtime=None) -> list[TrainJob]` | List TrainJobs, optionally filtered by runtime |
| `get_job(name) -> TrainJob` | Fetch one TrainJob, including `.steps` and `.trainer_status` |
| `get_job_logs(name, step="node-0", follow=False) -> Iterator[str]` | Stream/collect logs from a step |
| `get_job_events(name) -> list[Event]` | Pod-level events when logs don't explain a failure |
| `wait_for_job_status(name, status, timeout, polling_interval, callbacks) -> TrainJob` | Block until a target status |
| `delete_job(name)` | Delete the TrainJob and its resources |

Full dataclass field reference (`CustomTrainer`, `BuiltinTrainer`, `TorchTuneConfig`, `LoraConfig`,
`Initializer`, the three `*BackendConfig`s): [python-sdk.md](references/python-sdk.md).

## Common workflows

### Iterate locally before touching a cluster

```python
from kubeflow.trainer import TrainerClient, CustomTrainer, LocalProcessBackendConfig
client = TrainerClient(backend_config=LocalProcessBackendConfig())
job_id = client.train(trainer=CustomTrainer(func=my_train_fn))
```

Swap `LocalProcessBackendConfig()` → `ContainerBackendConfig(...)` (multi-node in containers) or
`KubernetesBackendConfig(namespace=...)` (real cluster) — the interface is identical across all three.

### Read progress/metrics without scraping logs (Trainer v2.2.0+)

```python
status = TrainerClient().get_job("my-trainjob")
print(status.trainer_status.progress_percentage, status.trainer_status.metrics)
```

Zero-code integration with HuggingFace `Trainer` (`transformers>=4.57.0`); manual
`update_trainjob_status()` for custom loops. Requires the `TrainJobStatus` alpha feature gate.

## Confirm-before-submit

There is no MCP `confirmed=True` gate, so treat this as a hard behavioral rule: before submitting
any TrainJob, render the full config (resolved SDK args or YAML), show it to the user, and only
proceed on explicit confirmation. Never submit a training job in the same turn you first produced it.

## Common issues

- **`NameError` on a training node** — every import used inside `CustomTrainer(func=...)` must be
  *inside* the function body; the SDK ships source, not a closure.
- **Custom runtime not recognized by the SDK** — missing the `trainer.kubeflow.org/framework` label.
- **Multi-node Megatron-Core / high-NCCL-communicator jobs hang silently** — default 64 MB `/dev/shm`
  is too small (~33 MB/communicator). Mount a memory-backed emptyDir on `/dev/shm`. See
  [troubleshooting.md](references/troubleshooting.md).
- **`torchrun --nproc_per_node=auto` wrong process count under GPU time-slicing** — `auto` reads
  `torch.cuda.device_count()`, not the pod's GPU request. Use `num_nodes=N, gpu=1` (multi-node)
  instead of `num_nodes=1, gpu=N` (multi-GPU).
- **`webhook-validation: disabled` label leaking** — only for the community's pre-validated built-in
  runtimes; never carry it on a runtime you author or modify.

Full error→fix table and recovery actions: [troubleshooting.md](references/troubleshooting.md).

## When to use vs alternatives

**Use Kubeflow Trainer when**: you're on Kubernetes, need one API across PyTorch/DeepSpeed/JAX/MLX/
XGBoost/Megatron-Core, want a Python-first workflow that doesn't require YAML by hand, or need
standardized LLM fine-tuning (TorchTune) with LoRA baked in.

**Use raw Kubernetes Jobs/JobSet instead when**: you need something Trainer's runtime/ML-policy
abstraction doesn't cover and don't want the extra CRD layer.

**Use Ray Train instead when**: you're not committed to Kubernetes-native APIs and want Python-native
cluster orchestration with built-in hyperparameter tuning.

## Resources

- Docs: https://trainer.kubeflow.org
- Trainer repo (CRDs, controller, runtimes, examples): https://github.com/kubeflow/trainer
- Python SDK: https://github.com/kubeflow/sdk
- MCP server this skill replaces: https://github.com/kubeflow/mcp-server
- Announcement: https://blog.kubeflow.org/trainer/intro/
