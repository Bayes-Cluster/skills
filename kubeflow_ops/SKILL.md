---
name: kubeflow-ops
description: Operational runbook for running Kubeflow Trainer v2 TrainJobs on a live Kubernetes cluster — cluster readiness checks, GPU/resource inventory, model-fit estimation, the three submission modes (BuiltinTrainer fine-tune, CustomTrainer script, CustomTrainerContainer image), TrainJob status monitoring and log/event inspection, suspend/resume/cleanup, error-to-fix troubleshooting, and platform-specific fixes (OpenShift SCC, EKS/GKE, GPU tolerations, NCCL env). Replaces the kubeflow/mcp-server's 23 tools with direct kubectl + Kubeflow SDK commands. Use when you need to check whether the cluster is ready for a training job, estimate if a model fits the available GPUs, submit/watch/debug/clean up TrainJobs, diagnose failures (OOMKilled, FailedScheduling, ImagePullBackOff, NCCL timeouts), or apply platform workarounds — as opposed to authoring TrainJob/TrainingRuntime YAML (use the kubeflow-trainer skill for that).
version: 1.0.0
author: Terence Liu
license: Apache-2.0
tags: [Kubeflow, Kubernetes, TrainJob, Operations, Troubleshooting, GPU, Resource Estimation, OpenShift, EKS, GKE, NCCL, Monitoring, Cluster Readiness, kubectl]
dependencies: [kubectl (>= 1.31), kubeflow (Python SDK), kubeconfig pointing at the target cluster]
---

# Kubeflow Ops

## What it is

The **operational companion** to the `kubeflow-trainer` skill. Where `kubeflow-trainer`
teaches you how to *author* `TrainJob` / `TrainingRuntime` YAML and use the SDK's field-level
API, this skill teaches you how to *operate* a live cluster: check it's ready, see if your model
fits, submit, watch, debug, and clean up — all via `kubectl` and the Kubeflow Python SDK.

It is a direct, dependency-free replacement for the [`kubeflow/mcp-server`](https://github.com/kubeflow/mcp-server).
The MCP server's 23 tools are reproduced here as `kubectl` / `TrainerClient` commands. Two reasons
you'd use this skill instead of running the MCP:

1. **No extra process** — the MCP needs `kubeflow-mcp serve` running and authenticated. This skill
   drives the cluster directly through the `kubeconfig` Claude Code already inherits.
2. **Wider coverage, fewer limits** — `kubectl logs -f` streams (the MCP's `follow=True` does not);
   raw YAML/`kubectl apply` carries no `env`/ownership-label restrictions the tool layer imposes;
   and this skill additionally covers Notebook/PVC/NetworkPolicy awareness the MCP never touches.

| If you want to… | Use this skill (`kubeflow-ops`) | Use `kubeflow-trainer` |
|---|---|---|
| Author or review TrainJob/TrainingRuntime YAML | | ✅ |
| Look up a `TrainerClient` method signature or dataclass field | | ✅ |
| Get a per-framework guide (PyTorch DDP/FSDP, DeepSpeed, Megatron, …) | | ✅ |
| Migrate a v1 `PyTorchJob`/`TFJob` to v2 | | ✅ |
| Check the cluster is ready / has GPUs / your model fits | ✅ | |
| Submit, watch, debug, suspend, or clean up a TrainJob | ✅ | |
| Diagnose a failure (OOM, scheduling, NCCL, image pull) | ✅ | |
| Apply platform workarounds (OpenShift SCC, tolerations, NCCL env) | ✅ | |
| Inspect the current Notebook's GPU / PVC capacity / namespace egress | ✅ | |

## Prerequisites

Everything below runs as the local user, so **the only real prerequisite is a working kubeconfig**
pointing at a cluster that has Kubeflow Trainer v2 installed.

```bash
# 1. kubeconfig present and pointing at the right context
kubectl config current-context          # must print something
kubectl config view --minify --output 'jsonpath={..namespace}'; echo   # default namespace

# 2. kubectl can actually reach the cluster
kubectl version --short 2>/dev/null || kubectl version --output=yaml

# 3. Trainer CRDs are installed
kubectl get crd trainjobs.trainer.kubeflow.org clustertrainingruntimes.trainer.kubeflow.org

# 4. (optional) Python SDK, if you prefer SDK over kubectl for submission
python3 -c "import kubeflow.trainer; print('ok')" 2>/dev/null || pip install -U kubeflow
```

If step 1 fails, the agent cannot reach the cluster — stop and tell the user to fix their
`~/.kube/config` (or, if the process runs inside a pod, ensure the ServiceAccount has RBAC). If
step 3 fails, Trainer isn't installed — see the `kubeflow-trainer` skill's Install section.

> **Identity = your kubeconfig.** There is no separate auth layer. Every command here runs as
> whatever identity the current kubeconfig context carries, subject to that identity's RBAC. A
> read-only user will see `Forbidden` on mutating commands; that is the cluster enforcing RBAC,
> not this skill.

## The operational lifecycle

```
        ┌─────────── 1. PLAN ───────────┐
        │ Is the cluster ready?          │   references/planning-and-preflight.md
        │ How much GPU is free?          │
        │ Does my model fit?             │
        └───────────────┬───────────────┘
                        ▼
        ┌─────────── 2. DISCOVER ───────┐
        │ What runtimes exist?           │   references/monitoring-and-lifecycle.md
        │ What jobs are already running? │
        └───────────────┬───────────────┘
                        ▼
        ┌─────────── 3. SUBMIT ─────────┐
        │ fine-tune / script / image     │   references/submission-modes.md
        │ (preview, then confirm)        │
        └───────────────┬───────────────┘
                        ▼
   ┌─── 4. MONITOR ◄───────────────────────┐
   │   status, logs -f, events              │   references/monitoring-and-lifecycle.md
   └───────────────┬────────────────────────┘
                   │ failed?
                   ▼
        ┌─────────── 5. DEBUG ──────────┐
        │ match error → fix              │   references/troubleshooting.md
        │ (+ platform-fixes if needed)   │   references/platform-fixes.md
        └───────────────┬───────────────┘
                        ▼
        ┌─────────── 6. CLEAN UP ───────┐
        │ delete / suspend / resume       │   references/monitoring-and-lifecycle.md
        └───────────────────────────────┘
```

## MCP tool → kubectl/SDK command quick reference

The `kubeflow/mcp-server` exposes 23 tools. Here is what to run instead. (`health_check` and
`get_server_logs` are about the MCP process itself and have no equivalent — they're omitted.)

### Planning

| MCP tool | Do this instead |
|---|---|
| `check_compatibility` | `kubectl get crd \| grep trainer` + version gate — [planning-and-preflight.md](references/planning-and-preflight.md) |
| `pre_flight` | combined readiness one-liner (CRDs + controller pods + GPU operator) — [planning-and-preflight.md](references/planning-and-preflight.md) |
| `get_cluster_resources` | `kubectl describe nodes` → parse `Allocatable: nvidia.com/gpu` — [planning-and-preflight.md](references/planning-and-preflight.md) |
| `estimate_resources` | model-size → GPU-memory → batch-size table — [planning-and-preflight.md](references/planning-and-preflight.md) |

### Discovery & Monitoring

| MCP tool | Do this instead |
|---|---|
| `list_training_jobs` | `kubectl get trainjobs -A` (or `-n <ns>`) — [monitoring-and-lifecycle.md](references/monitoring-and-lifecycle.md) |
| `get_training_job` | `kubectl get trainjob <name> -o yaml` / `kubectl describe trainjob <name>` |
| `list_runtimes` | `kubectl get clustertrainingruntime` + `kubectl get trainingruntime -A` |
| `get_runtime` | `kubectl get clustertrainingruntime <name> -o yaml` |
| `get_training_logs` | `kubectl logs -f <pod>` or `trainer.get_job_logs(follow=True)` (SDK streams; MCP does not) |
| `get_training_events` | `kubectl get events --field-selector involvedObject.name=<name>` / `kubectl describe trainjob <name>` |
| `wait_for_training` | poll loop: `kubectl get trainjob <name> -o jsonpath='{.status.conditions}'` — [monitoring-and-lifecycle.md](references/monitoring-and-lifecycle.md) |

### Training (submission)

| MCP tool | Do this instead |
|---|---|
| `fine_tune` | `TrainerClient().train(trainer=BuiltinTrainer(config=TorchTuneConfig(...)))` — [submission-modes.md](references/submission-modes.md) |
| `run_custom_training` | `TrainerClient().train(trainer=CustomTrainer(func=...))` |
| `run_container_training` | `TrainerClient().train(trainer=CustomTrainerContainer(image=...))` |

### Lifecycle & Platform

| MCP tool | Do this instead |
|---|---|
| `delete_training_job` | `kubectl delete trainjob <name>` — [monitoring-and-lifecycle.md](references/monitoring-and-lifecycle.md) |
| `update_training_job` | `kubectl patch trainjob <name> --type merge -p '{"spec":{"suspend":true}}'` |
| `inspect_crd` | `kubectl get crd trainjobs.trainer.kubeflow.org -o yaml` |
| `inspect_controller` | `kubectl logs -n kubeflow-system deploy/kubeflow-trainer-controller-manager` |
| `create_runtime` / `patch_runtime` / `delete_runtime` | `kubectl apply -f runtime.yaml` / `kubectl patch` / `kubectl delete clustertrainingruntime <name>` |
| `health_check` | `kubectl get pods -n kubeflow-system` (controller + jobset manager Running) |

## Confirm-before-submit discipline

The MCP enforces a `confirmed=True` gate on every mutating call. This skill cannot force that, so
it is a **behavioral rule** instead: before applying any TrainJob, render the intended config (the
YAML or the SDK call's resolved arguments), show it to the user, and only proceed on explicit
confirmation. Never submit a training job in the same turn you first produced it.

## References

| File | Covers | When to load |
|---|---|---|
| [planning-and-preflight.md](references/planning-and-preflight.md) | Readiness checks, GPU inventory, model-fit estimation | Before submitting anything |
| [submission-modes.md](references/submission-modes.md) | The three submission modes, scheduling params, confirm flow | When creating a TrainJob |
| [monitoring-and-lifecycle.md](references/monitoring-and-lifecycle.md) | Status/logs/events, suspend/resume, cleanup, controller health | After submitting, or managing existing jobs |
| [troubleshooting.md](references/troubleshooting.md) | Error→fix tables, GPU memory & batch-size reference | When a job fails or behaves oddly |
| [platform-fixes.md](references/platform-fixes.md) | OpenShift/EKS/GKE detection, SCC, tolerations, NCCL env | On platform errors (read-only FS, scheduling taints) |
| [environment-awareness.md](references/environment-awareness.md) | Notebook GPU, PVC remaining capacity, namespace egress — beyond the MCP | When deciding *where* a workload should run |

## Cross-reference convention

Deep API detail lives in the `kubeflow-trainer` skill and is referenced, not duplicated:

- **TrainJob/TrainingRuntime YAML fields** → `kubeflow-trainer/references/trainjob-runtime-api.md`
- **`TrainerClient` method signatures & dataclass fields** → `kubeflow-trainer/references/python-sdk.md`
- **Per-framework training patterns (DDP/FSDP/DeepSpeed/Megatron/JAX/MLX/XGBoost/TorchTune)** → `kubeflow-trainer/references/builtin-runtimes.md`
- **Migrating v1 `PyTorchJob`/`TFJob`/`MPIJob` → v2** → `kubeflow-trainer/references/migration-from-v1.md`

This skill only restates a field when the *operational* behavior differs from the authoring docs, or
when a parameter has a runtime gotcha the authoring reference doesn't call out.

## Resources

- MCP server being replaced: https://github.com/kubeflow/mcp-server
- Trainer repo (CRDs, controller, runtimes): https://github.com/kubeflow/trainer
- Python SDK: https://github.com/kubeflow/sdk
- Docs: https://trainer.kubeflow.org
