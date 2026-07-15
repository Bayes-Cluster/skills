---
name: kubeflow-trainer
description: Kubernetes-native platform for distributed AI model training and LLM fine-tuning. A single TrainJob CRD (plus TrainingRuntime/ClusterTrainingRuntime templates) replaces the framework-specific PyTorchJob/TFJob/MPIJob CRDs from Kubeflow Training Operator v1. Use when writing or reviewing TrainJob/TrainingRuntime YAML, using the Kubeflow Python SDK (TrainerClient, CustomTrainer, BuiltinTrainer) to launch PyTorch/DeepSpeed/JAX/MLX/XGBoost/Megatron-Core training or TorchTune LLM fine-tuning, running TrainJobs locally (process/Docker/Podman) before deploying to Kubernetes, or migrating PyTorchJob/TFJob/MPIJob workloads to Kubeflow Trainer v2.
version: 1.0.0
author: Terence Liu
license: Apache-2.0
tags: [Kubeflow, Kubernetes, Distributed Training, TrainJob, TrainingRuntime, PyTorch, DeepSpeed, JAX, MLX, XGBoost, Megatron-Core, TorchTune, LLM Fine-Tuning, MLOps, Training Operator Migration]
dependencies: [kubeflow (Python SDK), kubectl, helm]
---

# Kubeflow Trainer

## What it is

Kubeflow Trainer is a Kubernetes-native platform for distributed AI training and LLM
fine-tuning. It replaces the framework-specific CRDs from the legacy **Kubeflow Training
Operator v1** (`PyTorchJob`, `TFJob`, `MPIJob`, ...) with **one unified CRD**: `TrainJob`.

Three CRDs, three personas:

| CRD | Scope | Who writes it |
|---|---|---|
| `TrainJob` | namespace | AI practitioners (usually via the Python SDK, not raw YAML) |
| `TrainingRuntime` | namespace | Platform admins — reusable blueprint for one team/namespace |
| `ClusterTrainingRuntime` | cluster | Platform admins — reusable blueprint shared cluster-wide |

A `TrainJob` references a runtime via `runtimeRef` and supplies the actual training code/image,
node count, and resources. The runtime supplies the pod template, ML framework policy
(`mlPolicy`), and scheduling policy (`podGroupPolicy`).

Under the hood, Kubeflow Trainer builds a [JobSet](https://jobset.sigs.k8s.io/) from the
`TrainJob` + `TrainingRuntime`, and integrates with Kueue, Coscheduling, Volcano, KAI Scheduler
for gang scheduling, and an Apache Arrow/DataFusion distributed data cache for zero-copy
streaming to GPU nodes.

## Install

**Control plane** (Kubernetes >= 1.31, `kubectl` >= 1.31):

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

Full install options (Kustomize, selective runtimes, data cache add-on) are in
[references/trainjob-runtime-api.md](references/trainjob-runtime-api.md).

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

Everything the training function needs (imports included) must live **inside the function
body** — the SDK ships the function's source, not a closure, to every node.

## Core concepts

- **TrainJob** — the job instance. `spec.runtimeRef` (name/kind/apiGroup) + `spec.trainer`
  (image/command/numNodes/resources/env) + optional `spec.initializer` (dataset/model),
  `spec.suspend`, `spec.activeDeadlineSeconds`, `spec.runtimePatches`.
- **TrainingRuntime / ClusterTrainingRuntime** — the blueprint: `spec.mlPolicy` (how the job is
  launched: PlainML/Torch/MPI) + `spec.podGroupPolicy` (gang scheduling) + `spec.template`
  (JobSet `replicatedJobs`, i.e. pod templates for `dataset-initializer`, `model-initializer`,
  `node`, `launcher`, ...).
- **CustomTrainer vs BuiltinTrainer** — `CustomTrainer(func=...)` gives full control: write your
  own Python training function. `BuiltinTrainer(config=TorchTuneConfig(...))` uses a pre-built
  fine-tuning script (currently TorchTune) — you configure hyperparameters, not code.
- **Framework label** — every runtime must carry `trainer.kubeflow.org/framework: <torch|
  deepspeed|jax|mlx|xgboost|torchtune>` so the SDK knows how to talk to it.
- **RuntimePatches** — the supported way for a TrainJob author (or Kueue, or an admission
  webhook) to override pod-level fields (nodeSelector, tolerations, volumes, serviceAccount) of
  the referenced runtime *without* mutating the shared `TrainingRuntime`/`ClusterTrainingRuntime`
  object. Multi-owner: each contributor writes to its own `manager`-keyed entry.

Full API and YAML detail: [references/trainjob-runtime-api.md](references/trainjob-runtime-api.md).

## Common workflows

### Run distributed training (any framework) via CustomTrainer

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

job_id = TrainerClient().train(
    runtime=TrainerClient().get_runtime("torch-distributed"),  # or deepspeed-distributed,
                                                                  # jax-distributed, mlx-distributed,
                                                                  # xgboost-distributed
    trainer=CustomTrainer(
        func=my_train_fn,
        num_nodes=2,
        resources_per_node={"gpu": 4},
        packages_to_install=["transformers>=4.53.0", "boto3"],
        env={"SOME_VAR": "value"},
    ),
)
```

Framework-specific distributed-environment variables (`torchrun` vs `mpirun`-launched
DeepSpeed/MLX vs XGBoost's `DMLC_*` collective vars) and full runnable examples:
[references/builtin-runtimes.md](references/builtin-runtimes.md).

### Fine-tune an LLM with TorchTune BuiltinTrainer (no training loop code)

```python
from kubeflow.trainer import (
    TrainerClient, BuiltinTrainer, TorchTuneConfig, LoraConfig,
    TorchTuneInstructDataset, DataFormat, Initializer, HuggingFaceModelInitializer,
)

client = TrainerClient()
job_id = client.train(
    runtime="torchtune-llama3.2-1b",
    initializer=Initializer(
        model=HuggingFaceModelInitializer(
            storage_uri="hf://meta-llama/Llama-3.2-1B-Instruct",
            access_token="<HF_TOKEN>",
        )
    ),
    trainer=BuiltinTrainer(
        config=TorchTuneConfig(
            dataset_preprocess_config=TorchTuneInstructDataset(source=DataFormat.PARQUET),
            peft_config=LoraConfig(lora_rank=8, lora_alpha=16, lora_dropout=0.1),
            resources_per_node={"gpu": 1},
        )
    ),
)
```

LoRA is supported from Trainer v2.1.0 / SDK v0.2.0. Multi-node TorchTune fine-tuning is **not**
supported. Full config fields: [references/python-sdk.md](references/python-sdk.md).

### Iterate locally before touching a cluster

```python
from kubeflow.trainer import TrainerClient, CustomTrainer, LocalProcessBackendConfig

client = TrainerClient(backend_config=LocalProcessBackendConfig())
job_id = client.train(trainer=CustomTrainer(func=my_train_fn))
```

Swap `LocalProcessBackendConfig()` → `ContainerBackendConfig(container_runtime="docker")` for
multi-node testing in containers, or `KubernetesBackendConfig(namespace=...)` for the real
cluster — `TrainerClient`'s interface (`train`, `list_jobs`, `get_job_logs`, `wait_for_job_status`,
`delete_job`) is identical across all three. Details:
[references/python-sdk.md](references/python-sdk.md).

### Control TrainJob lifecycle

```yaml
spec:
  activeDeadlineSeconds: 3600   # auto-fail after 1h; immutable; timer resets on resume
  suspend: true                  # pause without deleting; flip to false via kubectl patch
```

### Read progress/metrics without scraping logs (Trainer v2.2.0+)

```python
status = TrainerClient().get_job("my-trainjob")
print(status.trainer_status.progress_percentage, status.trainer_status.metrics)
```

Zero-code integration if you use HuggingFace `Trainer` (`transformers>=4.57.0`); manual
`update_trainjob_status()` call for custom loops. Requires the `TrainJobStatus` alpha feature
gate. Detail: [references/trainjob-runtime-api.md](references/trainjob-runtime-api.md).

## TrainerClient API (Kubeflow SDK)

| Method | Purpose |
|---|---|
| `list_runtimes() -> list[Runtime]` | Enumerate available runtimes (namespaced preferred over cluster-scoped on name clash) |
| `get_runtime(name) -> Runtime` | Fetch one runtime by name |
| `get_runtime_packages(runtime)` | Print installed Python packages (+ GPUs) for a runtime |
| `train(runtime=None, initializer=None, trainer=None, options=None) -> str` | Create a TrainJob; returns its generated name. `trainer` is `CustomTrainer` \| `CustomTrainerContainer` \| `BuiltinTrainer` |
| `list_jobs(runtime=None) -> list[TrainJob]` | List TrainJobs, optionally filtered by runtime |
| `get_job(name) -> TrainJob` | Fetch one TrainJob, including `.steps` and `.trainer_status` |
| `get_job_logs(name, step="node-0", follow=False) -> Iterator[str]` | Stream/collect logs from a step (`dataset-initializer`, `model-initializer`, `node-N`) |
| `get_job_events(name) -> list[Event]` | Pod-level events when logs alone don't explain a failure |
| `wait_for_job_status(name, status={COMPLETE}, timeout=600, polling_interval=2, callbacks=None) -> TrainJob` | Block until a target status |
| `delete_job(name)` | Delete the TrainJob and its resources |

Full dataclass field reference for `CustomTrainer`, `BuiltinTrainer`, `TorchTuneConfig`,
`LoraConfig`, `Initializer` and its dataset/model sub-types, and the three backend configs:
[references/python-sdk.md](references/python-sdk.md).

## When to use vs alternatives

**Use Kubeflow Trainer when**: you're on Kubernetes, need one API across PyTorch/DeepSpeed/JAX/
MLX/XGBoost/Megatron-Core, want a Python-first workflow that doesn't require writing YAML by
hand, or need standardized LLM fine-tuning (TorchTune) with LoRA baked in.

**Use raw Kubernetes Jobs/JobSet instead when**: you need something Trainer's runtime/ML-policy
abstraction doesn't yet cover and don't want the extra CRD layer.

**Use Ray Train instead when**: you're not committed to Kubernetes-native APIs and want
Python-native cluster orchestration with built-in hyperparameter tuning.

**Migrating from Training Operator v1** (`PyTorchJob`/`TFJob`/`MPIJob`): see
[references/migration-from-v1.md](references/migration-from-v1.md) — v1 CRDs are collapsed into
one `TrainJob` + `runtimeRef`.

## Common issues

**Training function fails with `NameError` on a training node** — every import used inside a
`CustomTrainer(func=...)` must be declared *inside* the function body; the SDK ships the
function's source code to each node, not a pickled closure with the surrounding module scope.

**Custom `ClusterTrainingRuntime`/`TrainingRuntime` not recognized by the SDK** — it's missing
the `trainer.kubeflow.org/framework` label. Required, not optional.

**Multi-node Megatron-Core / high NCCL-communicator-count jobs hang or crash silently** —
Kubernetes pods default to a 64 MB `/dev/shm`, and NCCL needs ~33 MB per communicator (TP, DP,
etc. each get one). Mount `/dev/shm` as a memory-backed `emptyDir`, either patched into the
runtime cluster-wide or via a per-job `runtimePatches` entry. Detail:
[references/builtin-runtimes.md](references/builtin-runtimes.md).

**`torchrun --nproc_per_node=auto` launches the wrong process count under GPU time-slicing** —
`auto` reads `torch.cuda.device_count()`, not the Kubernetes GPU resource request. If your
cluster time-slices GPUs, use `num_nodes=N, gpu=1` (multi-node) instead of `num_nodes=1, gpu=N`
(multi-GPU) so the pod-level GPU count matches what CUDA reports.

**A shared runtime's webhook-validation label leaks into a runtime you're actively editing** —
`trainer.kubeflow.org/webhook-validation: disabled` opts a runtime out of admission-webhook
validation entirely (used by the Kubeflow community's own pre-validated built-in runtimes so
they can deploy before the webhook is ready). Never carry this label on a runtime you create or
modify yourself.

**Can't set env vars on `node`/`dataset-initializer`/`model-initializer` via `runtimePatches`** —
by design. Use the `Trainer` API (`trainer.env`) or `Initializer` API instead; `runtimePatches`
cannot override `command`/`args`/`image`/`resources`/`env` on those specific containers.

## Advanced topics

- **Full TrainJob/TrainingRuntime YAML API** — MLPolicy types (PlainML/Torch/MPI), JobSet
  template + ancestor labels, RuntimePatches ownership model, activeDeadlineSeconds/suspend,
  progress/metrics reporting, extension framework phases, supported-runtime deprecation policy:
  [references/trainjob-runtime-api.md](references/trainjob-runtime-api.md)
- **Python SDK field-level reference** — `TrainerClient`, `CustomTrainer`/`CustomTrainerContainer`/
  `BuiltinTrainer`, `TorchTuneConfig`/`LoraConfig`, `Initializer` and dataset/model initializer
  variants (HuggingFace/S3/DataCache), the three `*BackendConfig` classes, custom
  `runtime_source` for container backends: [references/python-sdk.md](references/python-sdk.md)
- **Per-framework guides** — PyTorch (DDP/FSDP), PyTorch on ROCm, DeepSpeed (ZeRO via mpirun),
  JAX/JAX-on-TPU, MLX (MPI + Apple Silicon eval), XGBoost (DMLC collective), Megatron-Core
  (tensor parallelism, `/dev/shm` fix, multi-node vs multi-GPU tradeoffs), Flux, TorchTune LoRA
  fine-tuning: [references/builtin-runtimes.md](references/builtin-runtimes.md)
- **Migrating off Training Operator v1** — CRD mapping, PyTorchJob → TrainJob YAML diff,
  PodTemplateOverrides → RuntimePatches: [references/migration-from-v1.md](references/migration-from-v1.md)

## Resources

- Docs: https://trainer.kubeflow.org
- Docs source: https://github.com/kubeflow/trainer/tree/master/docs
- Trainer repo (CRDs, controller, built-in runtime manifests, examples): https://github.com/kubeflow/trainer
- Python SDK repo: https://github.com/kubeflow/sdk
- Examples: https://github.com/kubeflow/trainer/tree/master/examples
- Announcement blog post: https://blog.kubeflow.org/trainer/intro/
