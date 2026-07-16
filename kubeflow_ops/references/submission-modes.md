# Submission Modes

Replaces the MCP `fine_tune`, `run_custom_training`, and `run_container_training` tools. Three ways
to create a `TrainJob`, mapped to their `TrainerClient` SDK equivalents. For full dataclass field
detail see `kubeflow-trainer/references/python-sdk.md`; this file covers the *operational* choice,
the confirm discipline, scheduling parameters, and runtime gotchas.

---

## The three modes at a glance

| Mode | SDK class | Use when | Framework |
|---|---|---|---|
| **Builtin fine-tune** | `BuiltinTrainer(config=TorchTuneConfig(...))` | LoRA/QLoRA fine-tuning of a supported model, no training-loop code | TorchTune (nccl/GPU only) |
| **Custom script** | `CustomTrainer(func=train_fn, ...)` | You want full control; write the Python training function | any (torch/deepspeed/jax/mlx/xgboost) |
| **Pre-built image** | `CustomTrainerContainer(image=..., command=...)` | You ship a container, not source; or your code is too large/proprietary for the function-shipper | any |

All three are passed to the same `TrainerClient.train(...)` call. The interface for monitoring
(`list_jobs`, `get_job_logs`, `wait_for_job_status`) is identical regardless of mode.

---

## 0. Confirm-before-submit (every mode)

This skill cannot enforce the MCP's `confirmed=True` gate, so treat it as a hard behavioral rule:

1. Resolve the full config — the `TrainerClient.train(...)` call with every argument filled in, or
   the equivalent YAML if submitting via `kubectl apply`.
2. **Show the user a readable preview**: runtime, num_nodes, resources_per_node, image, env (with
   secrets masked), volumes, and the model/dataset URIs.
3. **Do not submit in the same turn.** Wait for explicit confirmation, then run `train(...)`.
4. Capture the returned job name and hand it to monitoring.

For `kubectl apply` of a YAML file, the equivalent is: print the YAML, ask, then apply. Never apply
blindly.

---

## 1. Builtin fine-tune (`fine_tune` → `BuiltinTrainer`)

```python
from kubeflow.trainer import (
    TrainerClient, BuiltinTrainer, TorchTuneConfig, LoraConfig,
    TorchTuneInstructDataset, DataFormat, Initializer, HuggingFaceModelInitializer,
)

client = TrainerClient()
job_id = client.train(
    runtime="torchtune-llama3.2-1b",          # must match an installed runtime
    initializer=Initializer(
        model=HuggingFaceModelInitializer(
            storage_uri="hf://meta-llama/Llama-3.2-1B-Instruct",
            access_token="<HF_TOKEN>",         # required for gated models
        ),
    ),
    trainer=BuiltinTrainer(
        config=TorchTuneConfig(
            dataset_preprocess_config=TorchTuneInstructDataset(source=DataFormat.PARQUET),
            peft_config=LoraConfig(lora_rank=8, lora_alpha=16, lora_dropout=0.1),
            resources_per_node={"gpu": 1},
        ),
    ),
)
```

**Operational constraints of this mode:**

- **GPU-only, single-node.** TorchTune runtimes use the NCCL backend; multi-node TorchTune is not
  supported, and some runtimes enforce `num_nodes=1` via webhook. On a CPU-only cluster, use mode 2
  with a gloo script instead.
- **No `env` support.** `BuiltinTrainer`/`fine_tune` does not accept custom environment variables.
  If you need `NCCL_*` tuning or other env, switch to `CustomTrainer` with a LoRA script.
- **`loss` is fixed** to `CEWithChunkedOutputLoss`.
- **Model URI must use the `hf://` prefix** in `storage_uri`.
- **LoRA is supported from Trainer v2.1.0 / SDK v0.2.0.**

---

## 2. Custom script (`run_custom_training` → `CustomTrainer`)

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def train_fn():                       # EVERYTHING the function needs lives inside the body
    import os, torch, torch.distributed as dist
    from datetime import timedelta
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    rank = int(os.environ.get("RANK", "0"))
    # ... build model, DDP-wrap, train ...
    if rank == 0:
        pass  # save checkpoint
    if dist.is_initialized():
        dist.destroy_process_group()

job_id = TrainerClient().train(
    trainer=CustomTrainer(
        func=train_fn,
        num_nodes=2,
        resources_per_node={"cpu": 3, "memory": "16Gi", "gpu": 1},
        packages_to_install=["transformers>=4.53.0", "peft", "trl"],
        env={"NCCL_DEBUG": "INFO", "NCCL_TIMEOUT": "1800"},   # env IS supported here
    ),
)
```

**Operational constraints / gotchas:**

- **All imports inside the function body.** The SDK ships the function's *source*, not a pickled
  closure — module-scope imports are invisible on the worker nodes and cause `NameError`.
- **No `if __name__ == "__main__"` guard**, no top-level indentation errors — the script is wrapped
  into a function body.
- **Use `LOCAL_RANK` for device** (`torch.device(f"cuda:{local_rank}")`), never `RANK`, never
  `cuda:0`, never `device_map="auto"` (breaks DDP).
- **Rank-gate shared I/O** — only rank 0 downloads the model/dataset; `dist.barrier()` before other
  ranks proceed.
- **Check installed packages first** with `TrainerClient().get_runtime_packages(runtime)` — don't
  duplicate packages the runtime already has (slow + can break).
- **Script safety scan.** The SDK best-effort-scans for `os.system`, `subprocess.run`, `eval`,
  `shutil.rmtree`, `__import__`. Flagged patterns warn but don't block. This is **not** a security
  boundary — the script runs with full pod privileges.

---

## 3. Pre-built image (`run_container_training` → `CustomTrainerContainer`)

```python
from kubeflow.trainer import TrainerClient, CustomTrainerContainer

job_id = TrainerClient().train(
    trainer=CustomTrainerContainer(
        image="my-registry/train:latest",
        command=["python", "/workspace/train.py", "--lr", "1e-4"],
        num_nodes=2,
        resources_per_node={"gpu": 4},
        image_pull_secrets=[{"name": "regcred"}],   # if private registry
    ),
)
```

Use when the function source is too large to ship, the code is proprietary, or you need a frozen,
reproducible environment. The image must already contain the framework and entrypoint; the SDK does
not install packages into it.

Via `kubectl` (raw YAML), this is just a `TrainJob` whose `spec.trainer` uses `image`/`command`
instead of `func` — see `kubeflow-trainer/references/trainjob-runtime-api.md`.

---

## Scheduling parameters (all three modes)

Every submission can steer pod placement and attach volumes. These are passed as direct arguments
to the SDK call (or as fields in `runtimePatches` / the pod template in YAML).

| Parameter | Type | Applies to | Notes |
|---|---|---|---|
| `tolerations` | list[dict] | all | Tolerate tainted nodes, e.g. GPU nodes |
| `node_selector` | dict | all | Pin to labeled nodes |
| `affinity` | dict | all | Pod affinity/anti-affinity |
| `volumes` / `volume_mounts` | list[dict] | all | emptyDir, PVC, etc. **Scope differs — see below** |
| `env` | dict | modes 2 & 3 only | **`BuiltinTrainer` does NOT accept `env`** |
| `service_account_name` | str | all | K8s ServiceAccount for the pod |
| `image_pull_secrets` | list[dict] | all | Registry credentials |
| `labels` / `annotations` | dict | all | Extra metadata on the TrainJob |
| `resources_per_node` | dict | all | **Overrides `gpu_per_node` if both given** |

### Volume scope — the easy-to-miss gotcha

- `BuiltinTrainer` (`fine_tune`): volumes apply to **ALL replicated jobs** — `node`,
  `dataset-initializer`, `model-initializer`.
- `CustomTrainer` (`run_custom_training`): volumes apply to the **`node` only**. The SDK
  auto-injects a workspace emptyDir at `/workspace`.
- `CustomTrainerContainer`: add a workspace emptyDir only if your image writes to `/workspace`.

### Common scheduling recipes

GPU toleration (pass as `tolerations`):
```python
[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
```

OpenShift writable emptyDirs (read-only root FS fix — pass as `volumes`):
```python
[
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp",       "mount_path": "/tmp",    "empty_dir": {}},
]
```
More platform recipes in [platform-fixes.md](platform-fixes.md).

---

## Choosing a runtime

Before submitting, confirm the runtime exists and carries the right framework label:

```bash
kubectl get clustertrainingruntime -o custom-columns='NAME:.metadata.name,FW:.metadata.labels.trainer\.kubeflow\.org/framework'
```

The framework label (`torch|deepspeed|jax|mlx|xgboost|torchtune`) tells the SDK how to launch the
job. A custom runtime missing this label will be invisible to `TrainerClient` — the SDK won't list
it. (This is a common failure when authoring your own runtime; see `kubeflow-trainer` skill →
builtin-runtimes.)

## Parameter rules that aren't inferable

- `resources_per_node` overrides `gpu_per_node` when both are provided — don't set both.
- `packages_to_install` requires a writable `/.local`; on read-only root FS, add the `dot-local`
  emptyDir (see recipe above) or pip will fail.
- `pip_index_urls` is ignored unless `packages_to_install` is non-empty.
- `command` and `args` in `CustomTrainerContainer` are **separate** parameters.
- URI prefix: `fine_tune`/model initializers require `hf://`; bare model IDs are rejected.
