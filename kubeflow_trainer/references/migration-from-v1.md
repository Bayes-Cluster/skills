# Migrating from Kubeflow Training Operator v1

Kubeflow Trainer ("v2") is a rewrite of the Kubeflow Training Operator. It replaces the
framework-specific CRDs (`PyTorchJob`, `TFJob`, `MPIJob`, `XGBoostJob`, ...) with a **single**
`TrainJob` CRD plus reusable `TrainingRuntime`/`ClusterTrainingRuntime` blueprints.

## What's new in v2

- **Unified CRDs** — `TrainJob`, `TrainingRuntime`, `ClusterTrainingRuntime` (`apiVersion:
  trainer.kubeflow.org/v1alpha1`) replace the per-framework v1 CRDs. All functionality lives in
  one `TrainJob` CRD instead of being split across framework-specific ones.
- **Kubeflow Python SDK** — `TrainerClient`/`CustomTrainer`/`BuiltinTrainer` let AI practitioners
  build a `TrainJob` from Python without hand-writing YAML or shelling out to `kubectl`.
- **Dataset/model initializers** — dedicated init containers (`dataset-initializer`,
  `model-initializer`) offload I/O (downloading from HuggingFace/S3) to CPU workloads instead of
  burning GPU time on it, and standardize asset staging across nodes.
- **Enhanced MPI support** — MPI-Operator v2 features with SSH-based optimizations for MPI
  performance (used by DeepSpeed, MLX).

## CRD mapping

| Training Operator v1 | Kubeflow Trainer v2 |
|---|---|
| `PyTorchJob` | `TrainJob` + `runtimeRef` → `torch-distributed` (or custom) `ClusterTrainingRuntime` |
| `TFJob` | `TrainJob` + a custom runtime (TF isn't a built-in community runtime in v2) |
| `MPIJob` | `TrainJob` + `runtimeRef` → an MPI-policy runtime (e.g. `deepspeed-distributed`, `mlx-distributed`, or a custom one) |
| `XGBoostJob` | `TrainJob` + `runtimeRef` → `xgboost-distributed` |
| `PodTemplateOverrides` (a TrainJob v1alpha1-early field) | `RuntimePatches` (`manager`-keyed, hierarchical) — see [trainjob-runtime-api.md](trainjob-runtime-api.md#runtimepatches) |

## Example: PyTorchJob → TrainJob

**Before — `PyTorchJob` (v1)**:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-simple
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
              command: ["python3", "/opt/pytorch-mnist/mnist.py", "--epochs=1"]
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
              command: ["python3", "/opt/pytorch-mnist/mnist.py", "--epochs=1"]
```

**After — `TrainJob` (v2)**, using the default `torch-distributed` runtime:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: pytorch-simple
spec:
  runtimeRef:
    name: torch-distributed
  trainer:
    numNodes: 2
    image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
    command: ["python3", "/opt/pytorch-mnist/mnist.py", "--epochs=1"]
```

Note the collapse of `Master`/`Worker` replica specs into a single `numNodes` count — v2's
`torch-distributed` runtime uses `torchrun` for rendezvous, so there's no separate
master/worker role to declare.

For a Python-first migration (recommended over hand-writing YAML), wrap the same training code
in a `CustomTrainer` and call `TrainerClient().train(...)` — see
[python-sdk.md](python-sdk.md) and [builtin-runtimes.md](builtin-runtimes.md).

## Migration checklist

1. **Install Kubeflow Trainer v2 control plane** alongside or instead of Training Operator v1 —
   see [trainjob-runtime-api.md](trainjob-runtime-api.md#installation). They are separate
   controllers/CRD groups (`kubeflow.org/v1` vs `trainer.kubeflow.org/v1alpha1`), so this can be
   done incrementally.
2. **Pick or author a runtime.** Use a built-in `ClusterTrainingRuntime` (`torch-distributed`,
   `deepspeed-distributed`, `jax-distributed`, `mlx-distributed`, `xgboost-distributed`) if it
   fits, or author a custom `TrainingRuntime`/`ClusterTrainingRuntime` matching your v1 job's
   image/resources/MPI-vs-torchrun launch style. Every custom runtime needs the
   `trainer.kubeflow.org/framework` label.
3. **Rewrite the job spec**: replica-spec-per-role (`Master`/`Worker`/`Chief`/`PS`) collapses
   into `trainer.numNodes` + one pod template per `replicatedJobs` entry in the runtime (usually
   just `node`).
4. **Move any `PodTemplateOverrides` to `RuntimePatches`** — different structure (manager-keyed
   vs flat `targetJobs` list). See the migration example in
   [trainjob-runtime-api.md](trainjob-runtime-api.md#migrating-from-podtemplateoverrides).
5. **Prefer the SDK over hand-written YAML** for new jobs — `TrainerClient().train(...)`
   generates the `TrainJob` for you and is what the rest of this skill's workflows assume.
6. **TFJob users**: v2 does not ship a community-maintained TF-specific `ClusterTrainingRuntime`.
   Author a custom `TrainingRuntime` with a `PlainML` or `Torch`-equivalent `mlPolicy` (TF's own
   `tf.distribute` strategies manage their own rendezvous, so `PlainML` — plain Kubernetes Job
   parallelism — is usually the right fit) pointing at your existing TF training image.

## Further reading

- v1 (Training Operator) legacy docs are archived under "Legacy Training Operator v1" at
  https://trainer.kubeflow.org — only consult these for jobs you haven't migrated yet.
- Announcement / rationale for the v2 rewrite: https://blog.kubeflow.org/trainer/intro/
- Runtime/MLPolicy/JobTemplate detail: [trainjob-runtime-api.md](trainjob-runtime-api.md)
- SDK detail: [python-sdk.md](python-sdk.md)
