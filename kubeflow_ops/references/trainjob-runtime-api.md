# TrainJob / TrainingRuntime API Reference

`apiVersion: trainer.kubeflow.org/v1alpha1`. Three CRDs: `TrainJob`, `TrainingRuntime`
(namespace-scoped), `ClusterTrainingRuntime` (cluster-scoped).

## Installation

**Helm** (preferred):

```bash
export VERSION=v2.1.0
helm install kubeflow-trainer oci://ghcr.io/kubeflow/charts/kubeflow-trainer \
    --namespace kubeflow-system \
    --create-namespace \
    --version ${VERSION#v}
```

CRDs (`TrainJob`, `TrainingRuntime`, `ClusterTrainingRuntime`) install by default with the
chart. If you manage CRDs out-of-band, add `--set crds.enabled=false`.

Enable the community's default `ClusterTrainingRuntime`s in the same install (a post-install
Helm hook applies them once CRDs + controller are ready):

```bash
--set runtimes.defaultEnabled=true
```

Or enable specific ones:

```bash
--set runtimes.torchDistributed.enabled=true \
--set runtimes.deepspeedDistributed.enabled=true
```

`helm upgrade` with the same flags reconciles runtimes on every upgrade (newly enabled ones
applied, disabled ones removed). Disabling *all* runtimes removes the installer itself — delete
any remaining runtimes manually or via `helm uninstall` in that case.

**Kustomize** (alternative):

```bash
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=master"
```

**Verify**:

```bash
kubectl get pods -n kubeflow-system
# jobset-controller-manager-...   2/2 Running
# kubeflow-trainer-controller-manager-...   1/1 Running
```

Prerequisites: Kubernetes >= 1.31, `kubectl` >= 1.31. No cluster? `kind create cluster` or
`minikube start`.

## TrainingRuntime / ClusterTrainingRuntime

Both share the same `spec` shape (`mlPolicy`, `podGroupPolicy`, `template`); the only
difference is scope.

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime          # or TrainingRuntime (namespaced)
metadata:
  name: torch-distributed
  labels:
    trainer.kubeflow.org/framework: torch   # REQUIRED — the SDK uses this to recognize the runtime
spec:
  mlPolicy:
    numNodes: 1
    torch:
      numProcPerNode: auto
  template:
    spec:
      replicatedJobs:
        - name: node
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: trainer
            spec:
              template:
                spec:
                  containers:
                    - name: node
                      image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
```

A `TrainJob` references it:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: example-train-job
spec:
  runtimeRef:
    apiGroup: trainer.kubeflow.org
    name: torch-distributed
    kind: ClusterTrainingRuntime   # or TrainingRuntime
```

`TrainingRuntime` (namespaced) additionally requires the `TrainJob`'s namespace to match the
runtime's namespace — it's only visible within that namespace.

### Framework label (required)

Every deployed runtime must carry `trainer.kubeflow.org/framework: <torch|deepspeed|jax|mlx|
xgboost|torchtune>`. The SDK uses this label to decide how to configure `BuiltinTrainer`s and to
recognize the runtime at all.

### Skipping webhook validation (built-in runtimes only)

Kubeflow Trainer runs a validation webhook against every `TrainingRuntime`/
`ClusterTrainingRuntime` create/update. Runtimes shipped and pre-validated by the Kubeflow
community carry:

```yaml
trainer.kubeflow.org/webhook-validation: disabled
```

This lets them apply before the webhook server is ready (e.g. inside a single-step
`helm install` that applies CRDs + controller + runtimes together). It opts the runtime out of
validation **permanently**, even once the webhook is up — only use it on runtimes you trust and
don't intend to hand-edit. Runtimes you author yourself should omit this label so they stay
validated.

### Supported built-in ClusterTrainingRuntimes

For `CustomTrainer`:

| Runtime name | Framework |
|---|---|
| `torch-distributed` | PyTorch |
| `deepspeed-distributed` | DeepSpeed |
| `jax-distributed` | JAX |
| `mlx-distributed` | MLX |
| `xgboost-distributed` | XGBoost |

For TorchTune `BuiltinTrainer`:

| Runtime name | Pre-trained LLM |
|---|---|
| `torchtune-llama3.2-1b` | Llama 3.2 (1B) |
| `torchtune-llama3.2-3b` | Llama 3.2 (3B) |
| `torchtune-qwen2.5-1.5b` | Qwen 2.5 (1.5B) |

Full manifests: https://github.com/kubeflow/trainer/tree/master/manifests/base/runtimes

### Runtime deprecation policy

A deprecated `ClusterTrainingRuntime` stays available for **two minor releases** after
deprecation, marked with `trainer.kubeflow.org/support: "deprecated"`. Deprecation is documented
as a breaking change in release notes; the validation webhook and the SDK both surface a warning
when a deprecated runtime is deployed, referenced by a new `TrainJob`, or listed.

## MLPolicy

`spec.mlPolicy` picks how the training job is launched/orchestrated. One `MLPolicySource` per
runtime.

### PlainML

No distributed framework. Plain Kubernetes `Job` semantics — `numNodes` sets parallelism, TrainJob
env vars are added to each pod.

```yaml
mlPolicy:
  numNodes: 10
```

### Torch

Launches via `torchrun`. `numProcPerNode` sets processes (e.g. GPUs) per node.

```yaml
mlPolicy:
  numNodes: 3
  torch:
    numProcPerNode: gpu   # or: auto, cpu, or an explicit integer
```

By default `PET_*` topology env vars (torchrun's env-var protocol) are injected only into the
main trainer container. To also inject them into init containers or other auxiliary containers
(e.g. a distributed preflight check), opt in via `envInjection`:

```yaml
mlPolicy:
  numNodes: 2
  torch:
    envInjection:
      targets:
        - jobName: node                     # must match a replicatedJob name in the template
          containerNames: [preflight-check]  # must exist in that job's pod spec (regular or init)
```

`envInjection` targets always get the full `PET_*` set (`PET_NNODES`, `PET_NPROC_PER_NODE`,
`PET_NODE_RANK`, `PET_MASTER_ADDR`, `PET_MASTER_PORT`), regardless of trainer type. Exception:
under TorchTune, the main trainer container gets the rendezvous endpoint via a CLI argument
instead of `PET_MASTER_ADDR`/`PET_MASTER_PORT`.

### MPI

Launches via `mpirun` — used by frameworks like DeepSpeed and MLX that run on OpenMPI.

```yaml
mlPolicy:
  numNodes: 2
  mpi:
    numProcPerNode: 4
    mpiImplementation: OpenMPI
    sshAuthMountPath: /home/mpiuser/.ssh
    runLauncherAsNode: true
```

## Job Template (JobSet)

`spec.template` defines the [JobSet](https://jobset.sigs.k8s.io/docs/overview/) the controller
builds. `replicatedJobs` map to distinct pod roles (`dataset-initializer`, `model-initializer`,
`node`, `launcher`, ...); each gets a standard Kubernetes `Job` spec (image, command, resources).

### Ancestor label requirement

Each `replicatedJobs[].template` needs a `trainer.kubeflow.org/trainjob-ancestor-step` label so
the controller knows which part of the parent `TrainJob` spec to inject:

| Label value | Injected from |
|---|---|
| `trainer` | `TrainJob.spec.trainer` |
| `dataset-initializer` | `TrainJob.spec.initializer.dataset` |
| `model-initializer` | `TrainJob.spec.initializer.model` |

Full example with a 4-stage pipeline (`dataset-initializer` → `model-initializer` → `launcher` →
`node`, using `dependsOn` for ordering):

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: example-runtime
  labels:
    trainer.kubeflow.org/framework: mlx
spec:
  template:
    spec:
      replicatedJobs:
        - name: dataset-initializer
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: dataset-initializer
            spec:
              template:
                spec:
                  containers:
                    - name: dataset-initializer
                      image: ghcr.io/kubeflow/trainer/dataset-initializer
        - name: model-initializer
          dependsOn:
            - name: dataset-initializer
              status: Complete
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: model-initializer
            spec:
              template:
                spec:
                  containers:
                    - name: model-initializer
                      image: ghcr.io/kubeflow/trainer/model-initializer
        - name: launcher
          dependsOn:
            - name: model-initializer
              status: Complete
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: trainer
            spec:
              template:
                spec:
                  containers:
                    - name: node
                      image: ghcr.io/kubeflow/trainer/mlx-runtime
                      securityContext:
                        runAsUser: 1000
        - name: node
          template:
            spec:
              template:
                spec:
                  containers:
                    - name: node
                      image: ghcr.io/kubeflow/trainer/mlx-runtime
                      securityContext:
                        runAsUser: 1000
                      command: [/usr/sbin/sshd]
                      args: [-De, -f, /home/mpiuser/.sshd_config]
                      readinessProbe:
                        tcpSocket:
                          port: 2222
                        initialDelaySeconds: 5
```

## RuntimePatches

Replaces the old `PodTemplateOverrides` API. Lets multiple controllers (the user via the SDK,
Kueue, admission webhooks) each contribute a config patch to a `TrainJob`'s runtime spec without
mutating the shared `TrainingRuntime`/`ClusterTrainingRuntime` object, and without stepping on
each other.

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: pytorch-distributed
spec:
  runtimeRef:
    name: pytorch-distributed-gpu
  trainer:
    image: docker.io/custom-training
  runtimePatches:
    - manager: trainer.kubeflow.org/kubeflow-sdk
      trainingRuntimeSpec:
        template:
          spec:
            replicatedJobs:
              - name: node
                template:
                  spec:
                    template:
                      spec:
                        nodeSelector:
                          accelerator: nvidia-tesla-v100
```

### Ownership model

- `manager` (required, immutable, ≤253 chars, typically domain-prefixed e.g.
  `kueue.x-k8s.io/manager`) identifies who owns an entry. The SDK sets
  `manager: trainer.kubeflow.org/kubeflow-sdk` automatically when you pass
  [`options`](https://sdk.kubeflow.org/en/latest/train/options.html) to `train()`. Each manager
  maps to exactly one list entry; updates replace that entry in full — controllers can't
  accidentally clobber each other's patches.
- `time` is auto-stamped by the admission webhook on every create/update. Observability only,
  not a list key — don't set it yourself.
- Some fields are only mutable while the `TrainJob` is `suspend`ed, to prevent drift between
  running pods and the declared spec.

### Patch structure (TrainJob → JobSet → Job → Pod → container)

```yaml
runtimePatches:
  - manager: trainer.kubeflow.org/kubeflow-sdk
    trainingRuntimeSpec:
      template:
        metadata:                # JobSet-level labels/annotations
          labels: {team: ml-platform}
        spec:
          replicatedJobs:
            - name: node
              template:
                metadata:         # Job-level labels/annotations
                  labels: {job-label: value}
                spec:
                  template:
                    metadata:      # Pod-level labels/annotations
                      annotations: {monitoring: enabled}
                    spec:          # Pod spec patches
                      serviceAccountName: custom-sa
                      nodeSelector: {accelerator: nvidia-tesla-v100}
                      containers:
                        - name: trainer   # must match a container name in the runtime
                          volumeMounts:
                            - name: workspace
                              mountPath: /workspace
                      volumes:
                        - name: workspace
                          persistentVolumeClaim: {claimName: user-pvc}
```

Common uses: custom `serviceAccountName` + `nodeSelector`, mounting a PVC, adding
`tolerations` for tainted GPU/priority nodes.

### Restrictions

- Cannot set env vars on `node`, `dataset-initializer`, or `model-initializer` containers — use
  the `Trainer` API (`trainer.env`) / `Initializer` API instead.
- Cannot override `command`, `args`, `image`, or `resources` for the `trainer` container in the
  `node` replicatedJob — use the `Trainer` API fields on the `TrainJob` spec.

### Migrating from PodTemplateOverrides

Old (`targetJobs` flat list):

```yaml
spec:
  podTemplateOverrides:
    - targetJobs: [{name: node}]
      spec:
        serviceAccountName: ml-training-sa
        nodeSelector: {accelerator: nvidia-tesla-v100}
```

New (`manager`-keyed, hierarchical):

```yaml
spec:
  runtimePatches:
    - manager: trainer.kubeflow.org/kubeflow-sdk
      trainingRuntimeSpec:
        template:
          spec:
            replicatedJobs:
              - name: node
                template:
                  spec:
                    template:
                      spec:
                        serviceAccountName: ml-training-sa
                        nodeSelector: {accelerator: nvidia-tesla-v100}
```

## TrainJob Lifecycle

### activeDeadlineSeconds

Max seconds a `TrainJob` may run before automatic termination (mirrors Kubernetes `Job`
`activeDeadlineSeconds` semantics). Immutable after creation, minimum `1`. Timer starts at
creation and **resets on resume** if the job was suspended.

```yaml
spec:
  activeDeadlineSeconds: 3600
```

On deadline exceeded: all pods terminated, the JobSet deleted, `TrainJob` status → `Failed`,
reason `DeadlineExceeded`.

```bash
kubectl get trainjob my-trainjob -o jsonpath='{.status.conditions[?(@.status=="True")]}'
# {"type":"Failed","status":"True","reason":"DeadlineExceeded", ...}
```

### suspend / resume

```yaml
spec:
  suspend: true   # pods terminated, TrainJob + config preserved
```

```bash
kubectl patch trainjob my-trainjob --type=merge -p '{"spec":{"suspend":false}}'
```

## TrainJob Progress and Metrics (v2.2.0+)

Push structured progress (completion %, ETA, custom metrics) into
`TrainJob.status.trainerStatus` in real time — no log scraping. Requires the `TrainJobStatus`
alpha feature gate (`--feature-gates=TrainJobStatus=true` on the controller, or Helm
`manager.config.featureGates.TrainJobStatus=true`) and, for SDK-based reporting, Kubeflow SDK
`>= 0.5.0`.

**Read it**:

```bash
kubectl get trainjob <name> -o jsonpath='{.status.trainerStatus}'
```

```python
status = TrainerClient().get_job("my-trainjob")
print(status.trainer_status.progress_percentage, status.trainer_status.estimated_remaining_seconds,
      status.trainer_status.metrics)
```

**Report it — three options**:

1. **HuggingFace Transformers, zero code change** (`transformers >= 4.57.0`): a
   `KubeflowTrainerCallback` auto-registers when it detects it's running inside a TrainJob, and
   reports loss/learning_rate/epoch/completion at each logging step, plus anything returned by
   `Trainer(compute_metrics=...)`.

2. **Kubeflow SDK, custom training loop**:

   ```python
   from kubeflow.trainer import update_trainjob_status

   update_trainjob_status(progress_percent=0)
   for epoch in range(total_epochs):
       ...
       update_trainjob_status(
           progress_percent=int((epoch + 1) / total_epochs * 100),
           estimated_remaining_seconds=eta_seconds,
           metrics={"loss": train_loss, "accuracy": train_acc},
       )
   update_trainjob_status(progress_percent=100)
   ```

   The SDK throttles to at most one update per 5 seconds — safe to call every step.

3. **Raw HTTP, any language**: the controller injects `KUBEFLOW_TRAINER_SERVER_URL`,
   `KUBEFLOW_TRAINER_SERVER_CA_CERT` (path), `KUBEFLOW_TRAINER_SERVER_TOKEN` (path, JWT that
   rotates — re-read the file, don't cache) as env vars. POST a `{"trainerStatus": {...}}`
   payload with `Authorization: Bearer <token>`. Client-side rate-limit to ≥5s between requests,
   and never let reporting errors interrupt the training loop.

Planned follow-ons: automatic checkpointing triggered by ETA; `OptimizationJob` (Katib)
integration.

## Distributed Data Cache (Apache Arrow + DataFusion)

Optional add-on for zero-copy data streaming into distributed TrainJobs, backed by a distributed
Arrow cache cluster (accessed via Arrow Flight) that pre-processes/partitions object-store data
(Iceberg-on-S3) once and serves it to many training nodes/jobs.

```bash
export VERSION=v2.1.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/data-cache?ref=${VERSION}"
# or via Helm:
helm install kubeflow-trainer oci://ghcr.io/kubeflow/charts/kubeflow-trainer \
    --set dataCache.enabled=true --namespace kubeflow-system --create-namespace --version ${VERSION#v}
```

Requires the LeaderWorkerSet controller (installed automatically unless
`dataCache.lws.install=false`) and RBAC applied per-namespace where TrainJobs will run:

```bash
kubectl apply --server-side -n <NAMESPACE> -k "https://github.com/kubeflow/trainer.git/manifests/overlays/data-cache/namespace-rbac"
```

Uses the `torch-distributed-with-cache` `ClusterTrainingRuntime`. Data must be in Iceberg table
format on S3.

## Job Scheduling / Gang Scheduling

`spec.podGroupPolicy` on a runtime enables gang scheduling — a group of TrainJob pods only
start once *all* required resources are available, which matters when GPUs are scarce/expensive.
Supported `PodGroupPolicySource` integrations: Coscheduling (Kubernetes scheduler plugin),
Volcano, KAI Scheduler. Kueue integrates separately for topology-aware queueing/dispatch
(https://kueue.sigs.k8s.io/docs/tasks/run/trainjobs/).

## Extension Framework (for platform admins writing controller plugins)

Internal mechanism with four execution phases, each with internal APIs (fixed) and extension
points (pluggable):

1. **Startup** — runs once at controller-manager init. Extension point: `WatchExtension`
   (register custom reconciler builders).
2. **PreExecution** — runs on TrainJob create/update. Extension point: `CustomValidation`.
3. **Build** — builds/deploys the JobSet and related resources. Extension points:
   `EnforcePodGroupPolicy`, `EnforceMLPolicy`, `ComponentBuilder`.
4. **PostExecution** — runs after the job executes; checks status and applies terminal
   conditions.

Use cases: custom TrainJob field validation, dynamic per-framework resource deployment. This is
a controller-manager-level extension surface, not something end users touch via YAML/SDK.
