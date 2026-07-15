# Kubeflow Python SDK Reference

```bash
pip install -U kubeflow                              # latest stable
pip install git+https://github.com/kubeflow/sdk.git@main   # latest from source
```

Source: https://github.com/kubeflow/sdk (`python/kubeflow/trainer/`)

## TrainerClient

```python
from kubeflow.trainer import TrainerClient

client = TrainerClient(backend_config=None)  # defaults to KubernetesBackendConfig()
```

`backend_config` is one of `KubernetesBackendConfig`, `LocalProcessBackendConfig`,
`ContainerBackendConfig` — see [Backends](#backends) below. Same `TrainerClient` interface
regardless of backend.

### Methods

```python
list_runtimes() -> list[Runtime]
```
Available runtimes. Namespace-scoped `TrainingRuntime` preferred over cluster-scoped
`ClusterTrainingRuntime` on a name clash. Empty list if none found.

```python
get_runtime(name: str) -> Runtime
```
Fetch one runtime; namespace-scoped wins on name clash.

```python
get_runtime_packages(runtime: Runtime) -> None
```
Prints installed Python packages (and available GPUs, if any) for the runtime's single training
node.

```python
train(
    runtime: str | Runtime | None = None,
    initializer: Initializer | None = None,
    trainer: CustomTrainer | CustomTrainerContainer | BuiltinTrainer | None = None,
    options: list | None = None,
) -> str
```
Create a `TrainJob`, returns its generated name.
- `runtime`: name string or `Runtime` object. Defaults to `torch-distributed` if omitted.
- `initializer`: dataset/model download config (see [Initializer](#initializer)).
- `trainer`: exactly one of `CustomTrainer` (Python function), `CustomTrainerContainer`
  (pre-built image), or `BuiltinTrainer` (config-only, e.g. TorchTune). Omit to use the
  runtime's defaults.
- `options`: list of config options from `kubeflow.trainer.options` (e.g. what sets
  `runtimePatches[].manager` to `trainer.kubeflow.org/kubeflow-sdk`).

```python
list_jobs(runtime: Runtime | None = None) -> list[TrainJob]
get_job(name: str) -> TrainJob
get_job_logs(name: str, step: str = "node-0", follow: bool | None = False) -> Iterator[str]
get_job_events(name: str) -> list[Event]
wait_for_job_status(
    name: str,
    status: set[str] = {"Complete"},
    timeout: int = 600,
    polling_interval: int = 2,
    callbacks: list[Callable[[TrainJob], None]] | None = None,
) -> TrainJob
delete_job(name: str) -> None
```

`step` for `get_job_logs` is a replicatedJob name + index, e.g. `dataset-initializer`,
`model-initializer`, `node-0`, `node-1`. `status` for `wait_for_job_status` is a subset of
`{Created, Running, Complete, Failed}` (use `kubeflow.trainer.constants.constants` for the
string constants, e.g. `constants.TRAINJOB_COMPLETE`). `callbacks` fire on every poll with the
current `TrainJob`.

**Streaming logs**:

```python
for logline in TrainerClient().get_job_logs(name="s8d44aa4fb6d", follow=True):
    print(logline)
```

## Trainer configuration

### CustomTrainer

Full control — you write the training function.

```python
@dataclass
class CustomTrainer:
    func: Callable
    func_args: dict | None = None
    image: str | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = [DEFAULT_PIP_INDEX_URLS...]
    num_nodes: int | None = None
    resources_per_node: dict | None = None   # {"gpu": 4, "cpu": 5, "memory": "10G"}
                                               # fractional/MIG GPU: {"mig-1g.5gb": 1}
    env: dict[str, str] | None = None
```

**Every import used inside `func` must be declared inside `func`'s body.** The SDK ships the
function's source, not a closure — module-level imports outside the function are invisible to
the training nodes.

### CustomTrainerContainer

Same shape as `CustomTrainer` but for a pre-built image instead of a Python function:

```python
@dataclass
class CustomTrainerContainer:
    image: str
    num_nodes: int | None = None
    resources_per_node: dict | None = None
    env: dict[str, str] | None = None
```

### BuiltinTrainer + TorchTuneConfig

Configuration-driven — no training-loop code. Currently supports TorchTune LLM fine-tuning.

```python
@dataclass
class BuiltinTrainer:
    config: TorchTuneConfig

@dataclass
class TorchTuneConfig:
    dtype: DataType | None = None                       # DataType.BF16 | DataType.FP32
    batch_size: int | None = None
    epochs: int | None = None
    loss: Loss | None = None                             # Loss.CEWithChunkedOutputLoss
    num_nodes: int | None = None                         # multi-node NOT supported for TorchTune
    peft_config: LoraConfig | None = None
    dataset_preprocess_config: TorchTuneInstructDataset | None = None
    resources_per_node: dict | None = None
```

```python
@dataclass
class LoraConfig:
    apply_lora_to_mlp: bool | None = None
    apply_lora_to_output: bool | None = None
    lora_attn_modules: list[str] = ["q_proj", "v_proj", "output_proj"]   # also: "k_proj"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    quantize_base: bool | None = None    # QLoRA
    use_dora: bool | None = None         # DoRA
```
LoRA/QLoRA/DoRA supported from Trainer v2.1.0 / SDK v0.2.0.

```python
@dataclass
class TorchTuneInstructDataset:
    source: DataFormat | None = None       # JSON | CSV | PARQUET | ARROW | TEXT | XML
    split: str | None = None               # e.g. "train[:95%]", default "train"
    train_on_input: bool | None = None     # default False
    new_system_prompt: str | None = None
    column_map: dict[str, str] | None = None   # e.g. {"input": "incorrect", "output": "correct"}
```

Full example:

```python
from kubeflow.trainer import (
    TrainerClient, BuiltinTrainer, TorchTuneConfig, LoraConfig,
    TorchTuneInstructDataset, DataFormat, DataType, Loss,
    Initializer, HuggingFaceModelInitializer,
)

client = TrainerClient()
job_name = client.train(
    runtime="torchtune-llama3.2-1b",
    initializer=Initializer(
        model=HuggingFaceModelInitializer(
            storage_uri="hf://meta-llama/Llama-3.2-1B-Instruct",
            access_token="<YOUR_HF_TOKEN>",
        )
    ),
    trainer=BuiltinTrainer(
        config=TorchTuneConfig(
            dtype=DataType.BF16,
            batch_size=10,
            epochs=10,
            loss=Loss.CEWithChunkedOutputLoss,
            num_nodes=1,
            peft_config=LoraConfig(lora_rank=8, lora_alpha=16, lora_dropout=0.1),
            dataset_preprocess_config=TorchTuneInstructDataset(
                source=DataFormat.PARQUET,
                split="train[:95%]",
                train_on_input=True,
                new_system_prompt="You are an AI assistant. ",
                column_map={"input": "incorrect", "output": "correct"},
            ),
            resources_per_node={"gpu": 1},
        )
    ),
)
```

After completion, the fine-tuned model lands in `/workspace/output` on a PVC the TrainJob
creates and manages automatically — mount the same PVC under `/workspace` in another pod to
retrieve it. Watch dataset/model download progress via
`client.get_job_logs(job_name, step=constants.DATASET_INITIALIZER)` /
`step=constants.MODEL_INITIALIZER`.

## Initializer

Downloads dataset/model assets on dedicated init pods before training starts (offloads I/O to
CPU workloads instead of burning GPU time).

```python
@dataclass
class Initializer:
    dataset: HuggingFaceDatasetInitializer | S3DatasetInitializer | DataCacheInitializer | None = None
    model: HuggingFaceModelInitializer | S3ModelInitializer | None = None
```

```python
@dataclass
class HuggingFaceDatasetInitializer(BaseInitializer):   # storage_uri must start with "hf://"
    storage_uri: str          # "hf://<user>/<repo>" or "hf://<user>/<repo>/path/to/file_or_dir"
    ignore_patterns: list[str] | None = None
    access_token: str | None = None
```
`storage_uri` needs the exact path to data files: `hf://tatsu-lab/alpaca/data` (a directory — uses
all files under it) or `hf://tatsu-lab/alpaca/data/xxx.parquet` (a single file).

```python
@dataclass
class HuggingFaceModelInitializer(BaseInitializer):     # storage_uri must start with "hf://"
    storage_uri: str          # "hf://<user>/<repo>"
    ignore_patterns: list[str] | None = None   # defaults to a sensible ignore list
    access_token: str | None = None
```

```python
@dataclass
class S3DatasetInitializer(BaseInitializer):             # storage_uri must start with "s3://"
    storage_uri: str
    ignore_patterns: list[str] | None = None
    endpoint: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region: str | None = None
    role_arn: str | None = None

@dataclass
class S3ModelInitializer(BaseInitializer):               # same fields as S3DatasetInitializer
    ...
```

```python
@dataclass
class DataCacheInitializer(BaseInitializer):
    storage_uri: str          # "cache://<SCHEMA_NAME>/<TABLE_NAME>"
    metadata_loc: str         # path to the Iceberg table's metadata.json
    num_data_nodes: int        # must be > 1
    head_cpu: str | None = None
    head_mem: str | None = None
    worker_cpu: str | None = None
    worker_mem: str | None = None
    iam_role: str | None = None
```
Requires the distributed data cache control plane installed (see
[trainjob-runtime-api.md](trainjob-runtime-api.md#distributed-data-cache-apache-arrow--datafusion)).

## Backends

All three backends implement the identical `TrainerClient` surface — only construction changes,
so you can develop locally then flip one line to deploy to a real cluster.

```python
from kubeflow.trainer import (
    TrainerClient, CustomTrainer,
    LocalProcessBackendConfig, ContainerBackendConfig, KubernetesBackendConfig,
)

# Fastest: native Python process / venv. No multi-node support.
client = TrainerClient(backend_config=LocalProcessBackendConfig())

# Full container isolation with Docker or Podman. Multi-node supported.
client = TrainerClient(backend_config=ContainerBackendConfig(container_runtime="docker"))
#   container_runtime="podman"  or  None (auto-detect: tries Docker, then Podman)

# Production Kubernetes cluster.
client = TrainerClient(backend_config=KubernetesBackendConfig(namespace="kubeflow"))

trainer = CustomTrainer(func=train_model, num_nodes=4)
job_name = client.train(trainer=trainer)   # identical call across all three backends
```

| | Local Process | Docker | Podman |
|---|---|---|---|
| Setup | none | Docker Desktop/Engine | Podman install |
| Isolation | venv | full container | full container |
| Multi-node | no | yes | yes |
| Root required | no | Docker group/root | rootless supported |
| Startup | fastest | medium | medium |
| Best for | quick prototyping | general use, wide ecosystem | Linux servers, security |

Job management (`list_jobs`, `get_job_logs`, `wait_for_job_status`, `delete_job`,
`list_runtimes`) works identically across all backends. `delete_job` also tears down any
networks/containers/processes the job created.

### Custom runtime sources for Container backends

By default `ContainerBackendConfig` resolves runtimes from (in order): built-in default images
(fallback), then `github://kubeflow/trainer` (official runtimes, 24h cache). Override with
`runtime_source`:

```python
from kubeflow.trainer import ContainerBackendConfig, TrainingRuntimeSource

backend_config = ContainerBackendConfig(
    container_runtime="docker",
    runtime_source=TrainingRuntimeSource(sources=[
        "github://kubeflow/trainer",
        "github://myorg/myrepo/path/to/runtimes",
        "https://example.com/custom-runtime.yaml",
        "file:///absolute/path/to/runtime.yaml",
        "/absolute/path/to/runtime.yaml",
    ]),
)
client = TrainerClient(backend_config=backend_config)
```
Sources are checked in order; if no source has the runtime, it falls back to the default image
for that framework.

## Data types returned by the SDK

```python
@dataclass
class Runtime:
    name: str
    trainer: RuntimeTrainer     # .trainer_type, .framework, .image, .num_nodes, .device, .device_count
    pretrained_model: str | None = None

@dataclass
class TrainJob:
    name: str
    runtime: Runtime
    steps: list[Step]           # .name, .status, .pod_name, .device, .device_count
    num_nodes: int
    creation_timestamp: datetime
    status: str

@dataclass
class Event:
    involved_object_kind: str
    involved_object_name: str
    message: str
    reason: str
    event_time: datetime
```

## Progress/metrics reporting from a custom loop

```python
from kubeflow.trainer import update_trainjob_status

update_trainjob_status(
    progress_percent=45,
    estimated_remaining_seconds=120,
    metrics={"loss": 0.15, "accuracy": 0.92},
)
```
Requires Kubeflow SDK `>= 0.5.0` and Trainer's `TrainJobStatus` alpha feature gate enabled
server-side. Throttled to ~1 call/5s internally — safe to call every training step. Full detail
including the zero-code HuggingFace Transformers integration and the raw-HTTP protocol for
non-Python clients: [trainjob-runtime-api.md](trainjob-runtime-api.md#trainjob-progress-and-metrics-v220).
