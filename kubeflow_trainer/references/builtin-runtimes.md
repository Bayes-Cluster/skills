# Per-Framework Runtime Guides

All examples assume `pip install -U kubeflow` and a running Kubeflow Trainer control plane
(see [trainjob-runtime-api.md](trainjob-runtime-api.md#installation)). Every `CustomTrainer(func=...)`
must have **all imports inside the function body** — the SDK ships source, not a closure.

## PyTorch (`torch-distributed`)

Launched via `torchrun`. `torch.distributed` DDP/FSDP/FSDP2/any custom parallelism works as-is.

Distributed env available inside `func`: `dist.get_world_size()`, `dist.get_rank()`,
`os.environ["LOCAL_RANK"]`.

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def fine_tune_qwen():
    import torch, torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import boto3

    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B")
    train_loader = DataLoader(dataset, batch_size=128, sampler=DistributedSampler(dataset))

    for epoch in range(10):
        for batch in train_loader:
            output = model(...)
            model.backward(output.loss)
            model.step()
    if dist.get_rank() == 0:
        boto3.upload_file()   # export the model

job_id = TrainerClient().train(
    runtime=TrainerClient().get_runtime("torch-distributed"),
    trainer=CustomTrainer(
        func=fine_tune_qwen,
        num_nodes=2,
        resources_per_node={"gpu": 4},
        packages_to_install=["transformers>=4.53.0", "boto3"],
    ),
)
```

Inspect pre-installed packages: `TrainerClient().get_runtime_packages(runtime=TrainerClient().get_runtime("torch-distributed"))`.

### PyTorch on AMD ROCm

The built-in `torch-distributed` runtime targets NVIDIA CUDA. For AMD, override the image and
request `amd.com/gpu` instead of `nvidia.com/gpu` — the training code is unchanged (RCCL is a
drop-in NCCL replacement, `torch.cuda` calls work through ROCm's CUDA API surface):

```python
from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow.trainer.options import kubernetes as k8s_options

def train_pytorch_rocm():
    import os, torch, torch.distributed as dist
    dist.init_process_group(backend="nccl")   # RCCL under the hood
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"rank {dist.get_rank()}/{dist.get_world_size()}, GPUs: {torch.cuda.device_count()}")
    dist.destroy_process_group()

job_patch = k8s_options.RuntimePatch(
    training_runtime_spec=k8s_options.TrainingRuntimeSpecPatch(
        template=k8s_options.JobSetTemplatePatch(
            spec=k8s_options.JobSetSpecPatch(replicated_jobs=[
                k8s_options.ReplicatedJobPatch(
                    name="node",
                    template=k8s_options.JobTemplatePatch(spec=k8s_options.JobSpecPatch(
                        template=k8s_options.PodTemplatePatch(spec=k8s_options.PodSpecPatch(
                            tolerations=[{"key": "amd.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                        ))
                    ))
                )
            ])
        )
    )
)

job_id = TrainerClient().train(
    runtime="torch-distributed",
    trainer=CustomTrainer(
        func=train_pytorch_rocm,
        image="rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0",
        num_nodes=2,
        resources_per_node={"amd.com/gpu": 1},
    ),
    options=[job_patch],
)
```
Requires the AMD GPU Operator / `amd.com/gpu` device plugin on the cluster.

## DeepSpeed (`deepspeed-distributed`)

Launched via `mpirun` (OpenMPI hostfile auto-generated, OpenSSH auto-started on workers) — same
launch path DeepSpeed itself expects. All logs land on node-0 (the MPI launcher).

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def fine_tune_t5_deepspeed():
    import os, torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import deepspeed, boto3

    deepspeed.init_distributed(dist_backend="nccl")

    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": 3e-4, "betas": [0.9, 0.95],
                                                     "eps": 1e-8, "weight_decay": 0.1}},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
            "overlap_comm": True, "reduce_scatter": True, "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
    }

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    train_loader = DataLoader(wikihow(tokenizer), batch_size=16, sampler=DistributedSampler(dataset))
    model, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())

    for epoch in range(10):
        for batch in train_loader:
            outputs = model(batch)
            model.backward(outputs.loss)
            model.step()
    if dist.get_rank() == 0:
        boto3.upload_file()

job_id = TrainerClient().train(
    runtime="deepspeed-distributed",
    trainer=CustomTrainer(
        func=fine_tune_t5_deepspeed,
        packages_to_install=["boto3"],
        num_nodes=2,
        resources_per_node={"gpu": 2},
    ),
)
```

ZeRO stages: 0 (off, plain DP), 1 (shard optimizer states), 2 (+ gradients), 3 (+ parameters).
DeepSpeed's JSON config (`train_batch_size`, `zero_optimization`, `fp16`/`bf16`) is passed
directly to `deepspeed.initialize()` inside your function — it's not a separate TrainJob field.

## JAX (`jax-distributed`)

One Kubernetes pod → one JAX process. CPU and GPU only — TPU needs a separate runtime (below)
because `jax[cuda]` and `jax[tpu]` conflict in one image. Your script must explicitly call
`jax.distributed.initialize()`.

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def get_jax_dist():
    import os, jax, jax.distributed as dist
    dist.initialize(
        coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
        process_id=int(os.environ["JAX_PROCESS_ID"]),
    )
    print(f"Local devices: {jax.local_devices()}, Global: {jax.device_count()}")
    import jax.numpy as jnp
    x = jnp.ones((4,))
    y = jax.pmap(lambda v: v * jax.process_index())(x)

job_id = TrainerClient().train(
    runtime=TrainerClient().get_runtime("jax-distributed"),
    trainer=CustomTrainer(func=get_jax_dist),
)
```

### JAX on TPU (GKE)

Needs: a TPU-compatible image (e.g. `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu`),
`google.com/tpu` resource requests, and GKE TPU node selectors/topology — none of which the
default `jax-distributed` runtime provides, so override via `RuntimePatch`:

```python
from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow.trainer.options import kubernetes as k8s_options

def get_jax_tpu_dist():
    import os, jax, jax.distributed as dist
    dist.initialize(
        coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
        process_id=int(os.environ["JAX_PROCESS_ID"]),
    )
    import jax.numpy as jnp
    x = jnp.ones((jax.local_device_count(),))
    p_idx = jnp.array([jax.process_index()] * jax.local_device_count())
    y = jax.pmap(lambda v, p: v * p)(x, p_idx)

node_selector = {
    "cloud.google.com/compute-class": "tpu-multihost-v5-8",
    "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
    "cloud.google.com/gke-tpu-topology": "2x4",
}
# Wrap node_selector in a k8s_options.RuntimePatch(...) targeting replicatedJob "node",
# same nested shape as the ROCm tolerations example above, then pass via train(options=[...]).
```
GKE cluster prerequisite: a TPU `ComputeClass` (see GKE Cloud TPU docs) sized to your topology.

## MLX (`mlx-distributed`)

Apple's NumPy-like array framework. In-cluster training runs on **GPU only** via the MPI backend
(`mpirun` + auto-hostfile + auto-SSH, same mechanism as DeepSpeed) — CPU-only TrainJobs aren't
supported with this runtime. A common pattern: fine-tune on a GPU cluster, then evaluate locally
on Apple Silicon.

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def fine_tune_llama():
    import types, os, mlx.core as mx
    from mlx_lm.lora import train_model, CONFIG_DEFAULTS
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load
    from mlx_lm.generate import generate

    args = types.SimpleNamespace(model="meta-llama/Llama-3.2-3B-Instruct",
                                   data="mlx-community/WikiSQL", train=True)
    for k, v in CONFIG_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    os.environ["HF_TOKEN"] = "<YOUR_HF_TOKEN>"
    model, tokenizer = load(args.model)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    train_model(args, model, train_set, valid_set)

    dist = mx.distributed.init(strict=True, backend="mpi")
    if dist.rank() == 0:
        finetuned_model, finetuned_tokenizer = load(args.model, adapter_path=args.adapter_path)
        print(generate(model=finetuned_model, tokenizer=finetuned_tokenizer,
                        prompt="What is SQL?", max_tokens=1000))

job_id = TrainerClient().train(
    runtime="mlx-distributed",
    trainer=CustomTrainer(func=fine_tune_llama, num_nodes=2, resources_per_node={"gpu": 1}),
)
```

Distributed env: `mx.distributed.size()`, `mx.distributed.rank()`. Gradient averaging:
`mlx.nn.average_gradients(gradients)` (preferred) or `mx.distributed.all_sum(grad) / size()`.

## XGBoost (`xgboost-distributed`)

Workers coordinate via XGBoost's Collective protocol (formerly Rabit). Kubeflow Trainer deploys
workers as a JobSet and injects `DMLC_TRACKER_URI`, `DMLC_TRACKER_PORT`, `DMLC_TASK_ID`,
`DMLC_NUM_WORKER`. Rank-0 starts a `RabitTracker`; every worker joins via
`coll.CommunicatorContext`.

Worker count = `numNodes × workersPerNode` — 1 worker/node for CPU (OpenMP parallelizes across
cores), 1 worker/GPU for GPU training (GPU count derived from `resources_per_node`).

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def xgboost_train_classification():
    import os, xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import collective as coll
    from xgboost.tracker import RabitTracker

    rank = int(os.environ["DMLC_TASK_ID"])
    world_size = int(os.environ["DMLC_NUM_WORKER"])
    tracker_uri = os.environ["DMLC_TRACKER_URI"]
    tracker_port = int(os.environ["DMLC_TRACKER_PORT"])

    tracker = None
    if rank == 0:
        tracker = RabitTracker(host_ip="0.0.0.0", n_workers=world_size, port=tracker_port)
        tracker.start()

    with coll.CommunicatorContext(dmlc_tracker_uri=tracker_uri, dmlc_tracker_port=tracker_port,
                                   dmlc_task_id=str(rank)):
        X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                                     n_classes=2, random_state=42 + rank)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
        dvalid = xgb.QuantileDMatrix(X_valid, label=y_valid, ref=dtrain)
        model = xgb.train(
            {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 6,
             "eta": 0.1, "eval_metric": "logloss"},
            dtrain, num_boost_round=50, evals=[(dvalid, "validation")], verbose_eval=10,
        )
        if coll.get_rank() == 0:
            model.save_model("/workspace/xgboost_model.json")
    if tracker is not None:
        tracker.wait_for()

job_id = TrainerClient().train(
    runtime=TrainerClient().get_runtime("xgboost-distributed"),
    trainer=CustomTrainer(func=xgboost_train_classification, num_nodes=2),
)
```

## Megatron-Core (uses `torch-distributed`, not a dedicated runtime)

Megatron-Core is NVIDIA's library for tensor/pipeline/data parallelism on large transformers.
Since it launches via `torchrun`, it works natively on `torch-distributed` — no separate
runtime.

- **Tensor Parallelism (TP)**: splits each layer's weight matrices across GPUs.
- **Pipeline Parallelism (PP)**: assigns different layers to different GPUs, pipelines
  micro-batches.
- **Data Parallelism (DP)**: replicates the full model, splits data; Megatron-Core's
  `DistributedDataParallel` handles gradient sync.

To reuse an existing HuggingFace transformer instead of building from scratch, see
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for HF↔Megatron-Core
bidirectional conversion with parallelism-aware checkpoints.

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def train_megatron_gpt_tp():
    import os, torch
    from torch.optim import Adam
    from functools import partial
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    # ... dataset setup (see Megatron-LM's run_simple_mcore_train_loop.py example) ...

    parallel_state.destroy_model_parallel()
    rank, world_size, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tp_size = int(os.environ["TP_SIZE"])
    parallel_state.initialize_model_parallel(tp_size, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4,
                                 use_cpu_initialization=True, pipeline_dtype=torch.float32)
    gpt_model = GPTModel(config=config, transformer_layer_spec=get_gpt_layer_local_spec(),
                           vocab_size=100, max_sequence_length=64).to("cuda")
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=False,
                                                    overlap_grad_reduce=False,
                                                    use_distributed_optimizer=False),
        module=gpt_model,
    )
    optim = Adam(gpt_model.parameters())
    # ... training loop with get_forward_backward_func(), finalize_model_grads() ...
    torch.distributed.destroy_process_group()

tensor_model_parallel_size = 2
job_name = TrainerClient().train(
    runtime="torch-distributed",
    trainer=CustomTrainer(
        func=train_megatron_gpt_tp,
        num_nodes=2,
        resources_per_node={"memory": "16Gi", "gpu": 1},
        packages_to_install=["megatron-core", "pybind11"],
        env={"TP_SIZE": str(tensor_model_parallel_size)},
    ),
)
```

**`TP_SIZE` must equal `num_nodes * gpu`** (total GPUs across all nodes).

**Gotcha 1 — C/C++ toolchain**: Megatron-Core compiles a small C++ dataset helper on first use.
The default `pytorch/pytorch:*-runtime` image lacks `make`/`g++`. Either `apt-get install` them
inside your training function (slow, every run) or bake them into a custom runtime image.

**Gotcha 2 — `/dev/shm` too small for NCCL**: Megatron-Core opens one NCCL communicator per
parallel group (TP, DP, ...), each needing ~33 MB of `/dev/shm`. Kubernetes pods default to
64 MB, which isn't enough for multi-group workloads — NCCL fails silently or crashes. Fix: mount
`/dev/shm` as a memory-backed `emptyDir`, either patched into the `torch-distributed`
`ClusterTrainingRuntime` cluster-wide, or per-job via `runtimePatches`:

```yaml
# Per-job (TrainJob.spec.runtimePatches)
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
                      volumes:
                        - name: dshm
                          emptyDir: {medium: Memory}
                      containers:
                        - name: node
                          volumeMounts:
                            - name: dshm
                              mountPath: /dev/shm
```
See https://github.com/NVIDIA/nccl/issues/525.

**Multi-node vs multi-GPU tradeoff**:

| Config | `num_nodes` | `gpu` | `TP_SIZE` | Pods | Notes |
|---|---|---|---|---|---|
| Multi-GPU, 1 node | 1 | 2 | 2 | 1 | Faster — NVLink/PCIe, not network |
| Multi-node | 2 | 1 | 2 | 2 | Needed when >1 machine required |

Both yield `WORLD_SIZE=2, TP_SIZE=2`. Under **GPU time-slicing**, Kubernetes may advertise more
GPUs than `torch.cuda.device_count()` reports in-pod — `torchrun --nproc_per_node=auto` uses the
CUDA count, so prefer the multi-node config in that situation. If forced into multi-node anyway
but the target nodes have multiple physical GPUs, co-schedule the TrainJob's pods onto the same
node with a `podAffinity` `runtimePatches` entry keyed on the JobSet's
`jobset.sigs.k8s.io/jobset-name` label, so NCCL uses NVLink/PCIe instead of the pod network.

## TorchTune BuiltinTrainer (LLM fine-tuning, `torchtune-*` runtimes)

Config-driven — no training loop to write. Full field reference (`TorchTuneConfig`,
`LoraConfig`, `TorchTuneInstructDataset`) is in
[python-sdk.md](python-sdk.md#builtintrainer--torchtuneconfig). Multi-node fine-tuning is **not**
supported for TorchTune; LoRA/QLoRA/DoRA is (Trainer >= v2.1.0, SDK >= v0.2.0). Supported models:
https://github.com/kubeflow/trainer/tree/master/manifests/base/runtimes/torchtune

## Flux Framework (HPC / MPI without SSH)

For HPC-style MPI workloads (e.g. LAMMPS) where Flux's tree-based overlay network replaces
SSH-based MPI bootstrapping — no shared keys, consistent UIDs, or SSH permissions required. The
Flux plugin auto-handles cluster discovery, broker config, CURVE-certificate encryption, Flux
installation via an init container, and wraps your training command automatically. Configured
via a `PodGroupPolicy`/Flux-specific runtime rather than the SDK's `CustomTrainer`; see
https://github.com/kubeflow/trainer/tree/master/examples/flux for a full LAMMPS
`ClusterTrainingRuntime` + `TrainJob` example.

## Next steps

- Runnable notebooks for every framework above: https://github.com/kubeflow/trainer/tree/master/examples
- Full `TrainerClient()` API: [python-sdk.md](python-sdk.md)
- TrainJob/TrainingRuntime YAML API (MLPolicy, RuntimePatches, lifecycle): [trainjob-runtime-api.md](trainjob-runtime-api.md)
