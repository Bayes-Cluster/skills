# skills

A collection of [Claude Code](https://claude.com/claude-code) skills. Each skill lives in its
own subdirectory with a `SKILL.md` entry point (plus a `references/` folder for deeper detail
that gets loaded on demand).

## Available skills

| Skill | Description |
|---|---|
| [`kubeflow_trainer`](kubeflow_trainer/SKILL.md) | Kubeflow Trainer v2 — TrainJob/TrainingRuntime/ClusterTrainingRuntime CRDs, the Kubeflow Python SDK (TrainerClient, CustomTrainer, BuiltinTrainer), per-framework guides (PyTorch, DeepSpeed, JAX, MLX, XGBoost, Megatron-Core, TorchTune, Flux), and migration from Training Operator v1 |
| [`kubeflow_ops`](kubeflow_ops/SKILL.md) | Kubeflow Trainer operations — cluster readiness/GPU inventory/model-fit estimation, the three submission modes, TrainJob monitoring (status/logs/events), suspend/resume/cleanup, error-to-fix troubleshooting, platform fixes (OpenShift SCC, EKS/GKE, GPU tolerations, NCCL env), and environment awareness (Notebook GPU, PVC capacity, namespace egress). A dependency-free kubectl + SDK replacement for the kubeflow/mcp-server |
| [`slurm`](slurm/SKILL.md) | SLURM batch scheduling — sbatch/srun/salloc/squeue/sacct/sinfo, job arrays, dependencies, environment modules, plus Bayes Cluster (UIC-STAT USBC) specifics: partitions, access/VPN/2FA, quotas, software installs |

## Usage

Symlink or clone the skill directory you want into your Claude Code skills folder:

```bash
ln -s "$(pwd)/kubeflow_trainer" ~/.claude/skills/kubeflow_trainer
```

Claude Code picks up the `name`/`description` from each `SKILL.md`'s frontmatter and loads the
skill automatically when it's relevant to the task at hand.
