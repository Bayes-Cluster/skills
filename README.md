# skills

A collection of [Claude Code](https://claude.com/claude-code) skills. Each skill lives in its
own subdirectory with a `SKILL.md` entry point (plus a `references/` folder for deeper detail
that gets loaded on demand).

## Available skills

| Skill | Description |
|---|---|
| [`kubeflow_trainer`](kubeflow_trainer/SKILL.md) | Kubeflow Trainer v2 — TrainJob/TrainingRuntime/ClusterTrainingRuntime CRDs, the Kubeflow Python SDK (TrainerClient, CustomTrainer, BuiltinTrainer), per-framework guides (PyTorch, DeepSpeed, JAX, MLX, XGBoost, Megatron-Core, TorchTune, Flux), and migration from Training Operator v1 |

## Usage

Symlink or clone the skill directory you want into your Claude Code skills folder:

```bash
ln -s "$(pwd)/kubeflow_trainer" ~/.claude/skills/kubeflow_trainer
```

Claude Code picks up the `name`/`description` from each `SKILL.md`'s frontmatter and loads the
skill automatically when it's relevant to the task at hand.
