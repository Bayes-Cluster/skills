---
name: slurm
description: SLURM workload manager for HPC batch job scheduling — sbatch/srun/salloc job submission, squeue/sacct/sinfo monitoring, partitions, job arrays, dependencies, and environment modules. Includes Bayes Cluster (UIC-STAT USBC) specifics — node/partition layout, SSH+VPN+2FA access, home directory quotas, module system, conda/proxychains software installs. Use when writing or debugging a SLURM batch script, running an interactive HPC session, checking job/queue/node status, or working on the Bayes cluster specifically. For the newer Kubernetes/Kubeflow-based "Chebyshev" platform Bayes is migrating to, use the kubeflow_trainer skill instead.
version: 1.0.0
author: Terence Liu
license: Apache-2.0
tags: [SLURM, HPC, Batch Scheduling, sbatch, srun, salloc, squeue, Job Arrays, MPI, GPU Jobs, Bayes Cluster, UIC-STAT, USBC]
dependencies: [SLURM client tools (sbatch, srun, salloc, squeue, scancel, sacct, sinfo), ssh]
---

# SLURM

## What it is

SLURM (Simple Linux Utility for Resource Management) is the batch scheduler most HPC clusters
use to arbitrate shared compute: you describe the resources a job needs (nodes, CPUs, GPUs,
memory, walltime), submit it, and SLURM queues it until those resources are free, then runs it.

This skill covers two layers:
1. **Generic SLURM** — commands, job script anatomy, array jobs, dependencies — applicable to
   any SLURM cluster.
2. **Bayes Cluster specifics** — the UIC-STAT "USBC" cluster this org (Bayes-Cluster) operates:
   its node layout, partitions, access process, quotas, and module system. Load
   [references/bayes-cluster.md](references/bayes-cluster.md) whenever the user names Bayes,
   USBC, `hpc.uicstat.com`, or asks something that needs cluster-specific facts (partition
   names, quotas, how to get an account) rather than generic SLURM behavior.

**Relationship to `kubeflow_trainer`**: Bayes is explicitly the "First Generation" HPC platform
at this org. Between 2025–2026 it is being migrated to **Chebyshev**, a Kubernetes/Kubeflow-based
successor. If the user is asking about the newer platform (TrainJob, containers, Kubeflow SDK),
switch to the `kubeflow_trainer` skill instead of this one.

## Quick start (Bayes Cluster golden path)

```bash
# 1. SSH in (on-campus network, or UIC-VPN if off-campus)
ssh <user_name>@hpc.uicstat.com

# 2. Check what's free before submitting anything
sinfo
pestat            # Bayes-specific: per-node CPU load / free memory / running jobs

# 3. Load the software you need (module system only works on compute nodes,
#    e.g. inside a job or after `salloc && slink`)
module avail
module load computing/conda/5.3.1

# 4. Write a batch script (see references/job-patterns.md for full templates), then submit it
sbatch run.slurm
# Submitted batch job 4100

# 5. Watch it
squeue -u <user_name>

# 6. Cancel if needed
scancel 4100
```

Minimal `run.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --partition=CPU-Compute-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

module load computing/conda/5.3.1
source activate my_env
python train.py
```

## Core concepts

- **Node** — a physical machine SLURM schedules onto (`sinfo`/`scontrol show node` to inspect).
- **Partition** — a named pool of nodes with its own time limit and access policy (roughly:
  a queue). A job requests one partition with `-p`/`--partition`.
- **Job** — one `sbatch`/`srun`/`salloc` submission; gets a numeric Job ID.
- **Job step** — a unit of work within a job launched by `srun` (a job can have many steps).
- **QOS (Quality of Service)** — an optional policy layer (priority, limits) that can be
  stacked on top of partition rules; not every cluster uses it.
- **Job array** — one `sbatch --array=...` submission that expands into many jobs sharing a
  script, differentiated by `$SLURM_ARRAY_TASK_ID`.
- **Job states** — `PD` PENDING, `R` RUNNING, `CG` COMPLETING, `CD` COMPLETED, `F` FAILED,
  `CA` CANCELLED, `TO` TIMEOUT, `NF` NODE_FAIL. Full list: [references/slurm-commands.md](references/slurm-commands.md#job-state-codes).

## Common workflows

### Check what's available before submitting

```bash
sinfo                       # partitions, node counts, states, time limits
sinfo -N -l                 # per-node detail
squeue                      # everyone's queue
squeue -u $USER             # just yours
```

### Submit a batch job

```bash
sbatch run.slurm            # → "Submitted batch job <id>"
squeue -j <id>               # check its state
scancel <id>                 # cancel it
```

Full flag reference (memory, GPUs, array jobs, dependencies, mail notifications, output
redirection): [references/slurm-commands.md](references/slurm-commands.md).

### Interactive / debug session

Don't `ssh` straight to a compute node — most clusters (including Bayes) reject that unless you
already have a job running there. Allocate first, then attach:

```bash
salloc -N 1 --cpus-per-task=4 -t 5:00 -p CPU-Compute
# salloc: Granted job allocation 411
# ... on Bayes specifically, switch onto the granted node with `slink` ...
slink
# now on the compute node; run your debug session
exit        # leave the compute node
exit        # release the salloc allocation
```

On clusters that don't have a custom attach helper, `srun --pty bash` from inside the `salloc`
allocation does the equivalent.

### GPU job

```bash
#SBATCH --partition=GPU-Compute-1
#SBATCH --gres=gpu:1
```

Check `sinfo -p <gpu-partition> --Format=NodeList,Gres,GresUsed` to see what's actually free
before submitting — a `mix` state means partially occupied, not full.

### Job array (parameter sweep)

```bash
#SBATCH --array=0-15%4     # 16 tasks, ID 0-15, max 4 running at once
python train.py --seed=$SLURM_ARRAY_TASK_ID
```

### Dependency chain (e.g. preprocess → train → eval)

```bash
prep_id=$(sbatch --parsable preprocess.slurm)
train_id=$(sbatch --parsable --dependency=afterok:$prep_id train.slurm)
sbatch --dependency=afterok:$train_id eval.slurm
```

More patterns (multi-node MPI, checkpoint/requeue, output/error redirection conventions):
[references/job-patterns.md](references/job-patterns.md).

### Check job history / accounting

```bash
sacct -j <id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
sacct -u $USER --starttime=today
```

## Command cheat sheet

| Command | Purpose |
|---|---|
| `sbatch script.slurm` | Submit a batch job |
| `srun ...` | Run a job step (inside an allocation, or standalone for a quick foreground job) |
| `salloc ...` | Interactively allocate resources, then attach/run inside them |
| `squeue [-u user\|-j id\|-p partition]` | See queued/running jobs |
| `scancel <id>` \| `scancel -u user` | Cancel job(s) |
| `sinfo [-N -l]` | Partition/node status |
| `sacct -j <id>` | Job accounting history (after it's finished) |
| `scontrol show job <id>` | Full detail on one job (why it's pending, allocated nodes, ...) |
| `module avail \| load \| list \| unload` | Environment Modules — see software the cluster provides |

Full reference for every command's options, output env vars set inside a job, and job state
codes: [references/slurm-commands.md](references/slurm-commands.md).

## Common issues

**Job stays `PD` (pending) forever** — check the reason: `squeue -j <id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"` or `scontrol show job <id>` — the `Reason` field explains it
(`Resources`, `Priority`, `AssocMaxJobsLimit`, `PartitionTimeLimit`, etc.). "Resources" means
you're waiting for capacity; other reasons usually mean the request itself is unsatisfiable
(e.g. asking for more memory/GPUs than any node in the partition has).

**`ssh` to a compute node is refused** — you can only reach a compute node while you have an
active job allocation there (SLURM enforces this via PAM). Get an allocation first
(`salloc` on Bayes, then `slink`/`srun --pty bash`), don't `ssh` cold.

**`module` command not found / does nothing** — on Bayes, Environment Modules only works on
compute nodes, not the control/login node. Get onto a compute node first (via a job or an
interactive allocation).

**Job killed with no clear error** — check `sacct -j <id> --format=JobID,State,ExitCode,MaxRSS` first: `OUT_OF_MEMORY`/`oom-kill` means your `--mem`/`--mem-per-cpu` request was too low for what
the process actually used; `TIMEOUT` means it hit `--time`.

**Array job task IDs don't match what you expect** — `$SLURM_ARRAY_TASK_ID` is per-task, but
`$SLURM_ARRAY_JOB_ID` is the one shared ID for the whole array (what `squeue`/`scancel` often
show as `<job_id>_<task_id>`). Cancel the whole array with `scancel <job_id>`, or one task with
`scancel <job_id>_<task_id>`.

**Bayes-specific**: home-directory quota surprises, VPN/2FA login issues, `module`/conda/GitHub
connectivity workarounds — see
[references/bayes-cluster.md](references/bayes-cluster.md#common-issues).

## Advanced topics

- **Full command/flag/environment-variable reference** (sbatch/srun/salloc/squeue/scancel/sacct/sinfo, job state codes, array-job and dependency syntax in full): [references/slurm-commands.md](references/slurm-commands.md)
- **Job script patterns** (single-node CPU, multi-node MPI, GPU, array jobs, dependency chains, checkpoint/requeue): [references/job-patterns.md](references/job-patterns.md)
- **Bayes Cluster specifics** (hardware/partition table, access & VPN & 2FA, storage quotas, environment modules, software installation via conda/proxychains, file transfer, migration to Chebyshev): [references/bayes-cluster.md](references/bayes-cluster.md)

## Resources

- Official SLURM docs: https://slurm.schedmd.com/
- Command man pages: https://slurm.schedmd.com/sbatch.html, `/srun.html`, `/salloc.html`, `/squeue.html`, `/scancel.html`, `/sacct.html`, `/sinfo.html`
- Job state codes: https://slurm.schedmd.com/job_state_codes.html
- Bayes Cluster docs: https://bayes-cluster.github.io/docs/slurm-cluster/tutorial/quick-start
- Bayes Cluster docs source: https://github.com/Bayes-Cluster/Bayes-Cluster.github.io/tree/main/docs/slurm-cluster
- Bayes Cluster support: bayes@uicstat.com
