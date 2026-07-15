# SLURM Job Script Patterns

Generic templates. Swap `--partition` for whatever this cluster calls its queues (see
[bayes-cluster.md](bayes-cluster.md#partitions) for Bayes's names) and adjust `module load` /
`conda activate` to the environment you actually need.

## Single-node CPU job

```bash
#!/bin/bash
#SBATCH --job-name=cpu-job
#SBATCH --partition=CPU-Compute-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load computing/conda/5.3.1
source activate my_env

python train.py --num-workers=$SLURM_CPUS_PER_TASK
```

## GPU job

```bash
#!/bin/bash
#SBATCH --job-name=gpu-job
#SBATCH --partition=GPU-Compute-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

module load compiler/cuda/10.2
source activate my_env

nvidia-smi   # sanity check the GPU actually attached
python train.py --device=cuda
```

Check what's actually free before submitting — `mix` state means partially occupied, not fully
free:

```bash
sinfo -p GPU-Compute-1 --Format=NodeList,Gres,GresUsed,StateLong
```

## Multi-node MPI job

```bash
#!/bin/bash
#SBATCH --job-name=mpi-job
#SBATCH --partition=CPU-Compute-2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

module load compiler/mpi/openmpi/4.1.1

# srun is the SLURM-native launcher and is generally preferred over mpirun
# on SLURM clusters — it reads the allocation directly (no hostfile needed).
srun ./mpi_program

# If your MPI build doesn't have Slurm/PMI integration and you must use mpirun:
# mpirun -np $SLURM_NTASKS ./mpi_program
```

For applications that need a specific PMI version (`pmi2`, `pmix`), pass it explicitly:
`srun --mpi=pmix ./mpi_program` (`srun --mpi=list` shows what the cluster's SLURM build supports).

## Job array (parameter sweep / many independent runs)

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=CPU-Compute-1
#SBATCH --array=0-19%5          # 20 configs, at most 5 running concurrently
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out   # %A = array job ID, %a = task index

module load computing/conda/5.3.1
source activate my_env

# Use the task index to select this run's config/seed/data shard
python train.py --config=configs/run_${SLURM_ARRAY_TASK_ID}.yaml
```

Cancel the whole sweep: `scancel $SLURM_ARRAY_JOB_ID` (or the printed job ID). Cancel one task:
`scancel <job_id>_<task_id>`.

## Dependency chain (preprocess → train → evaluate)

```bash
#!/bin/bash
prep_id=$(sbatch --parsable preprocess.slurm)
train_id=$(sbatch --parsable --dependency=afterok:$prep_id train.slurm)
eval_id=$(sbatch --parsable --dependency=afterok:$train_id eval.slurm)
echo "preprocess=$prep_id train=$train_id eval=$eval_id"
```

Each stage only starts once the previous one exits 0. If `preprocess.slurm` fails, `train.slurm`
and `eval.slurm` never run (they stay queued, then get cancelled once the dependency is
permanently unsatisfiable).

## Interactive debugging session

```bash
salloc -N 1 --cpus-per-task=4 --mem=16G -t 01:00:00 -p CPU-Compute-1
# On Bayes specifically, this drops a shell on the control node; switch onto
# the granted compute node with `slink`. On other clusters, `srun --pty bash`
# from inside the allocation does the equivalent.
slink   # (Bayes) — or: srun --pty bash

module load computing/conda/5.3.1
source activate my_env
python -i debug_script.py   # interactive Python session on the compute node

exit   # leave the compute node
exit   # release the salloc allocation
```

Don't `ssh` cold to a compute node — you'll be rejected unless you already have an active job
allocation there.

## Long-running job with checkpoint + auto-requeue

For jobs that exceed the partition's time limit or might get preempted, checkpoint periodically
and let SLURM restart the script from the top (your own code must detect and resume from the
last checkpoint):

```bash
#!/bin/bash
#SBATCH --job-name=long-train
#SBATCH --partition=CPU-Compute-2
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120     # send SIGUSR1 to the batch shell 120s before the time limit

trap 'echo "caught SIGUSR1, checkpointing..."; touch checkpoint_requested; wait' USR1

module load computing/conda/5.3.1
source activate my_env

python train.py --resume-from-checkpoint --checkpoint-dir=./ckpt &
wait
```

`--requeue` makes the job eligible to restart from scratch (same job ID) on node failure or
preemption; combine with `--signal` + a trap so your own checkpointing logic gets a chance to
save state before `SIGTERM`/`SIGKILL` actually lands.

## Output/error file conventions

```bash
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job ID
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%A_%a.out # for array jobs: %A = array job ID, %a = task index
```

Create the `logs/` directory before submitting — SLURM does **not** create missing output
directories, and the job silently fails to write logs (or fails outright) if the path doesn't
exist.
