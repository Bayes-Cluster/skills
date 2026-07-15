# SLURM Command Reference

Generic SLURM — applies to any cluster. For Bayes-specific partition names/quotas/tools, see
[bayes-cluster.md](bayes-cluster.md).

## sbatch — submit a batch job

```bash
sbatch [options] script.slurm
sbatch --parsable script.slurm   # print just the job ID (useful for scripting dependency chains)
```

`#SBATCH` directives inside the script are equivalent to CLI flags and are the normal way to set
them; CLI flags on the `sbatch` invocation override directives in the script.

### Resource request flags

| Flag | Meaning |
|---|---|
| `-N, --nodes=<min>[-<max>]` | Node count. Job stays `PENDING` if outside the partition's allowed range; rejected outright if it exceeds the partition's node count. |
| `-n, --ntasks=<n>` | Max number of tasks steps in this job will launch (sbatch doesn't launch tasks itself — this just sizes the allocation). Default: one task per node. |
| `-c, --cpus-per-task=<n>` | CPUs reserved per task. Needed so the controller understands "4 tasks × 3 CPUs each" should land as 4 CPUs-per-node-capable nodes, not just "12 CPUs total". |
| `--ntasks-per-node=<n>` | Tasks per node (alternative to letting `-n`/`-N` imply it). |
| `--mem=<size>[K\|M\|G\|T]` | Memory **per node**. Mutually exclusive with `--mem-per-cpu`/`--mem-per-gpu`. |
| `--mem-per-cpu=<size>` | Memory per allocated CPU. |
| `--gres=<list>` | Generic resources, most commonly GPUs: `--gres=gpu:1`, `--gres=gpu:a100:2`. |
| `--gpus-per-node=[type:]<n>` / `-G, --gpus=<n>` | GPU count (node-scoped or job-total). |
| `-p, --partition=<name>` | Which partition/queue to submit to. |
| `-t, --time=<time>` | Walltime limit. Formats: `minutes`, `min:sec`, `hours:min:sec`, `days-hours`, `days-hours:min`, `days-hours:min:sec`. Job sent `SIGTERM` then `SIGKILL` at the limit. Exceeding the partition's own time limit leaves the job `PENDING` indefinitely — check `sinfo` for the partition's `TIMELIMIT` first. |
| `-o, --output=<pattern>` / `-e, --error=<pattern>` | Stdout/stderr file. `%j` = job ID, `%x` = job name, `%a` = array task ID, `%N` = node name. |
| `-J, --job-name=<name>` | Job name shown in `squeue`. |
| `-a, --array=<indexes>` | Job array — see [Array jobs](#array-jobs) below. |
| `-d, --dependency=<spec>` | Defer start until other jobs reach a state — see [Dependencies](#dependencies). |
| `--exclusive` | Don't share allocated nodes with other jobs. |
| `--mail-type=<type>` / `--mail-user=<addr>` | Email on `BEGIN`/`END`/`FAIL`/`ALL` etc. |
| `-C, --constraint=<features>` | Require nodes with specific features (e.g. `--constraint=haswell`). |
| `--export=<vars\|ALL\|NONE>` | Control which environment variables propagate into the job (default: all of the submitting shell's env). |

### Array jobs

```bash
#SBATCH --array=0-15          # 16 tasks, indices 0..15
#SBATCH --array=0,6,16-32     # explicit list + range, comma-separated
#SBATCH --array=0-15:4        # step function → equivalent to 0,4,8,12
#SBATCH --array=0-15%4        # cap: at most 4 tasks of this array running simultaneously
```

Minimum index is 0; maximum is `MaxArraySize - 1` (a cluster-wide config value). Inside each
task, `$SLURM_ARRAY_TASK_ID` gives that task's index — use it to fan out over a dataset, seed, or
hyperparameter list. `squeue`/`scancel` show/target array tasks as `<job_id>_<task_id>`;
`scancel <job_id>` cancels the whole array at once.

### Dependencies

```bash
sbatch --dependency=afterok:20:21,afterany:23 next.slurm
#   → runs only after job 20 AND 21 both exit 0, AND job 23 has terminated (any exit code)
sbatch --dependency=afterok:20?afterany:23 next.slurm
#   → "?" = OR instead of AND: either condition satisfies it
```

| Type | Condition |
|---|---|
| `after:job_id[+time]` | Starts after the listed job(s) start or are cancelled (+ optional delay in minutes) |
| `afterany:job_id` | After the listed job(s) terminate, any exit code — **default type if none given** |
| `afterok:job_id` | After the listed job(s) terminate with exit code 0 |
| `afternotok:job_id` | After the listed job(s) terminate with a non-zero exit code |
| `aftercorr:job_id` | For array jobs: after the *corresponding* array index in the listed job completes |
| `singleton` | After any previously-launched job with the same name+user finishes |

Chaining pattern:

```bash
prep_id=$(sbatch --parsable preprocess.slurm)
train_id=$(sbatch --parsable --dependency=afterok:$prep_id train.slurm)
sbatch --dependency=afterok:$train_id eval.slurm
```

## srun — launch a job step

```bash
srun [options] command   # inside an sbatch script or salloc allocation: launches a parallel step
srun --pty bash          # inside an allocation: get an interactive shell on the allocated node(s)
srun -n 4 ./mpi_program  # standalone: allocate + run in one shot (for quick one-off parallel runs)
```

Shares most resource flags with `sbatch` (`-N`, `-n`, `-c`, `--mem`, `--gres`, `-t`, ...). Inside
a batch script, multiple `srun` calls create multiple job steps that can run concurrently or
sequentially within the same allocation.

## salloc — interactive resource allocation

```bash
salloc -N 1 --cpus-per-task=4 -t 5:00 -p <partition>
# salloc: Granted job allocation <id>
# drops you into a new shell with the allocation active
```

The new shell is typically still on the login/control node — you then need `srun --pty bash` (or
a cluster-specific attach command) to actually land on the allocated compute node. Exiting the
shell releases the allocation. Never `ssh` directly to a compute node without an active
allocation — most clusters block this via PAM.

## squeue — view the queue

```bash
squeue                           # everyone
squeue -u $USER                  # just your jobs
squeue -j <id>                   # one job
squeue -p <partition>            # one partition
squeue -t PENDING                # filter by state
squeue -l                        # long format
squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"   # custom columns incl. pending Reason
```

### Job state codes

| Code | State | Meaning |
|---|---|---|
| `PD` | PENDING | Waiting for resources/scheduling |
| `R` | RUNNING | Executing |
| `CG` | COMPLETING | Finishing up, some processes may still be active |
| `CD` | COMPLETED | All processes exited with code 0 |
| `F` | FAILED | Non-zero exit or other failure |
| `CA` | CANCELLED | Explicitly cancelled by user or admin |
| `TO` | TIMEOUT | Hit its `--time` limit |
| `NF` | NODE_FAIL | A node it was running on failed |
| `BF` | BOOT_FAIL | Node/block failed to boot |
| `DL` | DEADLINE | Terminated on `--deadline` |
| `CF` | CONFIGURING | Resources allocated, waiting for them to become ready |

Full list: https://slurm.schedmd.com/job_state_codes.html. For a *pending* job, check **why**
with `scontrol show job <id>` — the `Reason` field (`Resources`, `Priority`,
`PartitionTimeLimit`, `AssocMaxJobsLimit`, ...) tells you whether it's just queued behind other
work or whether the request itself can't be satisfied.

## scancel — cancel jobs

```bash
scancel <job_id>                 # one job
scancel <job_id>_<task_id>       # one array task
scancel -u $USER                 # all of your jobs
scancel -t PENDING -u $USER      # all your pending jobs
scancel -p <partition>           # everything in a partition (admin use)
```

## sacct — accounting / job history

```bash
sacct -j <id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
sacct -u $USER --starttime=today
sacct -u $USER --starttime=2026-07-01 --endtime=2026-07-15
```

Most useful for post-mortem debugging: `State=OUT_OF_MEMORY` means raise `--mem`; `State=TIMEOUT`
means raise `--time`; `MaxRSS` tells you actual peak memory used, so you can right-size future
`--mem` requests instead of overbooking.

## sinfo — cluster/partition/node status

```bash
sinfo                              # partitions, node counts by state, time limits
sinfo -N -l                        # one line per node instead of per partition
sinfo -p <partition>
sinfo --Format=NodeList,Gres,GresUsed,StateLong   # check actual free GPUs before submitting
```

Node states you'll see: `idle` (free), `alloc` (fully occupied), `mix` (partially occupied —
still has free resources), `down`/`drain` (unavailable, admin-side issue).

## scontrol — detailed inspection (and limited job control)

```bash
scontrol show job <id>            # full job detail incl. PENDING Reason, allocated nodes, TRES
scontrol show node <name>         # full node detail
scontrol show partition <name>    # partition config: time limits, node list, access
scontrol update job <id> ...      # modify a pending job's settings (admin/owner only, limited fields)
```

## Environment variables set inside a job

SLURM injects these into the batch script / step environment — use them instead of hardcoding
resource counts:

| Variable | Meaning |
|---|---|
| `SLURM_JOB_ID` (`SLURM_JOBID`) | This job's ID |
| `SLURM_JOB_NAME` | Job name |
| `SLURM_JOB_NODELIST` | Nodes allocated to the job |
| `SLURM_JOB_NUM_NODES` (`SLURM_NNODES`) | Node count actually allocated |
| `SLURM_JOB_CPUS_PER_NODE` | CPUs per node, e.g. `72(x2),36` |
| `SLURM_CPUS_PER_TASK` | Set only if `--cpus-per-task` was requested |
| `SLURM_NTASKS` | Total task count |
| `SLURM_PROCID` | This task's global rank (inside an `srun` step) |
| `SLURM_LOCALID` | This task's node-local rank |
| `SLURM_JOB_PARTITION` | Partition the job is running in |
| `SLURM_JOB_ACCOUNT` | Account associated with the job |
| `SLURM_JOB_GPUS` | Global GPU IDs allocated to the job |
| `SLURM_GPUS_ON_NODE` | GPU count allocated on this node |
| `SLURM_ARRAY_JOB_ID` | Shared ID for the whole array |
| `SLURM_ARRAY_TASK_ID` | This task's index within the array |
| `SLURM_ARRAY_TASK_COUNT` / `_MIN` / `_MAX` / `_STEP` | Array shape |
| `SLURM_JOB_END_TIME` / `SLURM_JOB_START_TIME` | UNIX timestamps |
| `SLURM_SUBMIT_DIR` | Directory `sbatch` was invoked from |

Full list: https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES

## Environment Modules (module system)

Most clusters (Bayes included) expose pre-built software via [Environment
Modules](https://modules.sourceforge.net/), not a system package manager:

```bash
module avail              # list everything installable
module load <name>/<ver>  # activate one (adds to PATH/LD_LIBRARY_PATH/etc.)
module list                # what's currently loaded
module unload <name>       # deactivate
module purge                # unload everything
```

`module` commands often only work on compute nodes (inside a job or an interactive allocation),
not on the login/control node — check `bayes-cluster.md` for cluster-specific behavior.
