# Bayes Cluster (UIC-STAT USBC)

Bayes is the "First Generation" self-built HPC cluster at BNU-HKBU United International
College's Statistics department (UIC-STAT), publicly referred to as USBC. It's a Beowulf-style
SLURM cluster intended for smaller jobs and for developing/debugging/testing code before scaling
up elsewhere — not a large production supercomputer.

Source docs: https://github.com/Bayes-Cluster/Bayes-Cluster.github.io/tree/main/docs/slurm-cluster
(rendered at https://bayes-cluster.github.io/docs/slurm-cluster/). This reference summarizes
them; re-check the source if something here seems stale, since cluster hardware/partitions
change over time.

## Migration notice: Bayes → Chebyshev

Between 2025 and 2026, Bayes-Cluster is migrating this SLURM-based platform to **Chebyshev**, a
cloud-native, Kubernetes/Kubeflow-based successor. If the user's question is about the *new*
platform (containers, TrainJob, the Kubeflow SDK) rather than classic `sbatch`/`squeue` workflows,
switch to the `kubeflow_trainer` skill instead of this one. Both platforms may run side-by-side
during the migration window.

## Hardware

| Node | CPU | Memory | GPU | Storage |
|---|---|---|---|---|
| Control (login node) | 20 cores | 128 GB | — | 240 GB |
| Compute2030005000 | 20 cores | 128 GB | — | 600 GB |
| Compute2030005001 | 20 cores | 128 GB | — | 3.0 TB |
| Compute2030005002 | 20 cores | 128 GB | Tesla P100 12GB | 4.8 TB |
| Compute2030005003 | 20 cores | 128 GB | Tesla P100 12GB | 4.8 TB |

Compute nodes are 20-core / 2.20 GHz Intel Skylake with 256 GB RAM per the cluster overview (the
per-node table above, from the same docs, lists 128 GB — treat 128–256 GB as the working range
and confirm with `scontrol show node <name>` if it matters for your job).

**The Control node is login/submission only — do not run jobs on it**, aside from a test lasting
at most a few minutes. Everything real goes through `sbatch`/`salloc`.

## Partitions

| Partition | Backing node | Documented max runtime |
|---|---|---|
| `CPU-Compute-1` | Compute2030005000 | 24 hours |
| `CPU-Compute-2` | Compute2030005001 | 7 days |
| `GPU-Compute-1` | Compute2030005002 | 7 days |
| `GPU-Compute-2` | Compute2030005003 | 24 hours |

:::note
The cluster's own docs are internally inconsistent about GPU partition limits — the Quick Start
guide's table says `GPU-Compute-1`/24h and `GPU-Compute-2`/7 days, while a sample `sinfo` output
elsewhere in the same docs shows the reverse (`GPU-Compute-1` at `7-00:00:00`, `GPU-Compute-2` at
`infinite`). **Don't trust either table as authoritative — always run `sinfo` yourself** before
setting `--time`, since the real limit is whatever the partition is currently configured with.
:::

```bash
sinfo
# PARTITION      AVAIL  TIMELIMIT  NODES  STATE NODELIST
# CPU-Compute-1*    up 1-00:00:00      1   idle Compute2030005000
# CPU-Compute-2     up 7-00:00:00      1   idle Compute2030005001
# GPU-Compute-1     up 7-00:00:00      1    mix Compute2030005002
# GPU-Compute-2     up   infinite      1  alloc Compute2030005003
```

`pestat` (a Bayes-installed add-on, not stock SLURM) gives a denser per-node view — CPU load,
free memory, and who's running what:

```bash
pestat
# Hostname          Partition      Node  Num_CPU  CPUload  Memsize  Freemem  Joblist
#                                  State  Use/Tot            (MB)     (MB)   JobId User ...
# Compute2030005000 CPU-Compute-1*  idle    0/40     1.11    128000   128000
# Compute2030005002 GPU-Compute-1    mix    2/40    16.30    128000   128000  1422 user_2
```

## Access

1. **Get an account** — email the administrator at `bayes@uicstat.com` with: name + student/staff
   ID, department, purpose of use, and (for students/visiting staff) a supervisor consent
   letter. (An older Shimo application-form link is documented but the docs themselves mark it
   deprecated in favor of emailing directly.)
2. **Network** — Bayes is only reachable from the UIC campus network, or via UIC-VPN
   (https://itsc.uic.edu.cn/en/info/1030/1119.htm) if off-campus.
3. **SSH in**:
   ```bash
   ssh <user_name>@hpc.uicstat.com
   ```
4. **First login** forces a password change — pick something non-trivial, since input is masked
   with no length/typo feedback (no asterisks shown while typing).
5. **Change password anytime**: `passwd`, run on the cluster.

### Two-Factor Authentication (2FA)

The cluster enforces TOTP-based 2FA via `google-authenticator` for password logins:

```bash
google-authenticator   # run once on the cluster to generate a secret + QR code
```

Scan the QR code with a TOTP app (the docs mention "ndkey", any standard TOTP app — Google
Authenticator, Authy, etc. — works the same way), then log in with password + the 6-digit code
that follows. The secret lives in `~/.google_authenticator` — back it up if you want to move
devices, delete it to disable 2FA. **SSH key-based auth bypasses 2FA entirely** (the assumption
being a trusted local private key doesn't need a second factor). Lost your 2FA device → contact
the administrator to unbind it.

## Storage / quotas

- Default home directory: very limited (~100 MB) — do **not** use it for data or code.
- Real working directory: `~/<user_name>/` (absolute path like
  `/Storage2030005000/home/<user_name>`), quota **50 GB** per account.
- Need more than 50 GB → contact the maintainer (`bayes@uicstat.com`); there's no self-service
  quota increase documented.

## Environment Modules

Software is exposed via Environment Modules — **only works on compute nodes**, not on Control:

```bash
module avail          # what's installable
module load computing/conda/5.3.1
module list           # what's currently loaded
module unload computing/conda/5.3.1
```

Modules seen in the docs (confirm current list with `module avail` — this drifts):
`compiler/cuda/10.2`, `compiler/mpi/openmpi/4.1.1`, `computing/conda/5.3.1`,
`computing/gap/4.11.1`, `computing/MATLAB/2021b`, `computing/SageMath/9.3`.

### Preinstalled software (as documented)

| Category | Software | Version |
|---|---|---|
| Compiler | GCC | 7.3 |
| Compiler | CUDA | 10.2 |
| Compiler | OpenMP | 5.1 |
| Compiler | OpenMPI | 4.1 |
| Software | R | 3.5+ |
| Software | Python | 3.0+ |
| Software | Anaconda | 5.3.1 |
| Software | MATLAB | R2021b, R2022a |
| Software | GAP | 4.11.1 |
| Software | SageMath | 9.4 |

## Installing your own software (no sudo)

Regular users have no `sudo`. Two supported paths:

**1. Conda environments** (preferred for Python/R/Julia):

```bash
salloc && slink                          # get onto a compute node first
module load computing/conda/5.3.1

proxychains4 conda create -n py_env python=3.7
source activate py_env
proxychains4 python -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python main.py
```

**`proxychains4` matters** — outbound internet from compute nodes goes through a proxy; without
it, `pip`/`conda`/GitHub calls will hang or fail. Use the Beijing Foreign Studies University
(BFSU) mirrors for PyPI/CRAN/conda since direct GitHub/PyPI connectivity from the campus
intranet is unreliable:

```bash
# R packages
install.packages("<package_name>", repos="https://mirrors.bfsu.edu.cn/CRAN/")
```

Same pattern for R (`proxychains4 conda install -c conda-forge r-base`) and Julia
(`proxychains4 conda create -n julia_env julia`).

**2. Compile from source into your home directory**, for anything conda doesn't package:

```bash
apt-get source PACKAGE
./configure --prefix=$HOME/<package-path>
make
make install
```

## File transfer

```bash
# SCP — non-interactive, good for scripted transfers
scp /local/path/file user_name@hpc.uicstat.com:/remote/path/file      # push
scp user_name@hpc.uicstat.com:/remote/path/file /local/path/file      # pull

# SFTP — interactive session
sftp user_name@hpc.uicstat.com
put /local/path/file    # push
get /remote/path/file   # pull
```

## Interactive sessions (Bayes-specific attach step)

```bash
salloc -N 1 --cpus-per-task=4 -t 5:00 -p CPU-Compute
# salloc: Granted job allocation 411
slink        # Bayes-specific: switch the shell onto the granted compute node
# ... now actually on the compute node ...
exit         # leave the compute node
exit         # release the salloc allocation
```

Direct `ssh <compute-node>` without an active allocation is refused:
```
Access denied: user <user>(uid=xxx) has no active jobs on this node.
```

## Common issues

**Slow/hanging `pip install` or `conda install`** — prefix with `proxychains4` and/or point at
the BFSU mirror; direct connections from compute nodes to PyPI/conda-forge/GitHub are unreliable
from the campus network.

**`module: command not found`** — you're on the Control node. `module` only works on compute
nodes; get an allocation first (`salloc && slink`, or submit a batch job).

**Home directory fills up unexpectedly** — you're writing to `~` (100 MB quota) instead of
`~/<user_name>/` (50 GB quota). Always work under your named user directory.

**`ssh <compute-node>` refused** — no active job allocation there. Use `salloc`/`sbatch` first,
not a cold SSH.

**Can't tell if a GPU node actually has a free GPU** — `mix` state in `sinfo`/`pestat` means
partially occupied, not fully available. Check `pestat` or `scontrol show node <name>` for
per-resource detail before submitting a GPU job that might otherwise queue behind an
already-saturated card.
