# Platform Fixes

Actionable recipes for platform-specific issues (OpenShift SCC, EKS/GKE quirks, GPU tolerations,
NCCL environment). Replaces the MCP `platform-fixes.md` resource. These are copy-paste `volumes` /
`tolerations` / `env` snippets passed to whichever submission mode you're using
([submission-modes.md](submission-modes.md)).

---

## Detect the platform

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,LABELS:.metadata.labels' --no-headers \
  | grep -oE 'node\.openshift\.io|eks\.amazonaws\.com|cloud\.google\.com'
```

| Node label present | Platform |
|---|---|
| `node.openshift.io/os_id` | OpenShift |
| `eks.amazonaws.com/nodegroup` | Amazon EKS |
| `cloud.google.com/gke-nodepool` | Google GKE |
| _(none of the above)_ | Vanilla Kubernetes / bare-metal |

Error-based detection (from `kubectl logs`): `Read-only file system` or `Permission denied on
/.local` / `/.cache` → almost certainly OpenShift restricted SCC.

---

## OpenShift — read-only root filesystem

OpenShift's `restricted` SCC mounts the root filesystem read-only. pip installs, cache writes, and
`/tmp` writes all fail. The fix is always the same: mount writable emptyDirs over the paths the
workload writes to.

**Always pass these on OpenShift** (as the `volumes` parameter to your SDK call):

```python
[
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp",       "mount_path": "/tmp",    "empty_dir": {}},
]
```

Scope differs by submission mode (the easy-to-miss gotcha, restated from submission-modes.md):

- **`BuiltinTrainer` (fine_tune):** these volumes apply to **ALL** replicated jobs (node,
  dataset-initializer, model-initializer). Do **not** add a workspace emptyDir — `/workspace` comes
  from the runtime's PVC.
- **`CustomTrainer` (run_custom_training):** applies to the **node only**; the SDK auto-injects a
  workspace emptyDir at `/workspace`.
- **`CustomTrainerContainer`:** add a workspace emptyDir only if your image writes to `/workspace`.

Fuller set when `/home` is also needed:

```python
[
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp",       "mount_path": "/tmp",    "empty_dir": {}},
  {"name": "home",      "mount_path": "/home",   "empty_dir": {}},
]
```

---

## OpenShift — non-root random UID

OpenShift assigns a random UID (e.g. `1000660000`). Do not assume `root`.

- `HOME` may not be writable — write outputs to `/tmp`, not `/root` or `/home`.
- `/.local` emptyDir (above) fixes most pip issues.
- Avoid `chmod` / `chown` in training scripts — they assume root and fail.
- Don't write to `/etc`.

---

## OpenShift — SCC cannot be changed via the job

SCC (Security Context Constraints) is cluster-level and **cannot** be relaxed from inside a TrainJob.
Use the volume workarounds above instead of escalating privileges. If emptyDirs are genuinely
insufficient, the only path is a cluster-admin action:

```bash
oc adm policy add-scc-to-user anyuid -z <service-account> -n <namespace>
```

This is out of scope for this skill — flag it to the user; they need a cluster admin.

---

## GPU tolerations

GPU nodes are usually tainted `NoSchedule`. Without a toleration, pods requesting `nvidia.com/gpu`
stay `FailedScheduling`. Pass as the `tolerations` parameter:

```python
[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
```

Or rely on the shorthand: `resources_per_node={"gpu": 1}` auto-maps to the `nvidia.com/gpu` resource
(some runtimes also auto-tolerate, but don't count on it — add the toleration explicitly).

---

## NCCL environment (multi-node GPU)

Tune NCCL via `env`. **`CustomTrainer` and `CustomTrainerContainer` accept `env` directly;**
`BuiltinTrainer` does **not** — if you need NCCL tuning on a fine-tune, switch to a `CustomTrainer`
LoRA script.

```python
env={
  "NCCL_DEBUG": "INFO",
  "NCCL_P2P_DISABLE": "1",     # disable peer-to-peer (some EKS/GKE node topologies need it)
  "NCCL_SHM_DISABLE": "1",     # for restricted-network platforms
  "NCCL_TIMEOUT": "1800",      # seconds; raise for large model downloads on rank 0
}
```

When the platform blocks host networking (some SCCs), `NCCL_P2P_DISABLE=1` + `NCCL_SHM_DISABLE=1`
together are the usual unblock.

---

## Platform-specific notes

- **EKS** — GPU nodes need the NVIDIA device plugin (or GPU Operator). If `nvidia.com/gpu` is absent
  from `allocatable`, the AL2/AL2023 AMI lacks the driver; use a GPU AMI or install the operator.
- **GKE** — GDNP (Google's Node Problem Detector) plus the NVIDIA driver job must complete before
  GPUs are allocatable. New GPU node pools take a few minutes after first pod scheduling.
- **Vanilla / bare-metal** — install the NVIDIA GPU Operator or `nvidia-device-plugin` yourself.
  Without it, `nvidia.com/gpu` never appears and every GPU TrainJob fails scheduling.

---

## Quick: which fix do I need?

| Symptom | Likely platform | Jump to |
|---|---|---|
| `Read-only file system` | OpenShift | emptyDir volumes |
| `Permission denied: /.local` | OpenShift | `dot-local` emptyDir |
| `FailedScheduling` + GPU requested | any (missing toleration/plugin) | GPU toleration; check device plugin |
| Multi-node NCCL hang/timeout | any | NCCL env (P2P/SHM disable, timeout) |
| `nvidia.com/gpu` not in allocatable | EKS/GKE/bare-metal | GPU driver/operator install |
| Random UID errors (`chown` fails) | OpenShift | non-root UID rules |
