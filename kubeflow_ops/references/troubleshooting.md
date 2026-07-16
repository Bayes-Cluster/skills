# Troubleshooting

Diagnostic workflow and an error-to-fix table for failed or misbehaving TrainJobs. This is the
runbook layer — for status/logs/events commands see
[monitoring-and-lifecycle.md](monitoring-and-lifecycle.md), for platform-specific workarounds see
[platform-fixes.md](platform-fixes.md).

---

## Diagnostic workflow

Always go in this order — each step narrows the cause:

1. **Status** — `kubectl get trainjob <name>` → is it `Pending`, `Running`, `Failed`, `Restarting`?
2. **Events** — `kubectl get events --field-selector involvedObject.name=<name>` → scheduling/admission failures show here before any pod exists.
3. **Pods** — `kubectl get pods -l trainer.kubeflow.org/trainjob-name=<name>` → are they `Pending`, `CrashLoopBackOff`, `OOMKilled`?
4. **Logs** — `kubectl logs <pod> --previous` (use `--previous` if it crashed) → the actual runtime error.
5. **Match** the symptom in the table below.

---

## Error → fix table

| Symptom / event | Root cause | Fix |
|---|---|---|
| **OOMKilled** (exit 137) | GPU/CPU memory exceeded | Halve `batch_size`; enable QLoRA (`quantize_base=True`); turn on gradient checkpointing; see memory table below |
| **FailedScheduling** | No node matches the resource request | `kubectl describe node` — check free GPUs ([planning-and-preflight.md](planning-and-preflight.md)); lower `gpu_per_node`; add `tolerations` for tainted GPU nodes |
| **ErrImagePull / ImagePullBackOff** | Image not found, or registry auth missing | Verify the image name/tag; pass `image_pull_secrets` (SDK) or `imagePullSecrets` (YAML) |
| **NCCL timeout** (multi-node hang) | Inter-node communication failure | `env={"NCCL_TIMEOUT":"1800","NCCL_P2P_DISABLE":"1"}`; fall back to gloo; check node networking. Note `BuiltinTrainer` can't take `env` — use `CustomTrainer` |
| **403 Forbidden (HuggingFace)** | Gated model, no token | Accept the model license on HF; pass `access_token=<HF_TOKEN>` in the initializer |
| **Read-only file system** | Platform enforces read-only root FS (OpenShift restricted SCC) | Add writable emptyDir volumes — [platform-fixes.md](platform-fixes.md) |
| **Permission denied on `/.local`** | pip writing to read-only root FS | Add the `dot-local` emptyDir (`/.local`); see platform-fixes |
| **ProcessGroupNCCL … no GPUs** | TorchTune (nccl-only) on a CPU cluster | Switch to `CustomTrainer` with a gloo-backend script; don't use torchtune runtimes on CPU |
| **BackOff / CrashLoopBackOff** | Container exits repeatedly | `kubectl logs <pod> --previous` → read the root error (often one of the above) |
| **Script syntax / NameError** | Bad Python in `CustomTrainer` | The function source is wrapped into a body — no top-level indentation errors, no `if __name__` guard, **all imports inside the function** |
| **TrainJob stuck in Created, no pods** | Controller rejected it at admission, or JobSet controller missing | `kubectl logs -n kubeflow-system deploy/kubeflow-trainer-controller-manager`; confirm JobSet controller is Running; check the runtime has its `framework` label |
| **`OOM` but only at checkpoint/eval** | Activation spike outside the train step | Move checkpoint save off rank 0's GPU (to CPU/disk); lower eval batch separately |
| **Hangs silently (multi-node)** | `/dev/shm` too small for NCCL communicators | Mount `/dev/shm` as a memory-backed emptyDir (~33 MB per communicator) — see below |

### The `/dev/shm` fix (multi-node NCCL)

Kubernetes pods default to a 64 MB `/dev/shm`; NCCL needs ~33 MB **per communicator** (TP, DP, etc.
each get one). Megatron-Core and high-communicator-count jobs hang or crash silently. Add to your
submission's `volumes`:

```python
{"name": "dshm", "mount_path": "/dev/shm", "empty_dir": {"medium": "Memory"}}
```

Or in YAML, mount an `emptyDir` with `medium: Memory` onto `/dev/shm` in the runtime's pod template
(or via a per-job `runtimePatches` entry). This is the single most common multi-node silent failure.

---

## GPU memory & batch-size reference

Condensed from [planning-and-preflight.md](planning-and-preflight.md) — duplicated here because this
is where you look when `OOMKilled` hits.

| Params | Full (bf16) | QLoRA (int4) | Typical GPU |
|---|---|---|---|
| 1–3B | ~6–16 GB | ~3–6 GB | T4, RTX 3080 |
| 7–8B | ~24 GB | ~7 GB | A10, RTX 4090 |
| 13B | ~40 GB | ~12 GB | A100-40G |
| 70B | ~140 GB | ~40 GB | A100-80G ×2 |

| Free GPU mem | Safe `batch_size` |
|---|---|
| 8 GB | 1–2 |
| 16 GB | 2–4 |
| 24 GB | 4–8 |
| 40 GB+ | 8–16 |

Rule of thumb on OOM: halve the batch **first** (cheap), reach for QLoRA or a bigger GPU only if
that's not enough.

---

## GPU time-slicing confusion

If the cluster time-slices GPUs, `torchrun --nproc_per_node=auto` reads
`torch.cuda.device_count()`, which may exceed the pod's GPU allocation and launch the wrong process
count. Prefer `num_nodes=N, gpu=1` (multi-node) over `num_nodes=1, gpu=N` (multi-GPU) so the
pod-level GPU count matches what CUDA reports. Symptom: wrong world size, NCCL mismatch errors.

---

## Recovery actions

```bash
# Delete and retry from scratch
kubectl delete trainjob <name>

# Pause without losing the JobSet spec
kubectl patch trainjob <name> --type=merge -p '{"spec":{"suspend":true}}'
# Resume
kubectl patch trainjob <name> --type=merge -p '{"spec":{"suspend":false}}'
```

Note: after suspend the status shows `Created` (not `Suspended`) — confirm via events. The
`activeDeadlineSeconds` timer resets on resume.

---

## Things kubectl surfaces that the MCP hides

- **Streaming logs** — `kubectl logs -f` actually streams; the MCP's `follow=True` is static.
- **No ownership-label friction** — `kubectl delete` works on any TrainJob your RBAC allows; the MCP
  blocks non-admin users from deleting jobs it didn't create.
- **Full env freedom** — raw YAML / `kubectl apply` carries no `fine_tune`-can't-take-`env`
  restriction; patch `env` anywhere the runtime allows.
- **Any CRD, not just TrainJob** — Notebook, PVC, NetworkPolicy are all reachable (see
  [environment-awareness.md](environment-awareness.md)).
