# Environment Awareness

The one chunk the `kubeflow/mcp-server` does **not** cover. MCP is Trainer-only; this file lets the
agent perceive the *surroundings* that decide **where a workload should run**: the current
Notebook's GPU, PVC capacity, and whether the namespace can reach the public internet. Use it
*before* choosing between a Notebook session, a TrainJob, or a plain Kubernetes Job.

---

## The five awareness questions

| Question | Why it matters | Section |
|---|---|---|
| Does my Notebook have a GPU? | GPU code run in a CPU Notebook just fails slowly | [1](#1-notebook-gpu) |
| How much PVC space is left? | A 50 GB dataset won't fit on an 8 GB volume | [2](#2-pvc-capacity) |
| Can this namespace reach the public internet? | `pip install` / `hf download` from a no-egress namespace fails | [3](#3-namespace-egress) |
| Is a training task already running/submitted? | Don't double-submit; reuse results | [4](#4-running-workloads) |
| Where should this workload run? | Right-size the execution target | [5](#5-routing-decision) |

---

## 1. Notebook GPU

Kubeflow Notebooks are the `notebooks.kubeflow.org` CRD; the underlying pod carries the real
resource requests. Check both the CRD and the pod (the pod is the source of truth after admission).

```bash
# List notebooks in this namespace
kubectl get notebooks -n <ns>

# Does this notebook request a GPU? (look for nvidia.com/gpu in container resources)
kubectl get notebook <name> -n <ns> -o jsonpath='{.spec.template.spec.containers[*].resources.requests}' | python3 -m json.tool
```

If `nvidia.com/gpu` is absent or `0`, **the notebook has no GPU** — CUDA/torch workloads will run on
CPU (slow or broken). Tell the user and route the workload to a TrainJob requesting GPUs instead of
trying to run it in-place.

The backing pod (post-admission reality, in case a quota/limit stripped the GPU):

```bash
kubectl get pod -n <ns> -l app=<notebook-name> -o jsonpath='{.items[*].spec.containers[*].resources.requests.nvidia\.com/gpu}{"\n"}'
```

---

## 2. PVC capacity

PVC `status` reports the *provisioned capacity* and bind state, not live usage. Two-step check:

```bash
# Provisioned capacity + bind state
kubectl get pvc -n <ns> -o custom-columns='NAME:.metadata.name,CAPACITY:.status.capacity.storage,STATUS:.status.phase,STORAGECLASS:.spec.storageClassName'
```

For **actual free space**, there's no `kubectl` shortcut — exec into any pod that mounts the volume
and run `df`:

```bash
kubectl exec -n <ns> <pod> -- df -h /path/where/pvc/is/mounted
```

(If no pod mounts it, spin up a temporary one, or check the storage backend directly.) Compare
`Used`/`Available` against what the workload needs. Rule of thumb: if available space is under ~20%
of capacity or under the dataset/model size, warn before submitting — a TrainJob that fills the disk
fails with `no space left on device` mid-run.

---

## 3. Namespace egress

Whether `pip install`, `hf download`, or `git clone` will work depends on NetworkPolicies and any
egress proxy.

```bash
# NetworkPolicies targeting this namespace
kubectl get networkpolicy -n <ns>

# Inspect each: a default-deny egress looks like podSelector: {} + policyTypes: [Egress] + egress: []
kubectl get networkpolicy -n <ns> -o yaml | python3 -c "
import sys,yaml
for p in yaml.safe_load_all(sys.stdin):
    if not p: continue
    ps=p.get('spec',{}).get('podSelector',{})
    pt=p.get('spec',{}).get('policyTypes',[])
    eg=p.get('spec',{}).get('egress')
    if ps=={} and 'Egress' in pt and not eg:
        print(f\"{p['metadata']['name']}: DEFAULT-DENY EGRESS (public internet blocked)\")
    elif ps=={}:
        print(f\"{p['metadata']['name']}: broad policy, policyTypes={pt}\")
"
```

Interpreting:

- **No NetworkPolicies at all** → egress is open by default (traditional K8s behavior).
- **A `podSelector: {}` policy with `policyTypes: [Egress]` and empty/no `egress`** → **default-deny
  egress**. `pip install` and model downloads from the public internet will fail unless an egress
  rule allows them or an egress proxy (Squid, NAT, a mirrored registry) is in place.
- **Egress rules listing only private CIDRs / specific hosts** → only those destinations are
  reachable. HuggingFace/GitHub/PyPI must be in the allowlist or pre-proxied.

Workarounds when egress is blocked:
- Pre-stage models/datasets into a PVC and use an `Initializer` pointing at the PVC path.
- Use a private registry mirror for pip (`pip_index_urls`) and container images.
- Run a one-off egress-enabled Job (in an allowed namespace) to download, then mount the PVC.

---

## 4. Running workloads

Before submitting, check nothing is already doing the job:

```bash
kubectl get trainjobs -n <ns>
kubectl get pods -n <ns> --field-selector=status.phase=Running
```

Match by label/annotation if the user reuses names. Don't double-submit identical work — reuse the
existing TrainJob's logs/status (see [monitoring-and-lifecycle.md](monitoring-and-lifecycle.md)).

---

## 5. Routing decision — where should this run?

Use the environment facts above to pick the right execution target.

```
What is the workload?
│
├── Interactive exploration / quick test / editing
│     └── Notebook — IF it has the GPU/memory (§1) and the code is small.
│         No GPU? → warn; route heavy steps out (below).
│
├── Long-running training, needs GPUs, possibly multi-node
│     └── TrainJob (this skill) — gang-scheduled, restartable, logs streamed.
│         PVC too small (§2)? → grow it or stream from DataCache/S3 first.
│         Namespace no-egress (§3)? → pre-stage model/dataset.
│
├── One-off: a test needing 16GB+ RAM, a data-prep script, a cleanup
│     └── Plain Kubernetes Job (kubectl create job) — don't hog the Notebook.
│         Rationale: a 16 GB-RAM test in a 4 GB Notebook OOMs the server pod
│         and can crash the user's session. A Job gets its own pod+resources.
│
└── Already running (§4)? → attach to logs, don't resubmit.
```

The guiding principle: **a Notebook is a shared, long-lived interactive session — don't run
resource-heavy or long-running work inside it.** Anything exceeding the notebook pod's
`resources.requests` (memory/CPU/GPU), anything that runs for minutes, or anything that should
survive a notebook restart belongs in a TrainJob (training) or a Kubernetes Job (everything else).

### Concrete thresholds

- **RAM**: workload peak > notebook's `memory` request → route to a Job/TrainJob. (A 16 GB test in
  a 4 GB notebook gets OOMKilled and may take the kernel down.)
- **GPU**: notebook has 0 GPUs but the code calls `.cuda()` → route to a TrainJob requesting GPUs.
- **Duration**: expected runtime > a few minutes, or you'd be annoyed if the browser tab closed →
  route out of the notebook.
- **Egress**: namespace is default-deny and the workload needs to download → pre-stage or run in an
  egress-enabled namespace.

---

## Tying it together

This chunk is what makes the skill "aware" rather than just "scripted". The pattern:

1. Sense the environment (§1–§4).
2. Decide the target (§5).
3. Hand off: TrainJob submission → [submission-modes.md](submission-modes.md); readiness before that
   → [planning-and-preflight.md](planning-and-preflight.md); watch/debug after →
   [monitoring-and-lifecycle.md](monitoring-and-lifecycle.md) and [troubleshooting.md](troubleshooting.md).
