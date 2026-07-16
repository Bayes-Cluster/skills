# Monitoring & Lifecycle

Replaces the MCP discovery, monitoring, lifecycle, platform-inspection, and health tools. After a
TrainJob is submitted ([submission-modes.md](submission-modes.md)), everything you do to watch it,
debug it, pause it, or tear it down lives here.

---

## TrainJob status machine

A TrainJob moves through these states (in `.status.conditions` / the jobset phase):

```
Created ──▶ Running ──▶ Completed
   │           │
   │           └──▶ Failed
   │
   ├──▶ Restarting ──▶ Running
   │
   └──▶ Suspended  (flip spec.suspend)
```

```bash
# Current status, terse
kubectl get trainjob <name> -o jsonpath='{.status.conditions[-1:].type}{"\n"}'

# Full conditions
kubectl get trainjob <name> -o jsonpath='{.status.conditions}' | python3 -m json.tool
```

> **Quirk:** a suspended job reports `Created`, not `Suspended`. To confirm a suspend actually took
> effect, check events for the suspend event (see Events below), don't trust the status string.

---

## Discover jobs & runtimes

### List TrainJobs

```bash
kubectl get trainjobs -A                       # all namespaces
kubectl get trainjobs -n <ns>                  # one namespace
kubectl get trainjob <name> -o yaml            # full spec + status
kubectl describe trainjob <name>               # human-readable, includes events
```

The SDK equivalent (returns dataclasses, not text):

```python
from kubeflow.trainer import TrainerClient
jobs = TrainerClient().list_jobs(runtime="torch-distributed")   # optional runtime filter
job  = TrainerClient().get_job("my-trainjob")                    # .steps, .trainer_status
```

### List runtimes

```bash
kubectl get clustertrainingruntime              # cluster-scoped (the common ones)
kubectl get trainingruntime -A                  # namespace-scoped
kubectl get clustertrainingruntime <name> -o yaml
```

Namespace-scoped runtimes take precedence over cluster-scoped on a name clash. The framework label
drives SDK discovery:

```bash
kubectl get clustertrainingruntime -o custom-columns='NAME:.metadata.name,FW:.metadata.labels.trainer\.kubeflow\.org/framework'
```

---

## Logs

This is where direct `kubectl` beats the MCP: the MCP's `get_training_logs(follow=True)` returns a
**static** message (MCP is request/response, no streaming). `kubectl logs -f` actually streams.

A TrainJob builds a JobSet with several replicated jobs: `dataset-initializer`, `model-initializer`,
`node-N`, `launcher`. Get the pods first, then their logs:

```bash
# Pods for one TrainJob
kubectl get pods -l trainer.kubeflow.org/trainjob-name=<name>

# Stream the node-0 training logs
kubectl logs -f -l trainer.kubeflow.org/trainjob-name=<name>,jobset.sigs.k8s.io/job-name=node-0

# A specific pod (more reliable than label selectors when replicas overlap)
kubectl logs -f <pod-name>

# Previous container's logs (crash loop)
kubectl logs <pod-name> --previous
```

SDK equivalent (streams as an iterator — the real streaming the MCP lacks):

```python
for line in TrainerClient().get_job_logs("my-trainjob", step="node-0", follow=True):
    print(line)
```

If logs are empty but the pod is `Running`, the workload may be buffering output — check events and
the pod's restart count. If the pod is `Pending`, there are no logs yet; it hasn't scheduled.

---

## Events

When logs alone don't explain a failure, events usually do. `kubectl describe` shows them, but the
targeted form is faster:

```bash
# Events for one TrainJob object
kubectl get events --field-selector involvedObject.name=<name>,involvedObject.kind=TrainJob --sort-by=.lastTimestamp

# Events in a namespace, most recent first
kubectl get events -n <ns> --sort-by=.lastTimestamp | tail -20
```

Look for: `FailedScheduling`, `Failed`, `BackOff`, `Unhealthy`, `Pulling`/`Pulled`, and the
suspend/resume lifecycle events.

---

## Wait for completion (polling)

No blocking call (the MCP's `wait_for_training` blocks the connection up to 3600s — avoid). Poll
instead:

```bash
# Bash loop: print status every 5s until terminal
while :; do
  st=$(kubectl get trainjob <name> -o jsonpath='{.status.conditions[-1:].type}' 2>/dev/null)
  echo "$(date +%T)  $st"
  case "$st" in Completed|Failed) break;; esac
  sleep 5
done
```

```python
# SDK: callbacks fire on status change, returns the terminal job
from kubeflow.trainer import TrainerClient
job = TrainerClient().wait_for_job_status(
    "my-trainjob",
    status={"Completed"},
    timeout=3600,
    polling_interval=10,
)
```

---

## Suspend & resume

No dedicated tool — it's a `spec.suspend` flip via patch:

```bash
# Suspend (pause without deleting; pods are removed but the JobSet spec is retained)
kubectl patch trainjob <name> --type=merge -p '{"spec":{"suspend":true}}'

# Resume
kubectl patch trainjob <name> --type=merge -p '{"spec":{"suspend":false}}'
```

Remember the quirk: after suspending, `.status` shows `Created`, not `Suspended`. Verify via events.
`activeDeadlineSeconds` (if set) is immutable and its timer **resets on resume**.

---

## Delete & cleanup

```bash
kubectl delete trainjob <name>                  # removes the TrainJob + its JobSet + pods
kubectl delete trainjob -l <my-label>           # batch by label
kubectl delete trainjob --all -n <ns>           # nuke the namespace's jobs (careful)
```

The MCP labels the jobs it creates with `kubeflow-mcp/managed-by=mcp` and restricts non-admin
deletion to only those. **`kubectl` has no such restriction** — you can delete any TrainJob your
RBAC permits, regardless of origin. That's more power but also less guardrail: confirm the name
before hitting enter.

Pods sometimes linger after deletion if a finalizer hangs. Force-clean:

```bash
kubectl patch trainjob <name> --type=merge -p '{"metadata":{"finalizers":[]}}'  # last resort
```

---

## Controller & CRD inspection

For "why isn't my TrainJob making progress / why was it rejected":

```bash
# Controller logs (admission + reconciliation)
kubectl logs -n kubeflow-system deploy/kubeflow-trainer-controller-manager --tail=100

# The JobSet controller (gang scheduling)
kubectl logs -n kubeflow-system deploy/jobset-controller-manager --tail=100

# CRD schema (what fields are valid)
kubectl get crd trainjobs.trainer.kubeflow.org -o jsonpath='{.spec.versions[*].schema.openAPIV3Schema}' | python3 -m json.tool
```

If the controller rejected a TrainJob at admission (e.g. missing framework label, webhook validation
disabled incorrectly), the error appears in the controller logs, often before any pod is created.

---

## Runtime management

```bash
kubectl apply -f my-runtime.yaml                         # create / update
kubectl patch clustertrainingruntime <name> --type=merge -p '<json>'
kubectl get clustertrainingruntime <name> -o yaml | kubectl replace -f -   # full replace
kubectl delete clustertrainingruntime <name>
```

Operational notes:
- A runtime you create **must** carry `trainer.kubeflow.org/framework: <torch|...>` or the SDK
  won't list it.
- Never carry `trainer.kubeflow.org/webhook-validation: disabled` on a runtime you author — that
  label opts out of admission validation and is only for the community's pre-validated built-ins.
- Editing a shared runtime affects every TrainJob referencing it; prefer `runtimePatches` at the
  TrainJob level for per-job overrides (see `kubeflow-trainer` → trainjob-runtime-api.md).

---

## Cluster health

```bash
# Trainer + JobSet controllers up?
kubectl get pods -n kubeflow-system

# Rollout healthy?
kubectl rollout status deploy/kubeflow-trainer-controller-manager -n kubeflow-system
kubectl rollout status deploy/jobset-controller-manager -n kubeflow-system
```

Both deployments should be `Available`. If the controller pod is `CrashLoopBackOff`, the cluster
can't reconcile TrainJobs at all — check its logs before submitting anything.
