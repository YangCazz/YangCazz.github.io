# Two deterministic mechanism experiments

These standard-library-only models accompany the Harness and Subagent articles.
They do not call an LLM, run subprocesses, contact a service, or measure an agent's
success rate. Scheduling numbers are **chosen abstract time units**, not measured
seconds. Tests verify local transition and scheduling rules, not production safety.

From the repository root:

```sh
python3 examples/agent_tradeoffs/tradeoffs.py
python3 -m unittest discover -s examples/agent_tradeoffs -v
```

Actual demo output:

```text
lost-response retry: version=1, receipts=1, same_receipt=True
critical_path=12; serial=16; parallel_2=12; parallel_4=12; parallel_2_dispatch1_merge2=17
```

## 1. Version check and retry identity solve different problems

`apply(state, operation_id, payload, expected_version)` returns a new immutable
`LedgerState` and a `Receipt`. A state stores a value, its integer version, and
accepted operation receipts. The operation key is bound to the UTF-8 payload's
SHA-256 digest **and the original expected version** (stored explicitly in the
receipt). The single target object is implicit in this model; a multi-object API
would also bind its target identity. The key is not an authentication credential.

| Request | Outcome |
|---|---|
| New key, matching version | Replace value, increment version, append receipt |
| Same key, payload, and original expected version, now stale | Return original receipt without another state change |
| Same key, changed payload or expected version | Reject key reuse |
| New key, old expected version | Reject concurrent/stale write |
| Injected failure before returning new state | Caller retains unchanged old state |

The replay lookup must precede the version comparison: a committed request whose
reply was lost is not a fresh write. Replaying an old receipt after a later write
also must not restore the old value. Tests cover both cases.

The model assumes one caller publishes the returned state as a unit. It does not
provide a transaction manager, concurrent compare-and-swap, durable storage,
credential checks, crash recovery, ledger expiry, or any external side effect.
A service would need its actual effect and operation record in the same transaction,
or a separate reconciliation protocol if that is impossible. Discarding a reply in
this experiment means retaining the returned committed state but not using its
receipt; it is not a simulated process crash or remote network.

## 2. Delegation is limited by dependencies, overhead, and conflicts

`Job(name, duration, dependencies)` defines a DAG. The chosen graph is
`A(6) -> {B(4), C(4)} -> D(2)`. `critical_path(jobs)` returns its longest dependency
path, 12. `schedule(jobs, slots, dispatch=0, merge=0)` returns `(makespan, trace)`.
The deterministic list scheduler chooses alphabetically among ready jobs.

With no overhead, one slot requires 16 units, while two and four both require 12.
Extra slots cannot shorten the dependency path. With one dispatch unit per job
occupying its worker slot and a final two-unit serial merge, two slots require 17.
That is slower than the **undelegated/no-overhead sequential baseline** of 16;
it is not slower than the same DAG with overhead and one slot (22). This comparison
illustrates overhead erasing a theoretical gain, not a controlled agent benchmark.

The scheduler assumes fixed positive integer durations, non-preemptive execution,
known dependencies, identical slots, no failures, and no extra coordinator queue.
Its alphabetical list order need not be globally optimal. Real dispatch or merge
costs can have a different dependency structure and must be measured separately.

`Artifact(base_sha, reads, writes)` and `merge_risks(left, right, current_sha)` add a
minimal integration check. Results flag stale bases, write/write collisions, and
read/write dependency invalidation. Resource sets can name interfaces as well as
files: changing `api.py` and `client.py` can conflict through `interface:export`
even though a text merge is clean. A hazard calls for revalidation, serialization,
or redesign; it does not imply that the changes can never be combined.

Read/write declarations are assumed complete. A clean check cannot prove semantic
compatibility, test quality, permission correctness, or absence of undeclared
shared effects. The model does not apply real patches or implement a merge service.

## Verification

15 unittest cases pass: seven ledger transitions, four scheduler checks, and four
artifact-conflict checks. They include pre-publication failure, lost-response replay,
replay after a newer write, request-parameter binding, the CAS-only lost-reply
counterexample, overhead reversal, dependency/capacity constraints,
cycles/unknown dependencies, and conflicts hidden by disjoint filenames.
