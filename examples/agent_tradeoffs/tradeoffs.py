"""Deterministic teaching models, not a production executor or model benchmark."""
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Receipt:
    operation_id: str
    payload_digest: str
    expected_version: int
    committed_version: int


@dataclass(frozen=True)
class LedgerState:
    version: int = 0
    value: str = ""
    receipts: tuple[Receipt, ...] = ()


def apply(state: LedgerState, operation_id: str, payload: str,
          expected_version: int, *, fail_before_commit: bool = False):
    """Return a new immutable state and receipt; caller publishes them together.

    The ledger stores every accepted operation forever in this small example.
    Lookup precedes CAS: a retry is not a new write. Payload changes under the
    same key, including a changed expected version, are rejected. Real
    concurrency/durability must be supplied elsewhere.
    """
    if not operation_id:
        raise ValueError("operation_id required")
    digest = sha256(payload.encode("utf-8")).hexdigest()
    for receipt in state.receipts:
        if receipt.operation_id == operation_id:
            if receipt.payload_digest != digest:
                raise ValueError("operation_id reused with different payload")
            if receipt.expected_version != expected_version:
                raise ValueError("operation_id reused with different expected_version")
            return state, receipt
    if expected_version != state.version:
        raise ValueError("version conflict")
    receipt = Receipt(operation_id, digest, expected_version, state.version + 1)
    candidate = LedgerState(state.version + 1, payload,
                            state.receipts + (receipt,))
    if fail_before_commit:
        raise RuntimeError("injected failure before state publication")
    return candidate, receipt


@dataclass(frozen=True)
class Job:
    name: str
    duration: int
    dependencies: tuple[str, ...] = ()


def checked_jobs(jobs):
    jobs = tuple(jobs)
    by_name = {job.name: job for job in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("duplicate job")
    if any(job.duration <= 0 or not job.name for job in jobs):
        raise ValueError("positive duration and nonempty name required")
    if any(dep not in by_name for job in jobs for dep in job.dependencies):
        raise ValueError("unknown dependency")
    return by_name


def critical_path(jobs):
    """Longest dependency-path duration, assuming unlimited available slots."""
    pending = checked_jobs(jobs)
    finish = {}
    while pending:
        ready = sorted(name for name, job in pending.items()
                       if all(dep in finish for dep in job.dependencies))
        if not ready:
            raise ValueError("dependency cycle")
        for name in ready:
            job = pending.pop(name)
            finish[name] = job.duration + max(
                (finish[dep] for dep in job.dependencies), default=0)
    return max(finish.values(), default=0)


def schedule(jobs, slots, *, dispatch=0, merge=0):
    """Alphabetic non-preemptive list schedule in abstract time units.

    Dispatch occupies the worker slot for each job. Merge is one serial cost
    after all jobs. There is no extra coordinator bottleneck, communication,
    failure, stochastic duration, or claim of an optimal list ordering.
    """
    if slots < 1 or dispatch < 0 or merge < 0:
        raise ValueError("invalid capacity or overhead")
    pending = checked_jobs(jobs)
    running, done, trace = {}, set(), []
    now = 0
    while pending or running:
        for name in sorted(list(running)):
            if running[name] == now:
                done.add(name)
                del running[name]
        ready = sorted(name for name, job in pending.items()
                       if set(job.dependencies) <= done)
        for name in ready[:slots - len(running)]:
            job = pending.pop(name)
            end = now + dispatch + job.duration
            running[name] = end
            trace.append((name, now, end))
        if running:
            now = min(running.values())
        elif pending:
            raise ValueError("dependency cycle")
    return now + merge, tuple(trace)


@dataclass(frozen=True)
class Artifact:
    base_sha: str
    reads: frozenset[str]
    writes: frozenset[str]


def merge_risks(left: Artifact, right: Artifact, current_sha: str):
    """Detect declared-resource hazards, not all semantic patch conflicts.

    Resource labels can name files or shared interfaces. Accurate read/write
    sets are an assumption; a clean result is not proof of compatibility.
    """
    risks = set()
    if left.base_sha != current_sha or right.base_sha != current_sha:
        risks.add("stale-base")
    if left.writes & right.writes:
        risks.add("write-write")
    if (left.writes & right.reads) or (right.writes & left.reads):
        risks.add("read-write")
    return frozenset(risks)


if __name__ == "__main__":
    jobs = [Job("A", 6), Job("B", 4, ("A",)),
            Job("C", 4, ("A",)), Job("D", 2, ("B", "C"))]
    state, receipt = apply(LedgerState(), "edit-1", "new", 0)
    retried, replay = apply(state, "edit-1", "new", 0)
    print(f"lost-response retry: version={retried.version}, "
          f"receipts={len(retried.receipts)}, same_receipt={receipt == replay}")
    print(f"critical_path={critical_path(jobs)}; "
          f"serial={schedule(jobs, 1)[0]}; "
          f"parallel_2={schedule(jobs, 2)[0]}; "
          f"parallel_4={schedule(jobs, 4)[0]}; "
          f"parallel_2_dispatch1_merge2={schedule(jobs, 2, dispatch=1, merge=2)[0]}")
