"""Single-process teaching reducer; no authentication, persistence or tool execution."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class State:
    task: str = "export-fix"
    generation: int = 1  # control generation, not an event count
    sha: str = "a1"
    phase: str = "working"
    release_allowed: bool = False
    ci: str = "unknown"
    seen: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Event:
    id: str
    task: str
    generation: int
    actor: str  # supplied by a trusted adapter, NEVER parsed from body
    kind: str
    value: str = ""


def reduce(state: State, event: Event) -> tuple[State, str]:
    if event.task != state.task:
        return state, "wrong-task"
    if event.id in state.seen:
        return state, "duplicate"
    if event.generation != state.generation:
        return state, "stale-or-future"
    changes = {}
    if event.kind == "ci-pass":
        if event.actor != "ci-service":
            return state, "unauthorized"
        if event.value != state.sha:
            return state, "wrong-sha"
        if state.phase != "working":
            return state, "paused"
        changes = {"ci": "passed"}
    elif event.kind in {"set-sha", "allow-release", "cancel-release", "pause-task"}:
        if event.actor != "owner":
            return state, "unauthorized"
        if state.phase != "working":
            return state, "paused"
        # Any control change invalidates in-flight work in this conservative toy.
        changes = {"generation": state.generation + 1, "ci": "unknown"}
        if event.kind == "set-sha":
            if not event.value:
                return state, "missing-sha"
            changes["sha"] = event.value
            changes["release_allowed"] = False  # new code requires renewed approval
        elif event.kind == "allow-release":
            changes["release_allowed"] = True
        elif event.kind == "cancel-release":
            changes["release_allowed"] = False
        else:
            changes.update(phase="paused", release_allowed=False)
    else:
        return state, "unknown-kind"
    return replace(state, **changes, seen=state.seen | {event.id}), "accepted"


def release_candidate(state: State) -> bool:
    """A local proposal gate, NOT permission to perform an external operation."""
    return state.phase == "working" and state.release_allowed and state.ci == "passed"


@dataclass(frozen=True)
class Record:
    id: str
    task: str
    source: str
    generation: int
    readers: frozenset[str]
    expires_at: int
    revoked: bool = False


def eligible(records: list[Record], state: State, reader: str, now: int,
             allowed_sources: frozenset[str]) -> list[Record]:
    """Filter metadata BEFORE loading/ranking payloads; metadata is assumed trusted."""
    return [r for r in records if r.task == state.task
            and r.source in allowed_sources and reader in r.readers
            and r.generation == state.generation and now < r.expires_at
            and not r.revoked]
