# Effective task-state teaching example

Requires Python 3.10+; no third-party dependencies, model API, credentials or network.
From repository root:

```sh
python3 -m unittest discover -s examples/context_state -v
```

`state.py` is a pure, in-memory, single-writer reducer. The control generation is
not the code version and not a counter of all events. Any accepted control
change increments generation and conservatively invalidates CI state; this is a
teaching choice, not the optimal production invalidation policy. CI observations
do not change control generation. Changing the code SHA also revokes release
permission, requiring renewed approval. The reducer keeps no historical result
payloads; the diagram's evidence retention is part of the proposed full design.
Record TTL applies only to retrieval eligibility. State.release_allowed has no
clock or expiry and is not an execution-time authorization system.

Trusted adapters must authenticate and bind the `actor`, task, event ID and
version fields before calling this function. The sample does NOT implement
that boundary. User-provided text cannot select an authenticated identity.
The retrieval filter likewise assumes trusted metadata and filters before
payload loading/ranking. It does not establish payload truth.

A true `release_candidate` is only a proposal condition. No external operation
is performed. Recheck current authority and resource version at the execution
boundary. The example does not provide restart durability, distributed
consensus, concurrency control, ID collision detection, audit retention,
production prompt-injection defense or exactly-once external side effects.
Rejected events are not stored; production event intake needs explicit handling
of out-of-order delivery and rejected-event evidence. There is intentionally no
resume operation in this small model.

The 16 tests run on 2026-11-01 cover current CI without release permission,
unauthorized control input, scope-specific cancellation, whole-task pause,
stale/future generation, duplicate accepted event, wrong SHA, late result after
cancel, wrong task, a positive proposal path, each retrieval-filter dimension,
and unknown input. They do not measure language-model behavior or end-to-end
Agent reliability.

Additional tests cover accepted a1-to-b2 transition (including renewed approval),
an owner trying to fabricate CI observations, paused control, and missing SHA.
