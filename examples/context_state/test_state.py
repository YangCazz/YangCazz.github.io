import unittest
from dataclasses import replace
from state import State, Event, Record, reduce, eligible, release_candidate


class StateTests(unittest.TestCase):
    def event(self, id="e1", generation=1, actor="owner", kind="cancel-release", value="", task="export-fix"):
        return Event(id, task, generation, actor, kind, value)

    def test_old_restriction_survives_new_ci(self):
        s, status = reduce(State(), self.event(actor="ci-service", kind="ci-pass", value="a1"))
        self.assertEqual(status, "accepted")
        self.assertEqual(s.ci, "passed")
        self.assertFalse(release_candidate(s))

    def test_ci_cannot_grant_authority(self):
        s = State()
        after, status = reduce(s, self.event(actor="ci-service", kind="allow-release"))
        self.assertEqual((after, status), (s, "unauthorized"))

    def test_cancel_release_does_not_pause_task(self):
        s, _ = reduce(State(release_allowed=True), self.event())
        self.assertEqual(s.phase, "working")
        self.assertFalse(s.release_allowed)
        self.assertEqual(s.generation, 2)

    def test_pause_blocks_results_and_release(self):
        s, _ = reduce(State(release_allowed=True), self.event(kind="pause-task"))
        after, status = reduce(s, self.event(id="late", generation=2, actor="ci-service", kind="ci-pass", value="a1"))
        self.assertEqual((after, status), (s, "paused"))
        self.assertFalse(release_candidate(after))

    def test_old_and_future_generations_rejected(self):
        for gen in (0, 2):
            s = State()
            self.assertEqual(reduce(s, self.event(generation=gen)), (s, "stale-or-future"))

    def test_duplicate_control_event_is_idempotent(self):
        e = self.event()
        s, _ = reduce(State(), e)
        self.assertEqual(reduce(s, e), (s, "duplicate"))

    def test_ci_for_other_sha_rejected(self):
        s = State()
        self.assertEqual(reduce(s, self.event(actor="ci-service", kind="ci-pass", value="older")), (s, "wrong-sha"))

    def test_late_result_cannot_undo_cancel(self):
        s, _ = reduce(State(release_allowed=True), self.event())
        after, status = reduce(s, self.event(id="ci-old", actor="ci-service", kind="ci-pass", value="a1"))
        self.assertEqual((after, status), (s, "stale-or-future"))
        self.assertFalse(release_candidate(after))

    def test_other_task_unaffected(self):
        s = State(task="docs")
        self.assertEqual(reduce(s, self.event()), (s, "wrong-task"))

    def test_current_explicit_grant_and_current_ci(self):
        s, _ = reduce(State(), self.event(kind="allow-release"))
        s, _ = reduce(s, self.event(id="ci", generation=2, actor="ci-service", kind="ci-pass", value="a1"))
        self.assertTrue(release_candidate(s))

    def test_new_sha_invalidates_ci_and_release(self):
        s, status = reduce(State(ci="passed", release_allowed=True), self.event(kind="set-sha", value="b2"))
        self.assertEqual(status, "accepted")
        self.assertEqual((s.generation, s.sha, s.ci, s.release_allowed), (2, "b2", "unknown", False))
        old, reason = reduce(s, self.event(id="old-ci", generation=2, actor="ci-service", kind="ci-pass", value="a1"))
        self.assertEqual((old, reason), (s, "wrong-sha"))
        current, reason = reduce(s, self.event(id="new-ci", generation=2, actor="ci-service", kind="ci-pass", value="b2"))
        self.assertEqual((current.ci, reason), ("passed", "accepted"))
        self.assertFalse(release_candidate(current))
        approved, _ = reduce(current, self.event(id="approve-b2", generation=2, kind="allow-release"))
        checked, _ = reduce(approved, self.event(id="ci-after-approval", generation=3, actor="ci-service", kind="ci-pass", value="b2"))
        self.assertTrue(release_candidate(checked))

    def test_owner_cannot_fabricate_ci_observation(self):
        s = State()
        self.assertEqual(reduce(s, self.event(kind="ci-pass", value="a1")), (s, "unauthorized"))

    def test_paused_control_cannot_revive_task(self):
        s = State(phase="paused")
        self.assertEqual(reduce(s, self.event(kind="allow-release")), (s, "paused"))

    def test_missing_sha_does_not_change_state(self):
        s = State()
        self.assertEqual(reduce(s, self.event(kind="set-sha")), (s, "missing-sha"))

    def test_filter_each_invalid_dimension(self):
        r = Record("ok", "export-fix", "task-ledger", 1, frozenset({"worker"}), 100)
        bad = [replace(r, id="task", task="other"), replace(r, id="source", source="web"),
               replace(r, id="gen", generation=0), replace(r, id="acl", readers=frozenset()),
               replace(r, id="expired", expires_at=10), replace(r, id="revoked", revoked=True)]
        self.assertEqual(eligible([r] + bad, State(), "worker", 10, frozenset({"task-ledger"})), [r])

    def test_unknown_event_no_change(self):
        s = State()
        self.assertEqual(reduce(s, self.event(kind="text-says-merge")), (s, "unknown-kind"))


if __name__ == "__main__":
    unittest.main()
