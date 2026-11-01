import unittest
from tradeoffs import (Artifact, Job, LedgerState, apply, critical_path,
                       merge_risks, schedule)


class LedgerTests(unittest.TestCase):
    def test_lost_reply_replays_receipt_before_version_check(self):
        state, original = apply(LedgerState(), "one", "patch", 0)
        replayed, receipt = apply(state, "one", "patch", 0)
        self.assertIs(replayed, state)
        self.assertEqual(receipt, original)
        self.assertEqual((replayed.version, len(replayed.receipts)), (1, 1))

    def test_changed_payload_under_same_key_rejected(self):
        state, _ = apply(LedgerState(), "one", "patch", 0)
        with self.assertRaisesRegex(ValueError, "different payload"):
            apply(state, "one", "other patch", 1)

    def test_changed_expected_version_under_same_key_rejected(self):
        state, _ = apply(LedgerState(), "one", "patch", 0)
        with self.assertRaisesRegex(ValueError, "different expected_version"):
            apply(state, "one", "patch", 1)

    def test_lost_reply_cas_alone_cannot_recover_prior_receipt(self):
        state, original = apply(LedgerState(), "one", "patch", 0)
        original_expected_version = 0
        # A CAS-only retry would fail despite the original write succeeding.
        self.assertNotEqual(state.version, original_expected_version)
        replayed, receipt = apply(state, "one", "patch", original_expected_version)
        self.assertIs(replayed, state)
        self.assertEqual(receipt, original)

    def test_fresh_key_with_stale_version_conflicts(self):
        state, _ = apply(LedgerState(), "one", "patch", 0)
        with self.assertRaisesRegex(ValueError, "version conflict"):
            apply(state, "two", "other patch", 0)

    def test_failure_before_publication_has_no_effect(self):
        state = LedgerState()
        with self.assertRaises(RuntimeError):
            apply(state, "one", "patch", 0, fail_before_commit=True)
        self.assertEqual(state, LedgerState())
        retry, _ = apply(state, "one", "patch", 0)
        self.assertEqual(retry.version, 1)

    def test_replay_after_later_operation_does_not_restore_old_value(self):
        state, first = apply(LedgerState(), "one", "first", 0)
        state, _ = apply(state, "two", "second", 1)
        state, receipt = apply(state, "one", "first", 0)
        self.assertEqual(receipt, first)
        self.assertEqual((state.version, state.value), (2, "second"))


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [Job("A", 6), Job("B", 4, ("A",)),
                     Job("C", 4, ("A",)), Job("D", 2, ("B", "C"))]

    def test_critical_path_limits_extra_slots(self):
        self.assertEqual(critical_path(self.jobs), 12)
        self.assertEqual(schedule(self.jobs, 1)[0], 16)
        self.assertEqual(schedule(self.jobs, 2)[0], 12)
        self.assertEqual(schedule(self.jobs, 4)[0], 12)

    def test_overhead_can_erase_parallel_gain(self):
        self.assertEqual(schedule(self.jobs, 2, dispatch=1, merge=2)[0], 17)

    def test_trace_obeys_dependencies_and_capacity(self):
        _, trace = schedule(self.jobs, 2)
        by_name = {name: (start, end) for name, start, end in trace}
        for job in self.jobs:
            for dep in job.dependencies:
                self.assertGreaterEqual(by_name[job.name][0], by_name[dep][1])
        for time in range(16):
            self.assertLessEqual(sum(start <= time < end
                                     for _, start, end in trace), 2)

    def test_cycles_and_unknown_dependencies_rejected(self):
        for jobs in ([Job("A", 1, ("B",)), Job("B", 1, ("A",))],
                     [Job("A", 1, ("missing",))]):
            with self.assertRaises(ValueError):
                critical_path(jobs)
            with self.assertRaises(ValueError):
                schedule(jobs, 2)


class MergeTests(unittest.TestCase):
    def artifact(self, reads=(), writes=(), sha="a1"):
        return Artifact(sha, frozenset(reads), frozenset(writes))

    def test_disjoint_declared_resources(self):
        self.assertEqual(merge_risks(self.artifact(writes=["doc"]),
                                    self.artifact(writes=["test"]), "a1"), set())

    def test_separate_files_can_share_interface_dependency(self):
        left = self.artifact(writes=["api.py", "interface:export"])
        right = self.artifact(reads=["interface:export"], writes=["client.py"])
        self.assertEqual(merge_risks(left, right, "a1"), {"read-write"})

    def test_same_file_write_conflict(self):
        a = self.artifact(writes=["api.py"])
        self.assertEqual(merge_risks(a, a, "a1"), {"write-write"})

    def test_current_base_is_required(self):
        self.assertEqual(merge_risks(self.artifact(), self.artifact(), "b2"),
                         {"stale-base"})


if __name__ == "__main__":
    unittest.main()
