"""Regression tests for pedagogical article snippets, not sandbox certification.

Run: python3 tests/test_agent_foundations_examples.py
The tests execute the actual Markdown blocks to prevent code/document drift.
"""
import asyncio
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def article_namespace(filename, marker):
    text = (ROOT / '_posts' / filename).read_text()
    blocks = re.findall(r'```python\n(.*?)\n```', text, re.S)
    block, = [block for block in blocks if marker in block]
    namespace = {}
    exec(compile(block, filename, 'exec'), namespace)
    return namespace


read_file = article_namespace('2026-05-22-harness-engineering-agent-kernel.md',
                              'def read_workspace_file')['read_workspace_file']
NS = article_namespace('2026-05-22-subagent-architecture-isolation.md',
                       'class SubagentOrchestrator')
Task, Orchestrator = NS['SubagentTask'], NS['SubagentOrchestrator']


class ReadExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / 'work'
        self.root.mkdir()
        self.other = base / 'work-other'
        self.other.mkdir()
        (self.root / 'ok.txt').write_text('hello')
        (self.other / 'secret.txt').write_text('outside')

    def test_regular_file(self):
        self.assertEqual(read_file(str(self.root), 'ok.txt'), 'hello')

    def test_absolute_path(self):
        with self.assertRaises(ValueError):
            read_file(str(self.root), str(self.root / 'ok.txt'))

    def test_sibling_prefix_escape(self):
        with self.assertRaises(ValueError):
            read_file(str(self.root), '../work-other/secret.txt')

    def test_symlink_escape(self):
        (self.root / 'escape').symlink_to(self.other, target_is_directory=True)
        with self.assertRaises(ValueError):
            read_file(str(self.root), 'escape/secret.txt')

    def test_directory(self):
        with self.assertRaises(ValueError):
            read_file(str(self.root), '.')

    def test_size_limit(self):
        with self.assertRaises(ValueError):
            read_file(str(self.root), 'ok.txt', 4)
        self.assertEqual(read_file(str(self.root), 'ok.txt', 5), 'hello')

    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            read_file(str(self.root), 'missing.txt')


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_and_input_order(self):
        active = peak = 0
        async def run(task):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return task.instruction
            finally:
                active -= 1
        result = await Orchestrator(run, 2).delegate_all(
            [Task(str(i), str(i)) for i in range(5)])
        self.assertEqual(peak, 2)
        self.assertEqual([r.output for r in result], list(map(str, range(5))))
        self.assertTrue(all(r.success for r in result))

    async def test_failure_does_not_discard_sibling(self):
        async def run(task):
            if task.task_id == 'bad':
                raise RuntimeError('private detail')
            return 'ok'
        result = await Orchestrator(run).delegate_all([Task('bad', ''), Task('good', '')])
        self.assertFalse(result[0].success)
        self.assertEqual(result[0].output, 'task failed: RuntimeError')
        self.assertTrue(result[1].success)

    async def test_timeout_cancels_cooperative_coroutine(self):
        cleaned = asyncio.Event()
        async def run(task):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        result = await Orchestrator(run).delegate_all([Task('slow', '', .01)])
        self.assertFalse(result[0].success)
        self.assertTrue(cleaned.is_set())

    async def test_parent_cancellation_propagates(self):
        started, cleaned = asyncio.Event(), asyncio.Event()
        async def run(task):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        parent = asyncio.create_task(Orchestrator(run).delegate_all([Task('one', '')]))
        await started.wait()
        parent.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await parent
        self.assertTrue(cleaned.is_set())

    async def test_invalid_contract(self):
        async def run(task):
            return 'ok'
        with self.assertRaises(ValueError):
            Orchestrator(run, 0)
        with self.assertRaises(ValueError):
            await Orchestrator(run).delegate_all([Task('same', ''), Task('same', '')])
        with self.assertRaises(ValueError):
            await Orchestrator(run).delegate_all([Task('zero', '', 0)])


class SeriesNavigationTests(unittest.TestCase):
    # Source-level contract tests; rendered links still need Jekyll/browser QA.
    # Intentionally update expected order when another reviewed post is published.
    def test_published_series_order_and_targets(self):
        series = (ROOT / '_data' / 'series.yml').read_text()
        entries = re.findall(r'^- id: agent-systems\n(.*?)(?=^- id:|\Z)',
                             series, re.M | re.S)
        self.assertEqual(len(entries), 1)
        slugs = re.findall(r'^    - slug: (.+)$', entries[0], re.M)
        expected = ['harness-engineering-agent-kernel',
                    'subagent-architecture-isolation',
                    'super-agent-continuous-event-driven-architecture']
        self.assertEqual(slugs, expected)
        self.assertIn('  name: 从 Agent 到持续协作系统', entries[0])
        for slug in slugs:
            posts = list((ROOT / '_posts').glob('*-' + slug + '.md'))
            self.assertEqual(len(posts), 1, f'unresolved/duplicate slug: {slug}')
            text = posts[0].read_text()
            top = '{% include series-nav.html series="agent-systems" position="top" %}'
            bottom = '{% include series-nav.html series="agent-systems" position="bottom" %}'
            self.assertEqual(text.count(top), 1)
            self.assertEqual(text.count(bottom), 1)
            self.assertLess(text.index(top), text.index('## 引言'))
            self.assertTrue(text.rstrip().endswith(bottom))

    def test_original_eight_foundations_not_enrolled(self):
        posts = list((ROOT / '_posts').glob('*-build-ai-agent-0[1-8]-*.md'))
        self.assertEqual(len(posts), 8)
        for post in posts:
            self.assertNotIn('series="agent-systems"', post.read_text())


if __name__ == '__main__':
    unittest.main()
