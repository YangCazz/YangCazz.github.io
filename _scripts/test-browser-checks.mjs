import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { uniquePostPaths, validatePageState } from './browser-checks.mjs';

const valid = {
    main: true, post: true, headings: 2, tocLinks: 2, brokenImages: [],
    mermaidContainers: 2, mermaidSvg: 2, mermaidErrors: 0, unprocessedMermaid: 0,
    mathReady: true, mathErrors: 0,
};
test('existing CommonJS article helper still displays usage without creating a post', () => {
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('./create-blog.js', import.meta.url))], { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.match(result.stdout, /使用方法/);
    assert.equal(result.stderr, '');
});
test('article discovery deduplicates title/read-more links without losing archive entries', () => {
    assert.deepEqual(uniquePostPaths(['/2026/08/03/a/', '/2026/08/03/a/', '/2026/08/02/b/', '/blog/']), ['/2026/08/03/a/', '/2026/08/02/b/']);
});
test('valid page and article checks pass', () => {
    validatePageState(200, valid, { post: true });
    validatePageState(200, { main: true, brokenImages: [] });
});
test('HTTP errors and runtime errors fail the run', () => {
    assert.throws(() => validatePageState(404, valid, { post: true }), /HTTP 200/);
    assert.throws(() => validatePageState(200, valid, { post: true, errors: ['Uncaught ReferenceError'] }), /browser console/);
});
test('missing content and TOC fail the run', () => {
    assert.throws(() => validatePageState(200, { ...valid, post: false }, { post: true }), /article body/);
    assert.throws(() => validatePageState(200, { ...valid, tocLinks: 0 }, { post: true }), /table of contents/);
});
test('Mermaid render errors and unfinished renderings fail the run', () => {
    assert.throws(() => validatePageState(200, { ...valid, mermaidErrors: 1 }, { post: true }), /Mermaid reported/);
    assert.throws(() => validatePageState(200, { ...valid, mermaidSvg: 1 }, { post: true }), /never rendered/);
    assert.throws(() => validatePageState(200, { ...valid, mermaidContainers: 0, mermaidSvg: 0, unprocessedMermaid: 2 }, { post: true }), /never started/);
});
test('MathJax errors and broken images fail the run', () => {
    assert.throws(() => validatePageState(200, { ...valid, mathErrors: 1 }, { post: true }), /MathJax reported/);
    assert.throws(() => validatePageState(200, { ...valid, brokenImages: ['/missing.png'] }, { post: true }), /broken local images/);
});
