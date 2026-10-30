import assert from 'node:assert/strict';

export function uniquePostPaths(paths) {
    return [...new Set(paths.filter(path => /^\/\d{4}\/\d{2}\/\d{2}\/[^?#]+\/$/.test(path)))];
}

export function validatePageState(status, state, { post = false, errors = [] } = {}) {
    assert.equal(status, 200, 'page must return HTTP 200');
    assert.ok(state.main, 'main content is missing');
    assert.deepEqual(errors, [], 'browser console/page/resource errors');
    assert.deepEqual(state.brokenImages, [], 'broken local images');
    if (!post) return;
    assert.ok(state.post, 'article body is missing');
    if (state.headings > 0) assert.ok(state.tocLinks > 0, 'article table of contents is missing');
    assert.equal(state.mermaidErrors, 0, 'Mermaid reported a render error');
    assert.equal(state.unprocessedMermaid, 0, 'Mermaid conversion never started');
    assert.equal(state.mermaidSvg, state.mermaidContainers, 'some Mermaid diagrams never rendered');
    assert.ok(state.mathReady, 'MathJax did not initialize');
    assert.equal(state.mathErrors, 0, 'MathJax reported a render error');
}
