import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

test('versioned visual manifests resolve to real local assets and sources', () => {
  const output = execFileSync('python3', ['_scripts/check-visual-assets.py'], { encoding: 'utf8' });
  assert.match(output, /0 errors/);
});

test('showcase remains readable without scroll-jacking or fabricated product metrics', () => {
  const html = readFileSync('showcase.html', 'utf8');
  const css = readFileSync('_sass/_showcase.scss', 'utf8');
  for (const id of ['selected-work', 'research-notes', 'working-notes']) {
    assert.ok(html.includes(`id="${id}"`) && html.includes(`href="#${id}"`));
  }
  assert.doesNotMatch(html, /60fps|0\.3mm|SOTA|data-target=|<script/);
  assert.doesNotMatch(css, /scroll-snap-type:\s*y mandatory|user-select:\s*none/);
  assert.match(html, /实际应用截图/);
  assert.match(html, /原创结构示意/);
  assert.match(html, /图片导入仍待实现/);
});

test('showcase anchor links retain native browser history', () => {
  const navigation = readFileSync('assets/js/navigation.js', 'utf8');
  assert.ok(navigation.includes("if (anchor.closest('.workbench')) return;"));
});

test('research diagrams are not cropped beneath article title overlays', () => {
  for (const name of ['2026-08-02-continual-learning-comprehensive-guide','2026-08-03-class-incremental-learning-in-depth']) {
    const post = readFileSync(`_posts/${name}.md`, 'utf8');
    const image = post.match(/^image:\s*[\"']?([^\"'\n]+)/m)?.[1];
    assert.ok(image, `${name}: image metadata is present`);
    assert.match(image, /\.(svg|png|jpe?g|webp)$/i);
    assert.match(post, /^cover_image: "\/assets\/images\/covers\//m);
  }
  assert.match(readFileSync('_layouts/post.html','utf8'), /page\.cover_image == false/);
});

test('user-approved iCaRL scientific style anchor remains byte-identical', () => {
  const anchor = readFileSync('assets/images/research/20261030/icarl-training-classification-v3.png');
  assert.equal(createHash('sha256').update(anchor).digest('hex'), '69b318ffd725e979fef5ff6ce6fc08622d4f2104784f95f547db441a0e70e792');
  const guide = readFileSync('docs/SCIENTIFIC-FIGURE-STYLE.md', 'utf8');
  assert.ok(guide.length > 500, 'style guide must contain a reusable specification');
  assert.match(guide, /icarl-training-classification-v3\.png/);
  assert.match(guide, /69b318ffd725e979fef5ff6ce6fc08622d4f2104784f95f547db441a0e70e792/);
});

test('asset validation rejects missing provenance, hash drift and archived page references', () => {
  execFileSync('python3', ['-c', String.raw`
import json, runpy, tempfile
from pathlib import Path
validate = runpy.run_path('_scripts/check-visual-assets.py')['validate']
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / '_data/visual_assets').mkdir(parents=True)
    (root / 'assets').mkdir()
    (root / '_posts').mkdir()
    (root / 'assets/figure.svg').write_text('<svg/>')
    post = root / '_posts/test.md'
    post.write_text('![figure](/assets/figure.svg)')
    asset = dict(id='test', path='/assets/figure.svg', kind='original-diagram', title='Test', alt='Test', source_url=None, verified_at='2026-10-30', rights='original', used_by=['_posts/test.md'])
    manifest = root / '_data/visual_assets/test.json'
    def check():
        manifest.write_text(json.dumps([asset]))
        return validate(root)[0]
    assert check() == []
    asset['sha256'] = 'wrong'
    assert any('SHA-256 mismatch' in e for e in check())
    del asset['sha256']
    asset['kind'] = 'generated-illustration'
    assert any('generation evidence' in e for e in check())
    asset['generation'] = {'tool': 'image_gen', 'prompt': 'test'}
    assert check() == []
    asset.update(status='archived', used_by=[])
    assert any('archive_reason' in e for e in check())
    asset['archive_reason'] = 'User rejected style'
    assert any('archived asset still referenced' in e for e in check())
    post.write_text('Archived figure removed from active page.')
    assert check() == []
`], { encoding: 'utf8' });
});

test('article heading anchors leave space beneath the fixed navigation', () => {
  assert.match(readFileSync('_sass/_blog-post.scss', 'utf8'), /\.blog-post \.post-content h6 \{\s*scroll-margin-top: 90px;/);
  assert.match(readFileSync('_scripts/verify-pages.mjs', 'utf8'), /top >= navBottom \+ 8/);
});

test('legacy knitting guide imports the particle background as an ES module', () => {
  const html = readFileSync('apps/knitting-guide/index.html', 'utf8');
  assert.match(html, /<script type="module">\s*import \{ ParticleSystem \} from '\.\.\/\.\.\/assets\/js\/particle-system\.js';/);
  assert.doesNotMatch(html, /<script src="[^"]*particle-system\.js"><\/script>/);
});

test('all article covers resolve, with decorative covers separate from research figures', () => {
  const posts = readdirSync('_posts').filter(name => name.endsWith('.md'));
  assert.equal(posts.length, 86);
  for (const name of posts) {
    const front = readFileSync(`_posts/${name}`, 'utf8').split(/^---\s*$/m)[1];
    const image = front.match(/^image:\s*"([^"]+)"/m)?.[1];
    const cover = front.match(/^cover_image:\s*"([^"]+)"/m)?.[1] || image;
    assert.ok(cover, `${name}: cover is defined`);
    assert.doesNotMatch(front, /^cover_image:\s*false/m, `${name}: cover unexpectedly suppressed`);
    assert.doesNotMatch(cover, /\/research\//, `${name}: research diagram must not be cropped as a hero`);
    assert.ok(readFileSync(`.${cover}`).length > 0, `${name}: cover exists`);
  }
  const expected = {
    '2026-06-02-mamba-state-space-model-comprehensive': '/assets/images/covers/ai-dev-tools.jpg',
    '2026-06-03-topology-segmentation-learning': '/assets/images/covers/ai-dev-tools.jpg',
    '2026-06-21-world-models-dreamer-evolution': '/assets/images/posts/world-models/dreamer-evolution-cover.jpg',
    '2026-08-02-continual-learning-comprehensive-guide': '/assets/images/covers/ai-dev-tools.jpg',
    '2026-08-03-class-incremental-learning-in-depth': '/assets/images/covers/ai-dev-tools.jpg',
    '2026-10-31-super-agent-continuous-event-driven-architecture': '/assets/images/covers/ai-dev-tools.jpg',
  };
  for (const [name, cover] of Object.entries(expected)) {
    const post = readFileSync(`_posts/${name}.md`, 'utf8');
    assert.ok(post.includes(`cover_image: "${cover}"`), `${name}: decorative cover retained`);
    assert.match(post, /^image: "\/assets\/images\/research\//m);
    assert.ok(readFileSync(`.${cover}`).length > 0);
  }
});
