import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import '../assets/js/icons.js';
const ROOT = fileURLToPath(new URL('../', import.meta.url));
const read = name => readFile(new URL('../' + name, import.meta.url), 'utf8');
const icons = JSON.parse(await read('_data/icons.json'));
const names = Object.keys(icons);
async function filesBelow(dir) {
    const entries = await readdir(new URL('../' + dir, import.meta.url), { withFileTypes: true });
    const groups = await Promise.all(entries.map(e => e.isDirectory() ? filesBelow(dir + '/' + e.name) : [dir + '/' + e.name]));
    return groups.flat();
}
const templates = [...(await filesBelow('_layouts')), ...(await filesBelow('_includes')),
    ...(await filesBelow('apps')).filter(f => f.endsWith('.html')),
    ...((await readdir(ROOT)).filter(f => f.endsWith('.html')))];

test('every original icon has a semantic name, title, source path and matching standalone SVG', async () => {
    assert.equal(names.length, 67);
    for (const [name, icon] of Object.entries(icons)) {
        assert.match(name, /^[a-z]+(?:-[a-z]+)*$/);
        assert.ok(icon.title && icon.path);
        const svg = await read(`assets/icons/${name}.svg`);
        assert.ok(svg.includes(`d="${icon.path}"`), name);
        for (const attr of ['viewBox="0 0 24 24"', 'stroke="currentColor"', 'fill="none"', 'stroke-width="1.75"', 'role="img"', 'aria-label=']) assert.ok(svg.includes(attr), name + ' ' + attr);
        assert.doesNotMatch(svg, /<script|https?:\/\/[^w]|onload=|<image/);
    }
});

test('all SVG files are well-formed XML and sprite IDs are unique', () => {
    execFileSync('python3', ['-c', `from pathlib import Path; import xml.etree.ElementTree as E
for p in Path('assets/icons').rglob('*.svg'): E.parse(p)
r=E.parse('assets/icons/sprite.svg').getroot(); ids=[s.attrib['id'] for s in r]; assert len(ids)==len(set(ids))==67`], { cwd: ROOT });
});

test('sprite and runtime use the exact single-source geometry with decorative accessible markup', async () => {
    const sprite = await read('assets/icons/sprite.svg');
    for (const [name, icon] of Object.entries(icons)) {
        assert.ok(sprite.includes(`id="${name}"`));
        assert.ok(sprite.includes(`d="${icon.path}"`));
        const markup = globalThis.SiteIcons.render(name);
        assert.ok(markup.includes(`d="${icon.path}"`));
        assert.ok(markup.includes('aria-hidden="true"'));
        assert.ok(markup.includes('focusable="false"'));
    }
    assert.equal(globalThis.SiteIcons.render('<script>'), '');
    assert.equal(globalThis.SiteIcons.render('__proto__'), '');
    assert.doesNotMatch(globalThis.SiteIcons.render('home', 'x" onload="alert(1)'), /onload="/);
});

test('all literal icon includes, sprite references, dynamic calls and navigation names resolve', async () => {
    const sources = [...templates, ...(await filesBelow('assets/js')), ...(await filesBelow('apps')).filter(f => f.endsWith('.js'))];
    for (const file of sources) {
        const source = await read(file);
        const regex = /(?:include icon\.html name="|sprite\.svg#|SiteIcons\.render\(['"])([a-z-]+)/g;
        for (const match of source.matchAll(regex)) assert.ok(icons[match[1]], file + ': ' + match[1]);
    }
    for (const match of (await read('_data/navigation.yml')).matchAll(/icon: (.+)/g)) assert.ok(icons[match[1]], match[1]);
});

test('old UI symbol glyphs and non-brand Octicon calls are gone from templates and app controls', async () => {
    for (const file of templates) {
        const source = (await read(file)).replace(/<!--[\s\S]*?-->/g, '').replace(/<script\b[\s\S]*?<\/script>/g, '');
        assert.doesNotMatch(source, /{%\s*octicon\b/, file);
        assert.doesNotMatch(source, />[^<>\n]*[⌂◆◉◐◀▶▼↶↷‹›×✕📄][^<>\n]*</, file);
    }
    for (const file of ['apps/PixelKnit/assets/js/app.js','apps/knitting-guide/script.js']) {
        assert.doesNotMatch(await read(file), /innerHTML\s*=\s*['"][📋✏🗑]/u);
    }
});

test('icon-only app controls and calendar/close controls retain explicit accessible labels', async () => {
    for (const file of ['apps/PixelKnit/index.html','apps/knitting-guide/index.html']) {
        const source = await read(file);
        for (const match of source.matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)) {
            if (!match[2].includes('<svg')) continue;
            const visibleText = match[2].replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<[^>]+>/g, '').trim();
            if (!visibleText) assert.match(match[1], /aria-label="[^"]+"/, file);
        }
    }
    const layout = await read('_layouts/default.html');
    for (const id of ['prevMonth','nextMonth','mobileTocClose']) assert.match(layout, new RegExp('id="' + id + '"[^>]*aria-label="[^"]+"'));
});

test('brand artwork is explicitly separate from original icon library and remains locally stored', async () => {
    const brands = JSON.parse(await read('_data/brand_icons.json'));
    assert.deepEqual(Object.keys(brands), ['github','linkedin','twitter']);
    for (const [name, brand] of Object.entries(brands)) {
        assert.ok(!icons[name]);
        assert.ok((await read(`assets/icons/brands/${name}.svg`)).includes(`d="${brand.path}"`));
    }
});
