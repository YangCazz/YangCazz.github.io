import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { DISABLED_KEY, trackingAllowed, cleanPayload, loadTracker, setOptOut, initAnalytics } from '../assets/js/analytics.js';
const config = { enabled: 'true', production: 'true', trackPage: 'true', websiteId: '12345678-1234-1234-1234-123456789abc', hostname: 'yangcazz.github.io', scriptUrl: 'https://cloud.umami.is/script.js' };
function browser(overrides = {}) {
    const data = new Map();
    return { location: { hostname: config.hostname, protocol: 'https:', origin: 'https://yangcazz.github.io' }, navigator: {}, localStorage: { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, String(value)), removeItem: key => data.delete(key) }, ...overrides };
}
function mockDocument(dataset = config) {
    const elements = new Map([['blog-analytics-config', { dataset }], ['analytics-preference-status', { textContent: '' }]]);
    for (const id of ['analytics-disable', 'analytics-enable']) elements.set(id, { addEventListener: (name, fn) => { elements.get(id)[name] = fn; } });
    return { elements, getElementById: id => elements.get(id), createElement: () => ({ attrs: {}, setAttribute(name, value) { this.attrs[name] = value; } }), head: { appendChild: el => elements.set(el.id, el) } };
}
test('only explicitly configured production site may load', () => {
    assert.equal(trackingAllowed(config, browser()), true);
    for (const patch of [{ enabled: 'false' }, { production: 'false' }, { trackPage: 'false' }, { websiteId: '' }, { websiteId: 'api-key-not-site-id' }, { hostname: 'localhost' }, { scriptUrl: 'https://evil.example/script.js' }]) assert.equal(trackingAllowed({ ...config, ...patch }, browser()), false);
    assert.equal(trackingAllowed(null, browser()), false);
    for (const hostname of ['localhost', '127.0.0.1', 'preview.example', 'yangcazz.github.io.evil.example']) assert.equal(trackingAllowed(config, browser({ location: { hostname, protocol: 'https:' } })), false);
    assert.equal(trackingAllowed(config, browser({ location: { hostname: config.hostname, protocol: 'http:' } })), false);
});
test('browser preferences, automation and unavailable storage fail closed', () => {
    for (const navigator of [{ webdriver: true }, { globalPrivacyControl: true }, { doNotTrack: '1' }]) assert.equal(trackingAllowed(config, browser({ navigator })), false);
    for (const value of [1, '1', 'yes']) {
        assert.equal(trackingAllowed(config, browser({ doNotTrack: value })), false);
        assert.equal(trackingAllowed(config, browser({ navigator: { doNotTrack: value } })), false);
        assert.equal(trackingAllowed(config, browser({ navigator: { msDoNotTrack: value } })), false);
    }
    const b = browser();
    b.localStorage.setItem(DISABLED_KEY, '1');
    assert.equal(trackingAllowed(config, b), false);
    assert.equal(trackingAllowed(config, browser({ localStorage: { getItem() { throw Error('blocked'); } } })), false);
});
test('script loads once with no API credential and minimal collection options', () => {
    const b = browser(), doc = mockDocument();
    assert.equal(loadTracker(config, b, doc), true);
    assert.equal(loadTracker(config, b, doc), false);
    const el = doc.getElementById('umami-tracker');
    assert.equal(el.src, 'https://cloud.umami.is/script.js');
    assert.equal(el.attrs['data-website-id'], config.websiteId);
    for (const option of ['data-exclude-search', 'data-exclude-hash', 'data-do-not-track']) assert.equal(el.attrs[option], 'true');
    assert.equal(el.attrs['data-before-send'], 'blogAnalyticsBeforeSend');
    assert.equal(el.attrs['data-errors'], undefined);
    assert.equal(el.attrs['data-performance'], undefined);
    assert.equal(el.attrs['data-distinct-id'], undefined);
    assert.equal(loadTracker({ ...config, enabled: 'false' }, b, mockDocument()), false);
});
test('payload removes search and fragments in page and referral URLs', () => {
    const payload = { website: config.websiteId, url: '/blog/?q=secret#part', referrer: 'https://example.com/a?private=value#section', title: '文章' };
    const clean = cleanPayload('event', payload, config, browser());
    assert.equal(clean.url, '/blog/');
    assert.equal(clean.referrer, 'https://example.com/a');
    assert.equal(payload.url, '/blog/?q=secret#part', 'do not mutate tracker payload');
    for (const referrer of ['/post/?private=value#section', 'https://yangcazz.github.io/post/?private=value#section']) {
        assert.equal(cleanPayload('event', { ...payload, referrer }, config, browser()).referrer, '/post/');
    }
    assert.equal(cleanPayload('event', { ...payload, url: 'javascript:evil()' }, config, browser()), false);
    assert.equal(cleanPayload('identify', payload, config, browser()), false);
    assert.equal(cleanPayload('event', { ...payload, name: 'click' }, config, browser()), false);
    assert.equal(cleanPayload('event', { ...payload, data: { email: 'private' } }, config, browser()), false);
    assert.equal(cleanPayload('event', null, config, browser()), false);
});
test('changing preference prevents subsequent events, without clearing past data', () => {
    const b = browser(), doc = mockDocument();
    initAnalytics(b, doc);
    assert.ok(doc.getElementById('umami-tracker'));
    doc.getElementById('analytics-disable').click();
    assert.equal(b.localStorage.getItem(DISABLED_KEY), '1');
    assert.equal(b.blogAnalyticsBeforeSend('event', { url: '/blog/' }), false);
    assert.match(doc.getElementById('analytics-preference-status').textContent, /已排除/);
    doc.getElementById('analytics-enable').click();
    assert.equal(b.localStorage.getItem(DISABLED_KEY), null);
    assert.match(doc.getElementById('analytics-preference-status').textContent, /未设置排除/);
});
test('privacy page can change preference but never starts tracker', () => {
    const b = browser(), doc = mockDocument({ ...config, trackPage: 'false' });
    initAnalytics(b, doc);
    assert.equal(doc.getElementById('umami-tracker'), undefined);
    doc.getElementById('analytics-disable').click();
    doc.getElementById('analytics-enable').click();
    assert.equal(doc.getElementById('umami-tracker'), undefined);
});
test('storage write errors are visible and not claimed saved', () => {
    const b = browser({ localStorage: { getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); }, removeItem() { throw Error('blocked'); } } });
    const doc = mockDocument();
    assert.equal(setOptOut(true, b), false);
    initAnalytics(b, doc);
    assert.equal(doc.getElementById('umami-tracker'), undefined);
    assert.match(doc.getElementById('analytics-preference-status').textContent, /不允许/);
    doc.getElementById('analytics-disable').click();
    assert.match(doc.getElementById('analytics-preference-status').textContent, /未保存/);
});
test('template, privacy and CI integration preserve production and test separation', async () => {
    const template = await readFile('_includes/analytics.html', 'utf8');
    assert.match(template, /jekyll.environment == 'production'/);
    assert.match(template, /site.analytics.enabled == true/);
    assert.match(template, /page.analytics == false/);
    const ci = await readFile('.github/workflows/build-deploy.yml', 'utf8');
    assert.match(ci, /JEKYLL_ENV: production/);
    const qa = await readFile('_scripts/verify-pages.mjs', 'utf8');
    assert.match(qa, /addInitScript.*localStorage.setItem\('umami.disabled', '1'\)/);
    assert.match(qa, /gateway.umami.is/);
    const privacy = await readFile('privacy.html', 'utf8');
    assert.match(privacy, /analytics: false/);
    assert.match(privacy, /不是经过身份核验的真人数量/);
    assert.match(privacy, /尚未启用/);
});
