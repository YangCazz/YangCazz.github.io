// npm run test:browser; requires a running Jekyll preview and Playwright Chromium.
// Every unique article is checked, not just the first three anchor elements.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { uniquePostPaths, validatePageState } from './browser-checks.mjs';

const BASE = (process.env.BASE_URL || 'http://127.0.0.1:4000').replace(/\/$/, '');
const OUT = new URL('../_site-verify/', import.meta.url);
const PAGES = ['/', '/blog/', '/resume/', '/showcase/', '/apps/'];

async function settleArticle(page) {
    await page.waitForFunction(() => [...document.querySelectorAll('.mermaid-container')]
        .every(container => container.querySelector('svg, .mermaid-error')));
    await page.waitForFunction(() => Boolean(window.MathJax?.startup?.promise));
    await page.evaluate(() => window.MathJax.startup.promise);
    await page.waitForFunction(() => {
        const headings = document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6');
        return headings.length === 0 || document.querySelectorAll('#blogToc a').length > 0;
    });
}

async function readState(page) {
    return page.evaluate(() => ({
        main: Boolean(document.querySelector('.main-content')),
        resumeNavLinks: [...document.querySelectorAll('.top-nav a[href]')]
            .filter(link => /\/resume(?:\.html)?\/?$/.test(new URL(link.href).pathname))
            .map(link => link.href),
        post: Boolean(document.querySelector('.post-content')),
        headings: document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6').length,
        tocLinks: document.querySelectorAll('#blogToc a').length,
        mermaidContainers: document.querySelectorAll('.mermaid-container').length,
        mermaidSvg: document.querySelectorAll('.mermaid-container svg').length,
        mermaidErrors: document.querySelectorAll('.mermaid-error').length,
        unprocessedMermaid: document.querySelectorAll('pre code.language-mermaid').length,
        mathReady: Boolean(window.MathJax?.startup?.document),
        math: document.querySelectorAll('mjx-container').length,
        mathErrors: document.querySelectorAll('mjx-merror, [data-mjx-error]').length,
        brokenImages: [...document.images]
            .filter(image => image.src.startsWith(location.origin) && image.complete && image.naturalWidth === 0)
            .map(image => image.src),
    }));
}

async function checkInteractions(page) {
    await page.goto(BASE + '/blog/', { waitUntil: 'load' });
    await page.waitForSelector('#calendarDays .calendar-day');
    assert.equal(await page.locator('#calendarData').count(), 1, 'duplicate calendar datasets');
    assert.equal(await page.locator('#tag-network').count(), 1, 'missing keyword-network target');
    const monthNames = ['十月', '九月', '八月', '七月', '六月', '五月', '四月', '三月', '二月'];
    for (const [index, month] of monthNames.entries()) {
        if (index) await page.locator('#prevMonth').click();
        assert.equal(await page.locator('#calendarTitle').innerText(), '2026年 ' + month);
    }
    for (let index = 0; index < 3; index++) await page.locator('#nextMonth').click();
    const day = page.locator('[data-date="2026-05-21"]');
    const expected = await page.locator('#calendarData').evaluate(element =>
        JSON.parse(element.textContent).filter(post => post.date === '2026-05-21').map(post => post.url));
    assert.ok(expected.length > 1, 'same-day regression fixture must contain multiple articles');
    await day.click();
    const popup = page.locator('#calendarPostPopup');
    await popup.waitFor({ state: 'visible' });
    const actual = await popup.locator('.tooltip-link').evaluateAll(links => links.map(link => link.getAttribute('href')));
    assert.deepEqual(actual, expected, 'not all same-day articles are accessible');
    await day.click();
    await popup.waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#calendarTitle').innerText(), '2026年 五月', 'repeated click must not reload');
    await day.click();
    await page.keyboard.press('Escape');
    await popup.waitFor({ state: 'hidden' });
    await day.click();
    await popup.locator('.calendar-tooltip-close').click();
    await popup.waitFor({ state: 'hidden' });
    await day.click();
    await page.locator('#nextMonth').click();
    await popup.waitFor({ state: 'hidden' });
    await page.locator('#prevMonth').click();
    await day.click();
    await popup.locator('.tooltip-link').last().click();
    await page.waitForURL(BASE + expected.at(-1));
    assert.ok(await page.locator('.post-content').count(), 'last same-day article must open');

    await page.goto(BASE + '/blog/#blog-footer', { waitUntil: 'load' });
    await page.locator('a[href="#tag-network"]').click();
    await page.waitForFunction(() => Math.abs(document.querySelector('#tag-network').getBoundingClientRect().top) < 150);
    await page.locator('#tnv2Search').fill('zzzz_no_matching_tag');
    await page.waitForFunction(() => document.querySelector('#tnv2NoMatch').classList.contains('visible'));
    await page.locator('#tnv2Search').fill('持续学习');
    await page.waitForFunction(() => !document.querySelector('#tnv2NoMatch').classList.contains('visible'));
    await page.locator('#tnv2Search').fill('');
    const firstLink = page.locator('.blog-post-title a').first();
    const articleUrl = new URL(await firstLink.getAttribute('href'), BASE).href;
    await firstLink.click({ position: { x: 10, y: 10 } });
    await page.waitForURL(articleUrl);
    await settleArticle(page);
    const lastToc = page.locator('#blogToc a').last();
    const target = await lastToc.getAttribute('href');
    await lastToc.click();
    await page.waitForFunction(selector => {
        const top = document.querySelector(selector).getBoundingClientRect().top;
        const navBottom = document.querySelector('.top-nav')?.getBoundingClientRect().bottom || 0;
        return top >= navBottom + 8 && top < 150;
    }, target);
    await page.goBack();
    assert.ok(new URL(page.url()).pathname.startsWith('/blog'), 'Back must return to blog');
    await page.goForward();
    await page.waitForURL(articleUrl);
}

export async function runChecks({ diagramsOnly = false } = {}) {
    await mkdir(OUT, { recursive: true });
    const launch = { headless: true };
    if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) launch.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
    const browser = await chromium.launch(launch);
    const results = [];
    let errors = [];
    try {
        const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, timezoneId: 'Asia/Shanghai' });
        // Isolate only the external visit counter; local scripts/fonts/images/HTTP remain real.
        await context.route('https://visitor-badge.laobi.icu/**', route => route.fulfill({
            status: 200, contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>',
        }));
        const page = await context.newPage();
        await page.clock.setFixedTime(new Date('2026-10-29T12:00:00Z'));
        page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
        page.on('pageerror', error => errors.push(error.message));
        page.on('response', response => {
            if (response.url().startsWith(BASE + '/') && response.status() >= 400) errors.push('HTTP ' + response.status() + ' ' + response.url());
        });
        page.on('requestfailed', request => {
            if (request.url().startsWith(BASE + '/')) errors.push('Request failed: ' + request.url());
        });
        const response = await page.goto(BASE + '/blog/', { waitUntil: 'load' });
        assert.equal(response?.status(), 200, 'cannot collect article URLs from blog');
        const paths = uniquePostPaths(await page.locator('#blog-list a[href], #blog-archive a[href]').evaluateAll(links => links.map(link => link.getAttribute('href'))));
        const expectedCount = await page.locator('#calendarData').evaluate(element => JSON.parse(element.textContent).length);
        assert.ok(paths.length > 0, 'no articles discovered');
        assert.equal(paths.length, expectedCount, 'article coverage does not match calendar inventory');

        for (const path of [...(diagramsOnly ? [] : PAGES), ...paths]) {
            const post = paths.includes(path);
            errors = [];
            try {
                const response = await page.goto(BASE + path, { waitUntil: 'load' });
                if (post) await settleArticle(page);
                const state = await readState(page);
                validatePageState(response?.status(), state, { post, errors });
                results.push({ path, status: 'PASS', ...state });
                if (process.argv.includes('--screenshot')) {
                    const name = path === '/' ? 'home' : path.split('/').filter(Boolean).join('-');
                    await page.screenshot({ path: fileURLToPath(new URL(name + '.png', OUT)), fullPage: true });
                }
                console.log('PASS ' + path);
            } catch (error) {
                results.push({ path, status: 'FAIL', message: error.message, errors: [...errors] });
                console.error('FAIL ' + path + ': ' + error.message);
            }
        }
        if (!diagramsOnly) {
            errors = [];
            try {
                await checkInteractions(page);
                assert.deepEqual(errors, []);
                results.push({ path: 'interactions', status: 'PASS' });
            } catch (error) {
                results.push({ path: 'interactions', status: 'FAIL', message: error.message, errors: [...errors] });
            }
        }
        const summary = {
            articles: paths.length,
            mermaid: results.reduce((sum, result) => sum + (result.mermaidContainers || 0), 0),
            math: results.reduce((sum, result) => sum + (result.math || 0), 0),
            failures: results.filter(result => result.status === 'FAIL'),
            results,
        };
        await writeFile(new URL(diagramsOnly ? 'mermaid.json' : 'summary.json', OUT), JSON.stringify(summary, null, 2));
        assert.equal(summary.failures.length, 0, JSON.stringify(summary.failures, null, 2));
        console.log('RESULT: PASS (' + paths.length + ' articles; ' + summary.mermaid + ' diagrams; ' + summary.math + ' formulas)');
        return summary;
    } finally {
        await browser.close();
    }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    runChecks().catch(error => { console.error(error.message); process.exitCode = 1; });
}
