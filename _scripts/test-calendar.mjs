// 无浏览器依赖的日历回归：node --test _scripts/test-calendar.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
    shiftCalendarMonth,
    getCalendarPostsForDate,
    renderCalendarPosts,
    positionCalendarPopup,
} from '../assets/js/calendar.js';

function assertPopupFitsWithoutCoveringAnchor(anchor, popup, viewport) {
    const result = positionCalendarPopup(anchor, popup, viewport);
    assert.ok(result.left >= 10);
    assert.ok(result.top >= 10);
    assert.ok(result.left + popup.width <= viewport.width - 10);
    assert.ok(result.top + result.maxHeight <= viewport.height - 10);
    assert.ok(
        result.left >= anchor.right + 10 || result.left + popup.width <= anchor.left - 10 ||
        result.top >= anchor.bottom + 10 || result.top + result.maxHeight <= anchor.top - 10,
        'popup must leave the entire date button available for another click',
    );
    return result;
}

test('tall desktop popup moves beside its date instead of covering it', () => {
    const viewport = { width: 1280, height: 720 };
    const popup = { width: 320, height: 480 };
    for (const left of [150, 1040]) {
        const anchor = { left, right: left + 36, top: 320, bottom: 356, width: 36 };
        const result = assertPopupFitsWithoutCoveringAnchor(anchor, popup, viewport);
        assert.equal(result.maxHeight, 480, 'desktop side placement retains full scroll height');
    }
});

test('narrow-screen popup scrolls in available space above or below the date', () => {
    const viewport = { width: 375, height: 667 };
    const popup = { width: 280, height: 466 };
    for (const top of [10, 220, 400, 611]) {
        const anchor = { left: 160, right: 196, top, bottom: top + 36, width: 36 };
        const result = assertPopupFitsWithoutCoveringAnchor(anchor, popup, viewport);
        assert.ok(result.maxHeight > 0);
        assert.ok(result.maxHeight <= popup.height);
    }
});

test('small popups retain normal below/above placement when they fit', () => {
    const viewport = { width: 1280, height: 720 };
    const popup = { width: 320, height: 180 };
    for (const top of [100, 600]) {
        const anchor = { left: 300, right: 336, top, bottom: top + 36, width: 36 };
        const result = assertPopupFitsWithoutCoveringAnchor(anchor, popup, viewport);
        assert.equal(result.maxHeight, 180);
        assert.equal(result.top, top === 100 ? 146 : 410);
    }
});

test('short months never overflow when navigating from day 29, 30, or 31', () => {
    for (const day of [29, 30, 31]) {
        const original = new Date(2026, 2, day, 12);
        const previous = shiftCalendarMonth(original, -1);
        assert.deepEqual([previous.getFullYear(), previous.getMonth(), previous.getDate()], [2026, 1, 1]);
        assert.equal(original.getDate(), day, 'navigation must not mutate its input');
    }
    const next = shiftCalendarMonth(new Date(2026, 0, 31), 1);
    assert.deepEqual([next.getFullYear(), next.getMonth(), next.getDate()], [2026, 1, 1]);
});

test('leap years and year boundaries retain the requested month', () => {
    const leapFebruary = shiftCalendarMonth(new Date(2024, 2, 31), -1);
    assert.deepEqual([leapFebruary.getFullYear(), leapFebruary.getMonth(), leapFebruary.getDate()], [2024, 1, 1]);
    const january = shiftCalendarMonth(new Date(2026, 11, 31), 1);
    assert.deepEqual([january.getFullYear(), january.getMonth(), january.getDate()], [2027, 0, 1]);
    const december = shiftCalendarMonth(january, -1);
    assert.deepEqual([december.getFullYear(), december.getMonth(), december.getDate()], [2026, 11, 1]);
});

test('repeated previous/next clicks do not skip or repeat February', () => {
    let date = new Date(2026, 9, 29);
    const months = [date.getMonth()];
    for (let index = 0; index < 8; index++) {
        date = shiftCalendarMonth(date, -1);
        months.push(date.getMonth());
    }
    assert.deepEqual(months, [9, 8, 7, 6, 5, 4, 3, 2, 1]);
    for (let index = 0; index < 8; index++) date = shiftCalendarMonth(date, 1);
    assert.equal(date.getMonth(), 9);
});

const sameDayPosts = Array.from({ length: 7 }, (_, index) => ({
    date: '2026-05-21',
    title: `同日文章 ${index + 1}`,
    excerpt: `摘要 ${index + 1}`,
    tags: ['AI Agent', 'PyTorch'],
    url: `/2026/05/21/article-${index + 1}/`,
}));

test('date selection uses local calendar components and retains every same-day post', () => {
    const posts = [...sameDayPosts, { ...sameDayPosts[0], date: '2026-05-20', url: '/other/' }];
    assert.deepEqual(getCalendarPostsForDate(posts, new Date(2026, 4, 21, 23, 59)), sameDayPosts);
    assert.deepEqual(getCalendarPostsForDate(posts, new Date(2026, 4, 22)), []);
});

test('multi-article popup renders one distinct direct link for each article', () => {
    const html = renderCalendarPosts(sameDayPosts);
    const links = [...html.matchAll(/<a href="([^"]+)"/g)].map(match => match[1]);
    assert.deepEqual(links, sameDayPosts.map(post => post.url));
    assert.equal(new Set(links).size, 7);
    for (const post of sameDayPosts) assert.ok(html.includes(post.title));
    assert.ok(!links.includes('/blog/'), 'a generic blog reload is not an article selector');
});

test('popup text is escaped without breaking legitimate quotes or ampersands', () => {
    const html = renderCalendarPosts([{
        date: '2026-05-21',
        title: 'A < B & "quoted"',
        excerpt: '<img src=x onerror=alert(1)>',
        tags: ['<script>', "A'B"],
        url: '/article/?a=1&b=2',
    }]);
    assert.ok(html.includes('A &lt; B &amp; &quot;quoted&quot;'));
    assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'));
    assert.ok(html.includes('&lt;script&gt;'));
    assert.ok(html.includes('href="/article/?a=1&amp;b=2"'));
    assert.ok(!html.includes('<img'));
    assert.equal(renderCalendarPosts([]), '');
});

test('blog exposes the sidebar target and relies on one shared calendar dataset', async () => {
    const blog = await readFile(new URL('../blog.html', import.meta.url), 'utf8');
    const layout = await readFile(new URL('../_layouts/default.html', import.meta.url), 'utf8');
    assert.equal((blog.match(/id="tag-network"/g) || []).length, 1);
    assert.equal((blog.match(/id="calendarData"/g) || []).length, 0);
    assert.equal((layout.match(/id="calendarData"/g) || []).length, 1);
});
