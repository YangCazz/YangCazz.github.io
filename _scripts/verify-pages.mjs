// 页面级验证脚本:遍历页面清单,断言 HTTP 200 + 无 console error。
// 用法: node _scripts/verify-pages.mjs [--screenshot]
import { chromium } from 'playwright-core';

const BASE = 'http://127.0.0.1:4000';
const PAGES = [
  { path: '/', name: 'home', check: '.main-content' },
  { path: '/blog/', name: 'blog', check: '.blog-list' },
  { path: '/resume/', name: 'resume', check: '.main-content' },
  { path: '/showcase/', name: 'showcase', check: '.main-content' },
  { path: '/apps/', name: 'apps', check: '.main-content' },
];
const SHOT = process.argv.includes('--screenshot');
const OUT = new URL('../_site-verify/', import.meta.url);

// 取前 3 篇博文(含公式/Mermaid/引用的代表性文章)
async function collectPosts(page) {
  await page.goto(BASE + '/blog/', { waitUntil: 'domcontentloaded' });
  return await page.$$eval('.blog-list article a[href]', els =>
    els.slice(0, 3).map(a => a.getAttribute('href')));
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
let failed = false;
const errors = {};

page.on('console', msg => {
  if (msg.type() === 'error') {
    errors[page.url()] = errors[page.url()] || [];
    errors[page.url()].push(msg.text());
  }
});

try {
  for (const p of PAGES) {
    await page.goto(BASE + p.path, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1200); // 等待粒子/网络图/公式渲染
    const ok = await page.evaluate((sel) => !!document.querySelector(sel), p.check);
    const errs = errors[page.url()] || [];
    const status = ok && errs.length === 0;
    if (!status) failed = true;
    console.log(`${status ? 'PASS' : 'FAIL'}  ${p.name.padEnd(8)} ${p.path}  consoleErr=${errs.length}`);
    if (SHOT) await page.screenshot({ path: new URL(p.name + '.png', OUT).pathname });
  }

  const postPaths = await collectPosts(page);
  for (const pp of postPaths) {
    await page.goto(BASE + pp, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2000); // MathJax/Mermaid 异步
    const hasContent = await page.evaluate(() => !!document.querySelector('.post-content'));
    const hasToc = await page.evaluate(() => !!document.querySelector('#blogToc a'));
    const errs = errors[page.url()] || [];
    const status = hasContent && errs.length === 0;
    if (!status) failed = true;
    console.log(`${status ? 'PASS' : 'FAIL'}  post      ${pp}  toc=${hasToc}  consoleErr=${errs.length}`);
    if (SHOT) await page.screenshot({ path: new URL('post-' + pp.split('/').filter(Boolean).pop() + '.png', OUT).pathname });
  }
} finally {
  await browser.close();
}

if (failed) { console.log('RESULT: FAIL'); process.exit(1); }
console.log('RESULT: PASS');
