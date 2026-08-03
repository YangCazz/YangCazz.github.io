import { chromium } from 'playwright-core';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage();
for (const p of ['/2026/08/02/continual-learning-comprehensive-guide/','/2026/08/03/class-incremental-learning-in-depth/']) {
  await page.goto('http://127.0.0.1:4000' + p, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2500);
  const r = await page.evaluate(() => {
    const containers = document.querySelectorAll('.mermaid-container').length;
    const svgs = document.querySelectorAll('.mermaid-container svg').length;
    const errs = document.querySelectorAll('.mermaid-error').length;
    return { containers, svgs, errs };
  });
  console.log(`${r.containers}图容器 / ${r.svgs} SVG / ${r.errs} 错误  ${p}`);
}
await browser.close();
