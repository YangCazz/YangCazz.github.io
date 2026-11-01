// Umami Cloud loader. No account API keys belong in this public module.
export const DISABLED_KEY = 'umami.disabled';
const CLOUD_SCRIPT = 'https://cloud.umami.is/script.js';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function trackingAllowed(config, browser) {
    if (!config || config.enabled !== 'true' || config.production !== 'true' || config.trackPage === 'false') return false;
    if (!UUID.test(config.websiteId || '') || config.scriptUrl !== CLOUD_SCRIPT) return false;
    if (config.hostname !== 'yangcazz.github.io' || browser.location.hostname !== config.hostname || browser.location.protocol !== 'https:') return false;
    const dntEnabled = value => value === 1 || value === '1' || value === 'yes';
    if (browser.navigator.webdriver || browser.navigator.globalPrivacyControl === true ||
        [browser.navigator.doNotTrack, browser.doNotTrack, browser.navigator.msDoNotTrack].some(dntEnabled)) return false;
    try {
        // Treat inaccessible storage conservatively: do not accidentally ignore an opt-out.
        if (browser.localStorage.getItem(DISABLED_KEY)) return false;
    } catch { return false; }
    return true;
}

export function cleanPayload(type, payload, config, browser) {
    if (!trackingAllowed(config, browser) || type !== 'event' || !payload || typeof payload !== 'object') return false;
    // This integration intentionally supports page views only, not identities or custom events.
    if (payload.name || payload.data || payload.id) return false;
    const clean = { ...payload };
    for (const key of ['url', 'referrer']) {
        if (typeof clean[key] !== 'string' || !clean[key]) continue;
        try {
            const url = new URL(clean[key], browser.location.origin);
            if (url.protocol !== 'https:' && url.protocol !== 'http:') return false;
            clean[key] = key === 'url' || url.origin === browser.location.origin
                ? url.pathname : url.origin + url.pathname;
        } catch { return false; }
    }
    return clean;
}

export function loadTracker(config, browser, document) {
    if (!trackingAllowed(config, browser) || document.getElementById('umami-tracker')) return false;
    browser.blogAnalyticsBeforeSend = (type, payload) => cleanPayload(type, payload, config, browser);
    const script = document.createElement('script');
    script.id = 'umami-tracker';
    script.src = CLOUD_SCRIPT;
    script.async = true;
    script.setAttribute('data-website-id', config.websiteId);
    script.setAttribute('data-domains', config.hostname);
    script.setAttribute('data-do-not-track', 'true');
    script.setAttribute('data-exclude-search', 'true');
    script.setAttribute('data-exclude-hash', 'true');
    script.setAttribute('data-before-send', 'blogAnalyticsBeforeSend');
    document.head.appendChild(script);
    return true;
}

export function setOptOut(disabled, browser) {
    try {
        if (disabled) browser.localStorage.setItem(DISABLED_KEY, '1');
        else browser.localStorage.removeItem(DISABLED_KEY);
        return true;
    } catch { return false; }
}

export function initAnalytics(browser, document) {
    const config = document.getElementById('blog-analytics-config')?.dataset;
    const status = document.getElementById('analytics-preference-status');
    const refreshStatus = () => {
        if (!status) return;
        try {
            status.textContent = browser.localStorage.getItem(DISABLED_KEY)
                ? '当前浏览器已排除访问统计。清除站点数据后，需要重新设置。'
                : '当前浏览器未设置排除。只有正式站点启用统计且浏览器未发出隐私信号时，才会采集访问。';
        } catch { status.textContent = '浏览器不允许访问本地存储，本次访问不会加载统计脚本。'; }
    };
    for (const [id, disabled] of [['analytics-disable', true], ['analytics-enable', false]]) {
        document.getElementById(id)?.addEventListener('click', () => {
            if (!setOptOut(disabled, browser)) {
                if (status) status.textContent = '设置未保存：浏览器阻止了本地存储。请检查浏览器设置。';
                return;
            }
            refreshStatus();
        });
    }
    refreshStatus();
    loadTracker(config, browser, document);
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') initAnalytics(window, document);
