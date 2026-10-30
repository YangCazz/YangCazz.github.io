// 仅保存月份首日，避免 29/30/31 日跨入短月时发生 Date.setMonth 溢出。
export function shiftCalendarMonth(date, offset) {
    return new Date(date.getFullYear(), date.getMonth() + offset, 1);
}

export function getCalendarPostsForDate(posts, date) {
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    const dateStr = yyyy + '-' + mm + '-' + dd;
    return posts.filter(post => post.date === dateStr);
}

// 优先完整显示；空间不足时缩短滚动面板，而不是覆盖仍可再次点击的日期。
export function positionCalendarPopup(anchor, popup, viewport, gap = 10) {
    const availableAbove = Math.max(0, anchor.top - 2 * gap);
    const availableBelow = Math.max(0, viewport.height - anchor.bottom - 2 * gap);
    const left = Math.max(gap, Math.min(
        anchor.left + anchor.width / 2 - popup.width / 2,
        viewport.width - popup.width - gap,
    ));
    if (popup.height <= availableBelow) {
        return { left, top: anchor.bottom + gap, maxHeight: popup.height };
    }
    if (popup.height <= availableAbove) {
        return { left, top: anchor.top - popup.height - gap, maxHeight: popup.height };
    }

    const sideTop = Math.max(gap, Math.min(anchor.top, viewport.height - popup.height - gap));
    if (anchor.right + popup.width + 2 * gap <= viewport.width) {
        return { left: anchor.right + gap, top: sideTop, maxHeight: popup.height };
    }
    if (anchor.left >= popup.width + 2 * gap) {
        return { left: anchor.left - popup.width - gap, top: sideTop, maxHeight: popup.height };
    }

    const placeBelow = availableBelow >= availableAbove;
    const maxHeight = Math.min(popup.height, placeBelow ? availableBelow : availableAbove);
    return {
        left,
        top: placeBelow ? anchor.bottom + gap : anchor.top - gap - maxHeight,
        maxHeight,
    };
}

function escapeHtml(value) {
    const escapes = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return String(value ?? '').replace(/[&<>"']/g, character => escapes[character]);
}

// 每篇文章都保留独立入口；不要把同日多文折叠成首篇 + 一个无入口的数量。
export function renderCalendarPosts(posts) {
    return posts.map(post => `
        <article class="tooltip-post">
            <div class="tooltip-content">
                <h4 class="tooltip-title">${escapeHtml(post.title)}</h4>
                <p class="tooltip-excerpt">${escapeHtml(post.excerpt)}</p>
                <div class="tooltip-tags">
                    ${(post.tags || []).slice(0, 3).map(tag => `<span class="tooltip-tag">${escapeHtml(tag)}</span>`).join('')}
                </div>
            </div>
            <div class="tooltip-footer">
                <a href="${escapeHtml(post.url)}" class="tooltip-link" aria-label="阅读全文：${escapeHtml(post.title)}">阅读全文 →</a>
            </div>
        </article>
    `).join('');
}

// 日历面板功能
export function initCalendar() {
    const calendarTitle = document.getElementById('calendarTitle');
    const calendarDays = document.getElementById('calendarDays');
    const prevMonthBtn = document.getElementById('prevMonth');
    const nextMonthBtn = document.getElementById('nextMonth');
    const dataScript = document.getElementById('calendarData');

    if (!calendarTitle || !calendarDays || !prevMonthBtn || !nextMonthBtn || !dataScript) {
        return;
    }

    let currentDate = shiftCalendarMonth(new Date(), 0);
    const blogPosts = JSON.parse(dataScript.textContent);
    let activeDay = null;

    // 创建工具提示元素
    function createTooltip() {
        const tooltip = document.createElement('div');
        tooltip.className = 'calendar-tooltip';
        tooltip.id = 'calendarPostPopup';
        tooltip.setAttribute('role', 'region');
        tooltip.setAttribute('aria-label', '当日文章');
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);
        return tooltip;
    }

    const tooltip = createTooltip();

    // 显示工具提示
    function showTooltip(dayElement, posts) {
        if (posts.length === 0) return;
        hideTooltip();
        activeDay = dayElement;
        activeDay.setAttribute('aria-expanded', 'true');

        tooltip.innerHTML = `
            <div class="tooltip-header">
                <span class="tooltip-date">${escapeHtml(dayElement.dataset.date)}</span>
                <span class="tooltip-count">${posts.length}篇</span>
                <button type="button" class="calendar-tooltip-close" aria-label="关闭日历文章">×</button>
            </div>
            ${renderCalendarPosts(posts)}
        `;

        tooltip.dataset.activeDate = dayElement.dataset.date;
        tooltip.style.maxHeight = '';
        tooltip.style.display = 'block';
        tooltip.scrollTop = 0;

        const rect = dayElement.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();

        const position = positionCalendarPopup(rect, tooltipRect, {
            width: window.innerWidth,
            height: window.innerHeight,
        });
        tooltip.style.maxHeight = position.maxHeight + 'px';
        tooltip.style.left = position.left + 'px';
        tooltip.style.top = position.top + 'px';
    }

    // 隐藏工具提示
    function hideTooltip(restoreFocus = false) {
        tooltip.style.display = 'none';
        tooltip.dataset.activeDate = '';
        if (activeDay) {
            activeDay.setAttribute('aria-expanded', 'false');
            if (restoreFocus) activeDay.focus();
            activeDay = null;
        }
    }

    // 点击空白处关闭 tooltip
    document.addEventListener('click', function(e) {
        if (!tooltip.contains(e.target) && !e.target.closest('.calendar-day.has-posts')) {
            hideTooltip();
        }
    });
    tooltip.addEventListener('click', function(e) {
        if (e.target.closest('.calendar-tooltip-close')) hideTooltip(true);
    });
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') hideTooltip(true);
    });
    window.addEventListener('resize', () => hideTooltip());

    function renderCalendar() {
        const year = currentDate.getFullYear();
        const month = currentDate.getMonth();

        const monthNames = ['一月', '二月', '三月', '四月', '五月', '六月',
                          '七月', '八月', '九月', '十月', '十一月', '十二月'];
        calendarTitle.textContent = `${year}年 ${monthNames[month]}`;

        calendarDays.innerHTML = '';

        const firstDay = new Date(year, month, 1);
        const startDate = new Date(firstDay);
        startDate.setDate(startDate.getDate() - firstDay.getDay());

        for (let i = 0; i < 42; i++) {
            const date = new Date(startDate);
            date.setDate(startDate.getDate() + i);

            const dayElement = document.createElement('div');
            dayElement.className = 'calendar-day';
            dayElement.textContent = date.getDate();
            const yyyy = date.getFullYear();
            const mm = String(date.getMonth() + 1).padStart(2, '0');
            const dd = String(date.getDate()).padStart(2, '0');
            dayElement.dataset.date = yyyy + '-' + mm + '-' + dd;

            if (date.getMonth() !== month) {
                dayElement.classList.add('other-month');
            }

            const today = new Date();
            if (date.toDateString() === today.toDateString()) {
                dayElement.classList.add('today');
            }

            const postsForDate = getCalendarPostsForDate(blogPosts, date);
            if (postsForDate.length > 0) {
                dayElement.classList.add('has-posts');
                dayElement.setAttribute('role', 'button');
                dayElement.tabIndex = 0;
                dayElement.setAttribute('aria-label', `${dayElement.dataset.date}，${postsForDate.length}篇文章`);
                dayElement.setAttribute('aria-controls', tooltip.id);
                dayElement.setAttribute('aria-expanded', 'false');

                dayElement.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    // 单篇保留再次点击直达的行为；多篇再次点击只收起，不重载整页。
                    if (tooltip.style.display === 'block' && tooltip.dataset.activeDate === dayElement.dataset.date) {
                        if (postsForDate.length === 1) {
                            window.location.href = postsForDate[0].url;
                        } else {
                            hideTooltip();
                        }
                    } else {
                        showTooltip(dayElement, postsForDate);
                    }
                });
                dayElement.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        dayElement.click();
                    }
                });
            }

            calendarDays.appendChild(dayElement);
        }
    }

    prevMonthBtn.addEventListener('click', () => {
        hideTooltip();
        currentDate = shiftCalendarMonth(currentDate, -1);
        renderCalendar();
    });

    nextMonthBtn.addEventListener('click', () => {
        hideTooltip();
        currentDate = shiftCalendarMonth(currentDate, 1);
        renderCalendar();
    });

    renderCalendar();
}
