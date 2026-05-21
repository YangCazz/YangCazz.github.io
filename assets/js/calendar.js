// 日历面板功能
function initCalendar() {
    const calendarTitle = document.getElementById('calendarTitle');
    const calendarDays = document.getElementById('calendarDays');
    const prevMonthBtn = document.getElementById('prevMonth');
    const nextMonthBtn = document.getElementById('nextMonth');
    const dataScript = document.getElementById('calendarData');

    if (!calendarTitle || !calendarDays || !prevMonthBtn || !nextMonthBtn || !dataScript) {
        return;
    }

    let currentDate = new Date();
    const blogPosts = JSON.parse(dataScript.textContent);

    // 创建工具提示元素
    function createTooltip() {
        const tooltip = document.createElement('div');
        tooltip.className = 'calendar-tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);
        return tooltip;
    }

    const tooltip = createTooltip();

    // 显示工具提示
    function showTooltip(event, posts) {
        if (posts.length === 0) return;

        const post = posts[0];
        const multiplePosts = posts.length > 1;

        tooltip.innerHTML = `
            <div class="tooltip-header">
                <span class="tooltip-date">${event.target.dataset.date}</span>
                ${multiplePosts ? `<span class="tooltip-count">+${posts.length - 1}篇</span>` : ''}
            </div>
            <div class="tooltip-content">
                <h4 class="tooltip-title">${post.title}</h4>
                <p class="tooltip-excerpt">${post.excerpt}</p>
                <div class="tooltip-tags">
                    ${post.tags.slice(0, 3).map(tag => `<span class="tooltip-tag">${tag}</span>`).join('')}
                </div>
            </div>
            <div class="tooltip-footer">
                <a href="${post.url}" class="tooltip-link">阅读全文 →</a>
            </div>
        `;

        tooltip.style.display = 'block';

        const rect = event.target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();

        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
        let top = rect.bottom + 10;

        if (left < 10) left = 10;
        if (left + tooltipRect.width > window.innerWidth - 10) {
            left = window.innerWidth - tooltipRect.width - 10;
        }
        if (top + tooltipRect.height > window.innerHeight - 10) {
            top = rect.top - tooltipRect.height - 10;
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    // 隐藏工具提示
    function hideTooltip() {
        tooltip.style.display = 'none';
    }

    // 获取某日期的博客文章（字符串比较，避免 new Date() 的 UTC/local 歧义）
    function getPostsForDate(date) {
        const yyyy = date.getFullYear();
        const mm = String(date.getMonth() + 1).padStart(2, '0');
        const dd = String(date.getDate()).padStart(2, '0');
        const dateStr = yyyy + '-' + mm + '-' + dd;
        return blogPosts.filter(post => post.date === dateStr);
    }

    function renderCalendar() {
        const year = currentDate.getFullYear();
        const month = currentDate.getMonth();

        const monthNames = ['一月', '二月', '三月', '四月', '五月', '六月',
                          '七月', '八月', '九月', '十月', '十一月', '十二月'];
        calendarTitle.textContent = `${year}年 ${monthNames[month]}`;

        calendarDays.innerHTML = '';

        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
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

            const postsForDate = getPostsForDate(date);
            if (postsForDate.length > 0) {
                dayElement.classList.add('has-posts');

                dayElement.addEventListener('mouseenter', (e) => {
                    showTooltip(e, postsForDate);
                });

                dayElement.addEventListener('mouseleave', hideTooltip);

                dayElement.addEventListener('click', () => {
                    if (postsForDate.length === 1) {
                        window.location.href = postsForDate[0].url;
                    } else {
                        window.location.href = '/blog/';
                    }
                });
            }

            calendarDays.appendChild(dayElement);
        }
    }

    prevMonthBtn.addEventListener('click', () => {
        currentDate.setMonth(currentDate.getMonth() - 1);
        renderCalendar();
    });

    nextMonthBtn.addEventListener('click', () => {
        currentDate.setMonth(currentDate.getMonth() + 1);
        renderCalendar();
    });

    renderCalendar();
}
