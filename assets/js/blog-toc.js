// 博客目录 + 标题自动编号
function initBlogToc() {
    const tocContainer = document.getElementById('blogToc');
    if (!tocContainer) return;

    const headings = document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6');
    if (headings.length === 0) {
        tocContainer.innerHTML = '<p style="color: var(--secondary-color); font-size: 0.9rem; text-align: center; margin: 1rem 0;">暂无目录</p>';
        return;
    }

    // ---- 自动编号 ----
    let h2num = 0;
    let h3num = 0;
    const counters = new Map(); // heading element → "X" or "X.Y"

    headings.forEach((heading) => {
        const tag = heading.tagName.toLowerCase();
        let prefix = '';
        if (tag === 'h2') {
            h2num++;
            h3num = 0;
            prefix = h2num + '. ';
        } else if (tag === 'h3') {
            h3num++;
            prefix = h2num + '.' + h3num + ' ';
        }

        if (prefix) {
            counters.set(heading, prefix);
            // 只在还没有添加过编号时插入
            if (!heading.querySelector('.heading-num')) {
                const span = document.createElement('span');
                span.className = 'heading-num';
                span.textContent = prefix;
                heading.insertBefore(span, heading.firstChild);
            }
        }
    });

    // ---- 构建目录 ----
    let tocHtml = '';
    headings.forEach((heading, index) => {
        const level = parseInt(heading.tagName.charAt(1));
        const id = 'heading-' + index;
        heading.id = id;

        // 取得已编号的文本
        const numSpan = heading.querySelector('.heading-num');
        const bodyText = Array.from(heading.childNodes)
            .filter(n => n !== numSpan)
            .map(n => n.textContent)
            .join('')
            .trim();
        const displayText = numSpan ? numSpan.textContent + bodyText : bodyText;

        tocHtml += '<a href="#' + id + '" class="toc-item ' + tagName(level) + '" data-level="' + level + '">' + escapeHtml(displayText) + '</a>';
    });

    tocContainer.innerHTML = tocHtml;

    // ---- 移动端浮动目录 ----
    const mobileTocContainer = document.getElementById('mobileToc');
    const mobilePanel = document.getElementById('mobileTocPanel');
    const mobileOverlay = document.getElementById('mobileTocOverlay');
    const mobileToggle = document.getElementById('mobileTocToggle');
    const mobileClose = document.getElementById('mobileTocClose');

    if (mobileTocContainer) {
        mobileTocContainer.innerHTML = tocHtml;

        let mobileTocItems = mobileTocContainer.querySelectorAll('.toc-item');

        function openMobileToc() {
            mobilePanel.classList.add('open');
            mobileOverlay.classList.add('open');
            mobileToggle.style.opacity = '0';
            document.body.style.overflow = 'hidden';
        }

        function closeMobileToc() {
            mobilePanel.classList.remove('open');
            mobileOverlay.classList.remove('open');
            mobileToggle.style.opacity = '1';
            document.body.style.overflow = '';
        }

        mobileToggle.addEventListener('click', openMobileToc);
        mobileOverlay.addEventListener('click', closeMobileToc);
        mobileClose.addEventListener('click', closeMobileToc);

        // 移动端目录项点击后关闭面板
        mobileTocItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = item.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
                closeMobileToc();
            });
        });
    }

    // ---- 滚动监听 ----
    const tocItems = tocContainer.querySelectorAll('.toc-item');
    const mobileTocItems = mobileTocContainer ? mobileTocContainer.querySelectorAll('.toc-item') : [];

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // 桌面端目录
                tocItems.forEach(item => item.classList.remove('active'));
                const desktopItem = tocContainer.querySelector('a[href="#' + entry.target.id + '"]');
                if (desktopItem) {
                    desktopItem.classList.add('active');
                    scrollToActiveItem(desktopItem, tocContainer);
                }
                // 移动端目录
                if (mobileTocContainer) {
                    mobileTocItems.forEach(item => item.classList.remove('active'));
                    const mobileItem = mobileTocContainer.querySelector('a[href="#' + entry.target.id + '"]');
                    if (mobileItem) {
                        mobileItem.classList.add('active');
                        scrollToActiveItem(mobileItem, mobileTocContainer);
                    }
                }
            }
        });
    }, {
        rootMargin: '-20% 0px -70% 0px'
    });

    headings.forEach(heading => observer.observe(heading));

    tocItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    function scrollToActiveItem(activeItem, container) {
        const containerRect = container.getBoundingClientRect();
        const itemRect = activeItem.getBoundingClientRect();
        if (itemRect.top >= containerRect.top && itemRect.bottom <= containerRect.bottom) return;
        container.scrollTo({
            top: activeItem.offsetTop - (container.clientHeight / 2) + (activeItem.offsetHeight / 2),
            behavior: 'smooth'
        });
    }
}

function tagName(level) { return 'h' + level; }

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}
