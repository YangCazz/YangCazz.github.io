// 博客目录功能
function initBlogToc() {
    const tocContainer = document.getElementById('blogToc');
    if (!tocContainer) return;

    const headings = document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6');
    if (headings.length === 0) {
        tocContainer.innerHTML = '<p style="color: var(--secondary-color); font-size: 0.9rem; text-align: center; margin: 1rem 0;">暂无目录</p>';
        return;
    }

    let tocHtml = '';
    headings.forEach((heading, index) => {
        const level = parseInt(heading.tagName.charAt(1));
        const text = heading.textContent.trim();
        const id = `heading-${index}`;

        heading.id = id;
        tocHtml += `<a href="#${id}" class="toc-item h${level}" data-level="${level}">${text}</a>`;
    });

    tocContainer.innerHTML = tocHtml;

    const tocItems = tocContainer.querySelectorAll('.toc-item');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                tocItems.forEach(item => item.classList.remove('active'));

                const targetId = entry.target.id;
                const correspondingTocItem = tocContainer.querySelector(`a[href="#${targetId}"]`);
                if (correspondingTocItem) {
                    correspondingTocItem.classList.add('active');
                    scrollToActiveItem(correspondingTocItem, tocContainer);
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
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    function scrollToActiveItem(activeItem, container) {
        const containerRect = container.getBoundingClientRect();
        const itemRect = activeItem.getBoundingClientRect();

        const isVisible = itemRect.top >= containerRect.top &&
                         itemRect.bottom <= containerRect.bottom;

        if (!isVisible) {
            const scrollTop = container.scrollTop;
            const itemOffsetTop = activeItem.offsetTop;
            const containerHeight = container.clientHeight;
            const itemHeight = activeItem.offsetHeight;

            const targetScrollTop = itemOffsetTop - (containerHeight / 2) + (itemHeight / 2);

            container.scrollTo({
                top: targetScrollTop,
                behavior: 'smooth'
            });
        }
    }
}
