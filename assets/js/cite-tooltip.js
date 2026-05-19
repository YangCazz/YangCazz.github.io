// 引用角标悬浮预览
(function () {
    function initCiteTooltips() {
        var refList = document.querySelector('.post-content ol.references') ||
                      document.querySelector('ol.references');
        if (!refList) return;

        var refs = {};
        var items = refList.querySelectorAll('li');
        items.forEach(function (li, i) {
            var text = li.textContent
                .replace(/\s+/g, ' ')
                .trim();
            if (text.length > 200) {
                text = text.substring(0, 197) + '...';
            }
            refs[i + 1] = text;
        });

        if (Object.keys(refs).length === 0) return;

        var tooltip = document.createElement('div');
        tooltip.className = 'cite-tooltip';
        tooltip.innerHTML = '<span class="cite-tooltip-ref"></span><span class="cite-tooltip-body"></span>';
        document.body.appendChild(tooltip);

        var refSpan = tooltip.querySelector('.cite-tooltip-ref');
        var bodySpan = tooltip.querySelector('.cite-tooltip-body');
        var currentCite = null;
        var hideTimer = null;

        function positionTooltip(cite) {
            var citeRect = cite.getBoundingClientRect();
            var th = tooltip.offsetHeight || 60;
            var tw = tooltip.offsetWidth || 200;
            var gap = 8;
            var vw = window.innerWidth;
            var scrollY = window.pageYOffset;
            var scrollX = window.pageXOffset;

            // 优先放上方
            var top = citeRect.top - th - gap + scrollY;
            if (top < scrollY + 10) {
                // 空间不够，放下方
                top = citeRect.bottom + gap + scrollY;
            }

            // 水平居中于角标
            var left = citeRect.left + citeRect.width / 2 - tw / 2 + scrollX;
            left = Math.max(8, Math.min(left, vw - tw - 8 + scrollX));

            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }

        function show(cite, num) {
            clearTimeout(hideTimer);
            if (currentCite === cite) return;
            currentCite = cite;

            var refText = refs[num];
            if (!refText) return;

            refSpan.textContent = '[' + num + ']';
            bodySpan.textContent = refText;
            tooltip.classList.add('show');
            positionTooltip(cite);
        }

        function hide() {
            hideTimer = setTimeout(function () {
                tooltip.classList.remove('show');
                currentCite = null;
            }, 150);
        }

        // 委托在 post-content 上
        var container = document.querySelector('.post-content');
        if (!container) return;

        container.addEventListener('mouseover', function (e) {
            var cite = e.target.closest('cite');
            if (!cite) return;
            var m = cite.textContent.match(/\[(\d+)\]/);
            if (m) show(cite, parseInt(m[1], 10));
        });

        container.addEventListener('mouseout', function (e) {
            var cite = e.target.closest('cite');
            if (cite) hide();
        });

        tooltip.addEventListener('mouseenter', function () {
            clearTimeout(hideTimer);
        });
        tooltip.addEventListener('mouseleave', function () {
            tooltip.classList.remove('show');
            currentCite = null;
        });

        // 滚动时关闭
        window.addEventListener('scroll', function () {
            if (currentCite) {
                tooltip.classList.remove('show');
                currentCite = null;
            }
        }, { passive: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCiteTooltips);
    } else {
        initCiteTooltips();
    }
})();
