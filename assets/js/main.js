// 全站脚本入口 —— defer 执行,DOM 已解析
import { ParticleSystem } from './particle-system.js';
import { initializeNavigation } from './navigation.js';
import { initBackToTop } from './back-to-top.js';
import { initClickEffect } from './click-effect.js';
import { initCodeCopy } from './code-copy.js';

document.addEventListener('DOMContentLoaded', function () {
    new ParticleSystem();
    initializeNavigation();
    initBackToTop();
    initClickEffect();
    initCodeCopy(); // 全站:复制按钮可能出现在任何有代码块的页面

    // 文章页(post layout)专属模块
    if (document.querySelector('.post-content')) {
        import('./cite-tooltip.js').then(function (m) { m.initCiteTooltips(); });
        import('./image-zoom.js').then(function (m) { m.initImageZoom(); });
    }
    if (document.getElementById('blogToc')) {
        import('./blog-toc.js').then(function (m) { m.initBlogToc(); });
    }

    // 博客列表页:日历
    if (document.getElementById('calendarData')) {
        import('./calendar.js').then(function (m) { m.initCalendar(); });
    }

    // 标签网络图(当前无活跃页面含 #tagNetworkCanvas;#tagNetworkData 兜底)
    if (document.getElementById('tagNetworkCanvas') || document.getElementById('tagNetworkData')) {
        import('./tag-network.js').then(function (m) { m.initTagNetwork(); });
    }
});
