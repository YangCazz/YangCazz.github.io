/**
 * 访问量计数器 — 用 <object> 加载 badge SVG，删除内部渐变层和阴影文字，
 * 只保留纯数字文本，再通过 CSS filter 适配深浅背景。
 */
(function () {
  document.querySelectorAll('[data-badge]').forEach(function (el) {
    var obj = document.createElement('object');
    obj.type = 'image/svg+xml';
    obj.data = el.dataset.badge;
    obj.classList.add('visitor-badge-svg');

    obj.addEventListener('load', function () {
      var svg = obj.contentDocument;
      if (!svg) return;

      // 移除渐变定义
      var grad = svg.getElementById('smooth');
      if (grad) grad.remove();

      // 移除引用该渐变的 rect
      var overlay = svg.querySelector('rect[fill*="smooth"]');
      if (overlay) overlay.remove();

      // 移除阴影文字 (fill-opacity 的 text)
      svg.querySelectorAll('text[fill-opacity]').forEach(function (t) {
        t.remove();
      });

      // 移除用作 clipPath 的 rect 上的 fill 属性，避免某些浏览器渲染
      var clipRect = svg.querySelector('clipPath rect');
      if (clipRect) clipRect.removeAttribute('fill');
    });

    el.appendChild(obj);
  });
})();
