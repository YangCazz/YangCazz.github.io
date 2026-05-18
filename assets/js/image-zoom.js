/**
 * Image zoom — click any image in .post-content to view in a full-viewport overlay.
 * The overlay is appended directly to <body> to avoid CSS containing-block issues
 * caused by parent elements with backdrop-filter.
 */
(function () {
  if (typeof document === 'undefined') return;

  var overlay = null;

  function close() {
    if (!overlay) return;
    overlay.remove();
    overlay = null;
    document.body.style.overflow = '';
  }

  document.addEventListener('click', function (e) {
    // Close on overlay background click
    if (overlay && e.target === overlay) {
      close();
      return;
    }

    // Ignore clicks on already-zoomed image
    if (overlay && e.target.closest('.image-zoom-overlay')) return;

    var img = e.target.closest('.post-content img');
    if (!img) return;

    // Don't zoom images inside links
    if (img.closest('a')) return;

    // Already open — close first then re-open below
    if (overlay) close();

    // Create overlay appended to <body> (bypasses backdrop-filter containing block)
    overlay = document.createElement('div');
    overlay.className = 'image-zoom-overlay';

    var cloned = img.cloneNode(true);
    cloned.removeAttribute('class');
    overlay.appendChild(cloned);

    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden';
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') close();
  });
})();
