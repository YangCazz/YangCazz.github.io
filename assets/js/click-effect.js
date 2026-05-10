// 鼠标点击波纹效果
document.addEventListener('click', function(e) {
    const clickEffect = document.createElement('div');
    clickEffect.className = 'click-effect';
    clickEffect.style.left = e.clientX + 'px';
    clickEffect.style.top = e.clientY + 'px';
    document.body.appendChild(clickEffect);

    setTimeout(function() {
        if (document.body.contains(clickEffect)) {
            document.body.removeChild(clickEffect);
        }
    }, 800);
});
