/**
 * 代码块复制功能
 * 自动为所有代码块添加复制按钮
 */

(function() {
    'use strict';
    
    // 等待DOM加载完成
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCodeCopy);
    } else {
        initCodeCopy();
    }
    
    function initCodeCopy() {
        // 查找所有代码块
        const codeBlocks = document.querySelectorAll('pre code');
        
        codeBlocks.forEach((codeBlock) => {
            // 跳过已经添加过按钮的代码块
            if (codeBlock.parentElement.querySelector('.copy-code-button')) {
                return;
            }
            
            // 创建复制按钮
            const button = createCopyButton();
            
            // 将按钮添加到pre元素
            const pre = codeBlock.parentElement;
            pre.style.position = 'relative';
            pre.appendChild(button);
            
            // 添加点击事件
            button.addEventListener('click', () => {
                copyCode(codeBlock, button);
            });
        });
    }
    
    /**
     * 创建复制按钮
     */
    function createCopyButton() {
        const button = document.createElement('button');
        button.className = 'copy-code-button';
        button.type = 'button';
        button.setAttribute('aria-label', '复制代码');
        button.innerHTML = `
            <svg class="copy-icon" viewBox="0 0 24 24" width="16" height="16">
                <path fill="currentColor" d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
            <svg class="check-icon" viewBox="0 0 24 24" width="16" height="16" style="display: none;">
                <path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
            </svg>
            <span class="copy-text">复制</span>
        `;
        return button;
    }
    
    /**
     * 复制代码到剪贴板
     */
    async function copyCode(codeBlock, button) {
        // 获取代码文本
        const code = codeBlock.textContent || codeBlock.innerText;
        
        try {
            // 使用现代 Clipboard API
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(code);
            } else {
                // 降级方案：使用传统方法
                copyToClipboardFallback(code);
            }
            
            // 显示成功反馈
            showCopySuccess(button);
        } catch (err) {
            console.error('复制失败:', err);
            showCopyError(button);
        }
    }
    
    /**
     * 降级复制方法（兼容旧浏览器）
     */
    function copyToClipboardFallback(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            textArea.remove();
        } catch (err) {
            textArea.remove();
            throw err;
        }
    }
    
    /**
     * 显示复制成功
     */
    function showCopySuccess(button) {
        const copyIcon = button.querySelector('.copy-icon');
        const checkIcon = button.querySelector('.check-icon');
        const copyText = button.querySelector('.copy-text');
        
        // 切换图标
        copyIcon.style.display = 'none';
        checkIcon.style.display = 'inline-block';
        copyText.textContent = '已复制！';
        
        // 添加成功样式
        button.classList.add('copied');
        
        // 2秒后恢复
        setTimeout(() => {
            copyIcon.style.display = 'inline-block';
            checkIcon.style.display = 'none';
            copyText.textContent = '复制';
            button.classList.remove('copied');
        }, 2000);
    }
    
    /**
     * 显示复制错误
     */
    function showCopyError(button) {
        const copyText = button.querySelector('.copy-text');
        const originalText = copyText.textContent;
        
        copyText.textContent = '复制失败';
        button.classList.add('copy-error');
        
        setTimeout(() => {
            copyText.textContent = originalText;
            button.classList.remove('copy-error');
        }, 2000);
    }
    
    // 监听动态添加的代码块
    if (window.MutationObserver) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // 元素节点
                        const codeBlocks = node.querySelectorAll ? 
                            node.querySelectorAll('pre code') : [];
                        
                        if (codeBlocks.length > 0) {
                            initCodeCopy();
                        }
                    }
                });
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
})();

