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
        // 处理 .highlight 容器（Jekyll Rouge 输出的代码块）
        // 匹配 .highlight 容器，但排除：
        //   1) pre.highlight（Rouge 给 pre 也加了这个类）
        //   2) 内层还嵌套了另一个 .highlight 的外层容器（GitHub Pages 旧版 kramdown 会双重包裹）
        const highlights = document.querySelectorAll('.highlight');

        highlights.forEach((highlight) => {
            // 跳过 pre 元素
            if (highlight.tagName === 'PRE') return;
            // 如果这个 .highlight 包裹的是另一个 div.highlight（非 pre），
            // 说明是 GitHub Pages 旧版 kramdown 的双层外壳，跳过外层
            if (highlight.querySelector('div.highlight')) return;
            // 跳过已处理过的
            if (highlight.querySelector('.code-header')) return;

            // 找到 pre > code
            const codeBlock = highlight.querySelector('pre code');
            if (!codeBlock) return;

            // 注入代码头部栏
            const codeHeader = createCodeHeader(highlight);
            highlight.insertBefore(codeHeader, highlight.firstChild);

            // 在头部栏添加复制按钮
            const actions = codeHeader.querySelector('.code-header-actions');
            const button = createCopyButton();
            actions.appendChild(button);

            button.addEventListener('click', () => {
                copyCode(codeBlock, button);
            });
        });

        // 处理散落的 pre code（不在 div.highlight 内的，作为兜底）
        document.querySelectorAll('pre code').forEach((codeBlock) => {
            const pre = codeBlock.parentElement;
            // 如果 pre 在 div.highlight 内则跳过
            if (pre.closest('div.highlight')) return;
            if (pre.querySelector('.copy-code-button')) return;

            pre.style.position = 'relative';
            const button = createCopyButton();
            pre.appendChild(button);
            button.addEventListener('click', () => {
                copyCode(codeBlock, button);
            });
        });
    }

    /**
     * 从父级 div 的 class 中提取语言名
     */
    function extractLanguage(highlight) {
        // 从 .highlight 本身查找 language-* 类
        const classes = highlight.className.split(/\s+/);
        for (const cls of classes) {
            if (cls.startsWith('language-')) {
                return cls.replace('language-', '');
            }
        }
        // 也可能在外层 .highlighter-rouge 上
        const rouge = highlight.closest('.highlighter-rouge');
        if (rouge) {
            for (const cls of rouge.className.split(/\s+/)) {
                if (cls.startsWith('language-')) {
                    return cls.replace('language-', '');
                }
            }
        }
        return '';
    }

    /**
     * 创建代码头部栏（macOS 风格圆点 + 语言标签）
     */
    function createCodeHeader(highlight) {
        const header = document.createElement('div');
        header.className = 'code-header';

        const lang = extractLanguage(highlight);
        const langLabel = getLangLabel(lang);

        const left = document.createElement('div');
        left.className = 'code-header-left';
        left.innerHTML = `
            <span class="code-dots">
                <span class="code-dot code-dot--red"></span>
                <span class="code-dot code-dot--yellow"></span>
                <span class="code-dot code-dot--green"></span>
            </span>
            <span class="code-lang">${langLabel || lang || 'code'}</span>
        `;

        const actions = document.createElement('div');
        actions.className = 'code-header-actions';

        header.appendChild(left);
        header.appendChild(actions);
        return header;
    }

    function getLangLabel(lang) {
        const map = {
            'python': 'Python',
            'js': 'JavaScript',
            'javascript': 'JavaScript',
            'ts': 'TypeScript',
            'typescript': 'TypeScript',
            'html': 'HTML',
            'css': 'CSS',
            'scss': 'SCSS',
            'bash': 'Bash',
            'shell': 'Shell',
            'sh': 'Shell',
            'json': 'JSON',
            'yaml': 'YAML',
            'yml': 'YAML',
            'markdown': 'Markdown',
            'md': 'Markdown',
            'ruby': 'Ruby',
            'rb': 'Ruby',
            'c': 'C',
            'cpp': 'C++',
            'java': 'Java',
            'go': 'Go',
            'rust': 'Rust',
            'sql': 'SQL',
            'xml': 'XML',
            'dockerfile': 'Dockerfile',
            'makefile': 'Makefile',
            'plaintext': 'Plain Text',
        };
        return map[lang.toLowerCase()] || '';
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

