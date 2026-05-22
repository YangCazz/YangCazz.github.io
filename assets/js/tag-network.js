// 关键词网络图
function initTagNetwork() {
    const canvas = document.getElementById('tagNetworkCanvas');
    const tooltip = document.getElementById('tagTooltip');
    const dataScript = document.getElementById('tagNetworkData');

    if (!canvas || !dataScript) return;

    const ctx = canvas.getContext('2d');
    const data = JSON.parse(dataScript.textContent);

    let cssWidth, cssHeight;
    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        cssWidth = rect.width;
        cssHeight = rect.height;
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const maxCount = Math.max(...data.tags.map(t => t.count));

    function getNormalizedRadius(count, maxCount) {
        const isMobile = cssWidth < 500;
        const refSize = isMobile ? 720 : 500;
        const scale = Math.min(cssWidth, cssHeight) / refSize;
        const rawMin = 18 * scale;
        const rawMax = 55 * scale;
        // Mobile hard caps: max 14px radius, min 5px radius
        const minRadius = isMobile ? Math.max(rawMin, 5) : rawMin;
        const maxRadius = isMobile ? Math.min(rawMax, 14) : rawMax;
        const growthThreshold = 10;

        if (count <= growthThreshold) {
            const ratio = count / growthThreshold;
            return minRadius + (maxRadius - minRadius) * 0.7 * ratio;
        } else {
            const baseSize = minRadius + (maxRadius - minRadius) * 0.7;
            const extraGrowth = Math.log(1 + (count - growthThreshold)) * 3 * scale;
            return Math.min(maxRadius, baseSize + extraGrowth);
        }
    }

    function getColorByCount(count) {
        if (count === 1) {
            return { r: 96, g: 165, b: 250 };
        } else if (count <= 3) {
            return { r: 59, g: 130, b: 246 };
        } else if (count <= 6) {
            return { r: 37, g: 99, b: 235 };
        } else if (count <= 9) {
            return { r: 124, g: 58, b: 237 };
        } else {
            const intensity = Math.min((count - 10) / 5, 1);
            const r = 124 + (220 - 124) * intensity;
            const g = 58 + (38 - 58) * intensity;
            const b = 237 + (38 - 237) * intensity;
            return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
        }
    }

    const nodes = data.tags.map(tag => {
        const color = getColorByCount(tag.count);
        return {
            name: tag.name,
            count: tag.count,
            posts: tag.posts,
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            radius: getNormalizedRadius(tag.count, maxCount),
            color: color
        };
    });

    // 圆形初始分布：大节点靠近中心，小节点靠外
    function initCircularLayout() {
        const cx = cssWidth / 2;
        const cy = cssHeight / 2;
        const sortedByCount = [...nodes].sort((a, b) => b.count - a.count);
        const angleStep = (2 * Math.PI) / nodes.length;

        // Spread nodes to fill available space
        // On wide screens: circular within the shorter dimension
        // On narrow screens: elliptical, use more horizontal space
        const outerRadiusX = Math.min(cx, cy) * 0.85;
        const outerRadiusY = Math.min(cx, cy) * 0.75;

        sortedByCount.forEach((node, i) => {
            const angle = i * angleStep + (Math.random() - 0.5) * 0.3;
            // 按排名比例缩放半径：排名越前（count 越大）越靠近中心
            const rankRatio = i / Math.max(1, nodes.length - 1);
            const r = (0.35 + rankRatio * 0.65);
            node.x = cx + Math.cos(angle) * outerRadiusX * r;
            node.y = cy + Math.sin(angle) * outerRadiusY * r;
        });
    }
    initCircularLayout();

    const links = [];
    const connectionCount = {};

    data.connections.forEach(conn => {
        const source = nodes.find(n => n.name === conn.source);
        const target = nodes.find(n => n.name === conn.target);
        if (source && target) {
            const key = [conn.source, conn.target].sort().join('-');
            connectionCount[key] = (connectionCount[key] || 0) + 1;
            if (!links.find(l =>
                (l.source === source && l.target === target) ||
                (l.source === target && l.target === source)
            )) {
                links.push({ source, target, weight: 1 });
            }
        }
    });

    links.forEach(link => {
        const key = [link.source.name, link.target.name].sort().join('-');
        link.weight = connectionCount[key] || 1;
    });

    let dragNode = null;
    let offsetX = 0, offsetY = 0;
    let pinnedNode = null;
    let isPinned = false;
    let isTouchDevice = false;

    // 模拟状态 —— alpha 冷却机制
    let simAlpha = 0.3;          // 初始震荡强度
    const alphaDecay = 0.005;    // 每帧衰减率
    const alphaMin = 0.002;      // 收敛阈值
    const velocityDecay = 0.55;  // 速度阻尼
    let simSettled = false;

    function applyForces() {
        if (!dragNode && simSettled) return;

        const width = cssWidth;
        const height = cssHeight;
        const centerX = width / 2;
        const centerY = height / 2;

        // 速度阻尼
        nodes.forEach(node => {
            node.vx *= velocityDecay;
            node.vy *= velocityDecay;
        });

        // 中心引力（弱）
        nodes.forEach(node => {
            if (node !== dragNode) {
                node.vx += (centerX - node.x) * 0.002 * simAlpha;
                node.vy += (centerY - node.y) * 0.002 * simAlpha;
            }
        });

        // 碰撞排斥 —— 多遍迭代 + 线性力（小屏减到 2 轮）
        const collisionIters = cssWidth < 500 ? 2 : 3;
        for (let iter = 0; iter < collisionIters; iter++) {
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[j].x - nodes[i].x;
                    const dy = nodes[j].y - nodes[i].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const minDist = nodes[i].radius + nodes[j].radius + 4;

                    if (dist < minDist && dist > 0.01) {
                        const overlap = minDist - dist;
                        const force = overlap * 0.45 * simAlpha;
                        const fx = (dx / dist) * force;
                        const fy = (dy / dist) * force;
                        if (nodes[i] !== dragNode) { nodes[i].vx -= fx; nodes[i].vy -= fy; }
                        if (nodes[j] !== dragNode) { nodes[j].vx += fx; nodes[j].vy += fy; }
                    }
                }
            }
        }

        // 链接弹簧
        links.forEach(link => {
            const dx = link.target.x - link.source.x;
            const dy = link.target.y - link.source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 0.01) return;
            const linkGap = cssWidth < 500 ? 12 : 18;
            const ideal = link.source.radius + link.target.radius + linkGap;
            const force = (dist - ideal) * 0.004 * link.weight * simAlpha;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            if (link.source !== dragNode) { link.source.vx += fx; link.source.vy += fy; }
            if (link.target !== dragNode) { link.target.vx -= fx; link.target.vy -= fy; }
        });

        // 应用速度 + 边界钳制
        let maxSpeed = 0;
        nodes.forEach(node => {
            if (node !== dragNode) {
                node.x += node.vx;
                node.y += node.vy;
                node.x = Math.max(node.radius, Math.min(width - node.radius, node.x));
                node.y = Math.max(node.radius, Math.min(height - node.radius, node.y));
            }
            maxSpeed = Math.max(maxSpeed, Math.abs(node.vx), Math.abs(node.vy));
        });

        // 冷却：速度低于阈值时衰减 alpha
        if (maxSpeed < 0.15) {
            simAlpha += (alphaMin - simAlpha) * alphaDecay;
        }
        if (!dragNode && simAlpha <= alphaMin) {
            simSettled = true;
        }
    }

    function render() {
        const width = cssWidth;
        const height = cssHeight;

        ctx.clearRect(0, 0, width, height);

        const maxWeight = Math.max(...links.map(l => l.weight), 1);

        links.forEach(link => {
            const avgColor = {
                r: Math.round((link.source.color.r + link.target.color.r) / 2),
                g: Math.round((link.source.color.g + link.target.color.g) / 2),
                b: Math.round((link.source.color.b + link.target.color.b) / 2)
            };

            const normalizedWeight = link.weight / maxWeight;
            const isMobile = cssWidth < 500;
            const lineWidth = isMobile ? 0.3 + normalizedWeight * 2 : 0.5 + normalizedWeight * 3;
            const alpha = isMobile ? 0.12 + normalizedWeight * 0.25 : 0.15 + normalizedWeight * 0.3;

            ctx.beginPath();
            ctx.moveTo(link.source.x, link.source.y);
            ctx.lineTo(link.target.x, link.target.y);
            ctx.strokeStyle = `rgba(${avgColor.r}, ${avgColor.g}, ${avgColor.b}, ${alpha})`;
            ctx.lineWidth = lineWidth;
            ctx.stroke();
        });

        // 按文章数排序绘制：小圆底层，大圆上层
        const sortedNodes = [...nodes].sort((a, b) => a.count - b.count);
        sortedNodes.forEach(node => {
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius
            );

            gradient.addColorStop(0, `rgba(${node.color.r + 30}, ${node.color.g + 30}, ${node.color.b + 30}, 1)`);
            gradient.addColorStop(1, `rgb(${node.color.r}, ${node.color.g}, ${node.color.b})`);

            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            const isMobile = cssWidth < 500;
            const borderBrightness = node.count > 6 ? 0.3 : 0.5;
            ctx.strokeStyle = `rgba(255, 255, 255, ${borderBrightness})`;
            ctx.lineWidth = isMobile ? 1.5 : 2.5;
            ctx.stroke();

            ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
            ctx.shadowBlur = isMobile ? 2 : 3;
            ctx.shadowOffsetX = 1;
            ctx.shadowOffsetY = 1;

            ctx.fillStyle = '#ffffff';
            const fontSize = cssWidth < 400 ? 8 : cssWidth < 600 ? 10 : 12;
            const maxLen = cssWidth < 400 ? 5 : cssWidth < 600 ? 7 : 8;
            const text = node.name.length > maxLen ? node.name.substring(0, maxLen - 1) + '…' : node.name;

            // Skip text if node too small to contain it
            if (node.radius >= fontSize * 1.2) {
                ctx.font = `bold ${fontSize}px Inter, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(text, node.x, node.y);
            }

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;
        });
    }

    function showTooltip(node, x, y, pinned = false, isTouch = false) {
        tooltip.innerHTML = `
            <button class="tag-tooltip-close" onclick="closeTooltip()">×</button>
            <h4>${node.name}</h4>
            <p>文章数量: ${node.count}</p>
            <p>相关文章:</p>
            <ul>
                ${node.posts.slice(0, 3).map(post =>
                    `<li><a href="${post.url}" target="_blank">${post.title}</a></li>`
                ).join('')}
                ${node.posts.length > 3 ? '<li>...</li>' : ''}
            </ul>
        `;

        // Force layout so we can measure tooltip height for touch positioning
        tooltip.classList.add('show');
        if (pinned) {
            tooltip.classList.add('pinned');
            isPinned = true;
        } else {
            tooltip.classList.remove('pinned');
        }

        const wrapperWidth = cssWidth;
        const wrapperHeight = cssHeight;
        const tooltipWidth = tooltip.offsetWidth;
        const tooltipHeight = tooltip.offsetHeight;

        let tooltipX, tooltipY;

        if (isTouch) {
            // Position above finger so it's not covered
            tooltipX = Math.max(5, Math.min(x - tooltipWidth / 2, wrapperWidth - tooltipWidth - 5));
            tooltipY = Math.max(5, y - tooltipHeight - 20);
        } else {
            tooltipX = Math.min(x + 15, wrapperWidth - tooltipWidth - 10);
            tooltipY = Math.min(y + 15, wrapperHeight - tooltipHeight - 10);
        }

        tooltip.style.left = tooltipX + 'px';
        tooltip.style.top = tooltipY + 'px';

        if (!pinned) {
            pinnedNode = null;
        }
    }

    window.closeTooltip = function() {
        tooltip.classList.remove('show', 'pinned');
        isPinned = false;
        pinnedNode = null;
    };

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function findNodeAtPos(x, y) {
        const hitPadding = cssWidth < 500 ? 12 : 8;
        return nodes.find(node => {
            const dx = node.x - x;
            const dy = node.y - y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const clickRadius = node.radius + hitPadding;
            return distance < clickRadius;
        });
    }

    let longPressTimer = null;
    let longPressTriggered = false;
    let touchStartPos = null;

    function clearLongPress() {
        if (longPressTimer) {
            clearTimeout(longPressTimer);
            longPressTimer = null;
        }
        longPressTriggered = false;
    }

    function handlePointerDown(e) {
        e.preventDefault();
        isTouchDevice = !!e.touches;
        const pos = getMousePos(e);
        const node = findNodeAtPos(pos.x, pos.y);

        // Long-press for touch: pin tooltip after 500ms hold
        if (isTouchDevice && node) {
            touchStartPos = pos;
            longPressTriggered = false;
            clearLongPress();
            longPressTimer = setTimeout(() => {
                longPressTriggered = true;
                pinnedNode = node;
                showTooltip(node, pos.x, pos.y, true, true);
                dragNode = null;
            }, 500);
        }

        if (!longPressTriggered && node) {
            dragNode = node;
            offsetX = pos.x - dragNode.x;
            offsetY = pos.y - dragNode.y;
            simSettled = false;
            simAlpha = 0.15;
        }
    }

    function handlePointerMove(e) {
        e.preventDefault();
        const pos = getMousePos(e);
        const isTouch = !!e.touches;

        // Cancel long-press if finger moves too much
        if (isTouch && touchStartPos) {
            const dx = pos.x - touchStartPos.x;
            const dy = pos.y - touchStartPos.y;
            if (Math.sqrt(dx * dx + dy * dy) > 8) {
                clearLongPress();
                touchStartPos = null;
            }
        }

        if (dragNode) {
            dragNode.x = pos.x - offsetX;
            dragNode.y = pos.y - offsetY;
        }

        if (!isPinned && !longPressTriggered) {
            const hoverNode = findNodeAtPos(pos.x, pos.y);
            if (hoverNode) {
                showTooltip(hoverNode, pos.x, pos.y, false, isTouch);
            } else {
                tooltip.classList.remove('show');
            }
        }
    }

    function handlePointerUp(e) {
        clearLongPress();
        touchStartPos = null;
        dragNode = null;
    }

    function handlePointerLeave() {
        clearLongPress();
        touchStartPos = null;
        dragNode = null;
        if (!isPinned) {
            tooltip.classList.remove('show');
        }
    }

    canvas.addEventListener('mousedown', (e) => { if (e.button === 0) handlePointerDown(e); });
    canvas.addEventListener('touchstart', handlePointerDown, { passive: false });
    canvas.addEventListener('mousemove', handlePointerMove);
    canvas.addEventListener('touchmove', handlePointerMove, { passive: false });
    canvas.addEventListener('mouseup', handlePointerUp);
    canvas.addEventListener('touchend', handlePointerUp);
    canvas.addEventListener('mouseleave', handlePointerLeave);
    canvas.addEventListener('touchcancel', handlePointerLeave);

    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const pos = getMousePos(e);
        const clickedNode = findNodeAtPos(pos.x, pos.y);
        if (clickedNode) {
            pinnedNode = clickedNode;
            showTooltip(clickedNode, pos.x, pos.y, true);
        }
    });

    canvas.addEventListener('click', (e) => {
        const pos = getMousePos(e);
        const clickedNode = findNodeAtPos(pos.x, pos.y);
        if (!clickedNode && isPinned) {
            closeTooltip();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isPinned) {
            closeTooltip();
        }
    });

    function animate() {
        applyForces();
        render();
        // 收敛后降低渲染频率以节省资源
        if (simSettled && !dragNode) {
            setTimeout(() => requestAnimationFrame(animate), 500);
        } else {
            requestAnimationFrame(animate);
        }
    }

    animate();
}
