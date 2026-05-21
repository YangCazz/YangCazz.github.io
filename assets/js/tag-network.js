// 关键词网络图
function initTagNetwork() {
    const canvas = document.getElementById('tagNetworkCanvas');
    const tooltip = document.getElementById('tagTooltip');
    const dataScript = document.getElementById('tagNetworkData');

    if (!canvas || !dataScript) return;

    const ctx = canvas.getContext('2d');
    const data = JSON.parse(dataScript.textContent);

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const maxCount = Math.max(...data.tags.map(t => t.count));

    function getNormalizedRadius(count, maxCount) {
        const minRadius = 18;
        const maxRadius = 55;
        const growthThreshold = 10;

        if (count <= growthThreshold) {
            const ratio = count / growthThreshold;
            return minRadius + (maxRadius - minRadius) * 0.7 * ratio;
        } else {
            const baseSize = minRadius + (maxRadius - minRadius) * 0.7;
            const extraGrowth = Math.log(1 + (count - growthThreshold)) * 3;
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
        const cx = canvas.width / window.devicePixelRatio / 2;
        const cy = canvas.height / window.devicePixelRatio / 2;
        const sortedByCount = [...nodes].sort((a, b) => b.count - a.count);
        const angleStep = (2 * Math.PI) / nodes.length;
        const outerRadius = Math.min(cx, cy) * 0.50;
        sortedByCount.forEach((node, i) => {
            const angle = i * angleStep + (Math.random() - 0.5) * 0.3;
            // 按排名比例缩放半径：排名越前（count 越大）越靠近中心
            const rankRatio = i / Math.max(1, nodes.length - 1);
            const r = outerRadius * (0.35 + rankRatio * 0.65);
            node.x = cx + Math.cos(angle) * r;
            node.y = cy + Math.sin(angle) * r;
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

    // 模拟状态 —— alpha 冷却机制
    let simAlpha = 0.3;          // 初始震荡强度
    const alphaDecay = 0.005;    // 每帧衰减率
    const alphaMin = 0.002;      // 收敛阈值
    const velocityDecay = 0.55;  // 速度阻尼
    let simSettled = false;

    function applyForces() {
        if (!dragNode && simSettled) return;

        const width = canvas.width / window.devicePixelRatio;
        const height = canvas.height / window.devicePixelRatio;
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

        // 碰撞排斥 —— 多遍迭代 + 线性力
        for (let iter = 0; iter < 3; iter++) {
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
            const ideal = link.source.radius + link.target.radius + 18;
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
        const width = canvas.width / window.devicePixelRatio;
        const height = canvas.height / window.devicePixelRatio;

        ctx.clearRect(0, 0, width, height);

        const maxWeight = Math.max(...links.map(l => l.weight), 1);

        links.forEach(link => {
            const avgColor = {
                r: Math.round((link.source.color.r + link.target.color.r) / 2),
                g: Math.round((link.source.color.g + link.target.color.g) / 2),
                b: Math.round((link.source.color.b + link.target.color.b) / 2)
            };

            const normalizedWeight = link.weight / maxWeight;
            const lineWidth = 0.5 + normalizedWeight * 3;
            const alpha = 0.15 + normalizedWeight * 0.3;

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

            const borderBrightness = node.count > 6 ? 0.3 : 0.5;
            ctx.strokeStyle = `rgba(255, 255, 255, ${borderBrightness})`;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
            ctx.shadowBlur = 5;
            ctx.shadowOffsetX = 1;
            ctx.shadowOffsetY = 1;

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const text = node.name.length > 8 ? node.name.substring(0, 7) + '...' : node.name;
            ctx.fillText(text, node.x, node.y);

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;
        });
    }

    function showTooltip(node, x, y, pinned = false) {
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

        const wrapperWidth = canvas.width / window.devicePixelRatio;
        const wrapperHeight = canvas.height / window.devicePixelRatio;

        const tooltipX = Math.min(x + 15, wrapperWidth - 250);
        const tooltipY = Math.min(y + 15, wrapperHeight - 150);

        tooltip.style.left = tooltipX + 'px';
        tooltip.style.top = tooltipY + 'px';

        if (pinned) {
            tooltip.classList.add('show', 'pinned');
            isPinned = true;
        } else {
            tooltip.classList.add('show');
            tooltip.classList.remove('pinned');
        }
    }

    window.closeTooltip = function() {
        tooltip.classList.remove('show', 'pinned');
        isPinned = false;
        pinnedNode = null;
    };

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    function findNodeAtPos(x, y) {
        return nodes.find(node => {
            const dx = node.x - x;
            const dy = node.y - y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const clickRadius = node.radius + 8;
            return distance < clickRadius;
        });
    }

    canvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        const pos = getMousePos(e);
        dragNode = findNodeAtPos(pos.x, pos.y);
        if (dragNode) {
            offsetX = pos.x - dragNode.x;
            offsetY = pos.y - dragNode.y;
            simSettled = false;   // 拖拽时重新激活模拟
            simAlpha = 0.15;
        }
    });

    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const pos = getMousePos(e);
        const clickedNode = findNodeAtPos(pos.x, pos.y);

        if (clickedNode) {
            pinnedNode = clickedNode;
            showTooltip(clickedNode, pos.x, pos.y, true);
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        const pos = getMousePos(e);

        if (dragNode) {
            dragNode.x = pos.x - offsetX;
            dragNode.y = pos.y - offsetY;
        }

        if (!isPinned) {
            const hoverNode = findNodeAtPos(pos.x, pos.y);
            if (hoverNode) {
                showTooltip(hoverNode, pos.x, pos.y, false);
            } else {
                tooltip.classList.remove('show');
            }
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
            dragNode = null;
        }
    });

    canvas.addEventListener('mouseleave', () => {
        dragNode = null;
        if (!isPinned) {
            tooltip.classList.remove('show');
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
