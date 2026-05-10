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
            x: Math.random() * canvas.width / window.devicePixelRatio,
            y: Math.random() * canvas.height / window.devicePixelRatio,
            vx: 0,
            vy: 0,
            radius: getNormalizedRadius(tag.count, maxCount),
            color: color
        };
    });

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

    function applyForces() {
        const width = canvas.width / window.devicePixelRatio;
        const height = canvas.height / window.devicePixelRatio;
        const centerX = width / 2;
        const centerY = height / 2;

        nodes.forEach(node => {
            node.vx = 0;
            node.vy = 0;
        });

        nodes.forEach(node => {
            if (node !== dragNode) {
                const dx = centerX - node.x;
                const dy = centerY - node.y;
                node.vx += dx * 0.001;
                node.vy += dy * 0.001;
            }
        });

        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[j].x - nodes[i].x;
                const dy = nodes[j].y - nodes[i].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const minDistance = nodes[i].radius + nodes[j].radius + 30;

                if (dist < minDistance && dist > 0) {
                    const force = 800 / (dist * dist);
                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;
                    if (nodes[i] !== dragNode) {
                        nodes[i].vx -= fx;
                        nodes[i].vy -= fy;
                    }
                    if (nodes[j] !== dragNode) {
                        nodes[j].vx += fx;
                        nodes[j].vy += fy;
                    }
                }
            }
        }

        links.forEach(link => {
            const dx = link.target.x - link.source.x;
            const dy = link.target.y - link.source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const idealDistance = link.source.radius + link.target.radius + 50;
            const force = (dist - idealDistance) * 0.01 * link.weight;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            if (link.source !== dragNode) {
                link.source.vx += fx;
                link.source.vy += fy;
            }
            if (link.target !== dragNode) {
                link.target.vx -= fx;
                link.target.vy -= fy;
            }
        });

        nodes.forEach(node => {
            if (node !== dragNode) {
                node.x += node.vx;
                node.y += node.vy;
                node.x = Math.max(node.radius, Math.min(width - node.radius, node.x));
                node.y = Math.max(node.radius, Math.min(height - node.radius, node.y));
            }
        });
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

        nodes.forEach(node => {
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
        requestAnimationFrame(animate);
    }

    animate();
}
