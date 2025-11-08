document.addEventListener('DOMContentLoaded', () => {
    const mapsListContainer = document.getElementById('maps-list-container');

    const mainGridCanvas = document.getElementById('main-grid-canvas');
    const mainGridCtx = mainGridCanvas.getContext('2d');

    // 动态创建放大镜容器并追加到 body，确保 fixed 定位相对于视口
    const magnifierContainer = document.createElement('div');
    magnifierContainer.id = 'magnifier-container';
    magnifierContainer.className = 'magnifier-container hidden';
    document.body.appendChild(magnifierContainer);

    const magnifierCanvas = document.createElement('canvas');
    magnifierCanvas.id = 'magnifier-canvas';
    magnifierContainer.appendChild(magnifierCanvas);

    const magnifierCtx = magnifierCanvas.getContext('2d');
    const MAGNIFIER_SIZE = 200; // 放大镜直径
    const MAGNIFIER_SCALE = 3; // 放大倍数
    const MAGNIFIER_RADIUS = MAGNIFIER_SIZE / 2; // 放大镜半径
    
    // 保存最后鼠标位置，用于在 drawMainGrid 后更新放大镜
    let lastMouseX = null;
    let lastMouseY = null;

    let pixelMapData = null;
    let mapsList = []; // 存储所有地图列表
    let currentMapName = null; // 当前选中的地图名称
    let mapsDataCache = {}; // 缓存已加载的地图数据 {mapName: data}
    
    // 性能优化：缓存DOM元素引用，避免频繁查询
    const mapItemCache = new Map(); // 缓存地图项的DOM引用 {mapName: {mapItem, canvas}}
    
    // 性能优化：离屏Canvas缓存静态网格
    let staticGridCanvas = null; // 离屏Canvas，用于缓存静态网格
    let staticGridCtx = null;
    let staticGridDirty = true; // 标记静态网格是否需要重新绘制
    
    const SQUARE_SIZE_MINI = 5; // 小地图每个方块的尺寸 (增大)

    const SQUARE_SIZE_MAIN = 10; // 主网格每个方块的尺寸 (减小)
    const BORDER_WIDTH_MAIN = 1; // 主网格方块边框宽度
    const BORDER_COLOR_MAIN = '#808080'; // 未完成方块边框颜色 (灰色)
    const COMPLETED_BORDER_COLOR = '#000000'; // 已完成方块边框颜色 (黑色)
    const PATH_HIGHLIGHT_COLOR = '#8B0000'; // 路径高亮颜色 (深红色)
    const PATH_HIGHLIGHT_WIDTH = BORDER_WIDTH_MAIN + 2; // 路径高亮边框宽度 (加粗)

    let completedSquares = []; // 存储已完成方块的坐标 {row, col}
    let completedSquaresSet = new Set(); // 用于快速查找已完成方块的 Set
    let currentPath = []; // 存储当前路径上的方块坐标 {row, col}
    let currentPathSet = new Set(); // 用于快速查找当前路径的 Set
    let currentPathIndexMap = new Map(); // 用于快速查找坐标在路径中的索引 {coordKey: index}
    let currentPathData = null; // 存储当前路径的预计算数据

    let startPoint = null; // 路径起点 {row, col}
    let endPoint = null;   // 路径终点 {row, col}
    let isDrawing = false; // 是否正在绘制路径

    // 鼠标悬停相关
    let hoveredSquare = null; // 当前悬停的方格坐标 {row, col} 或 null
    const HOVER_HIGHLIGHT_COLOR = '#4299e1'; // 悬停高亮颜色（使用主题色）
    const HOVER_HIGHLIGHT_WIDTH = 3; // 悬停高亮边框宽度
    let hoverUpdateScheduled = false; // 用于节流重绘
    let tooltipPositionUpdateScheduled = false; // 用于节流工具提示位置更新

    const markCompletedBtn = document.getElementById('mark-completed-btn');

    // 对角预设模式相关元素
    const enterDiagonalModeBtn = document.getElementById('enter-diagonal-mode-btn');
    const prevDiagonalBtn = document.getElementById('prev-diagonal-btn');
    const nextDiagonalBtn = document.getElementById('next-diagonal-btn');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressText = document.getElementById('progress-text');

    const saveProgressBtn = document.getElementById('save-progress-btn');
    const loadProgressBtn = document.getElementById('load-progress-btn');
    const resetProgressBtn = document.getElementById('reset-progress-btn');

    // 对角预设模式状态变量
    let isDiagonalMode = false;
    let diagonalPaths = []; // 存储所有对角线路径
    let currentDiagonalIndex = -1; // 当前显示的对角线索引
    let isConfirmedCompleted = false; // 标记是否已确认完成（完成且已保存）

    // 当前路径状态显示元素
    const currentPathStatusDiv = document.getElementById('current-path-status');
    const pathTotalSquaresSpan = document.getElementById('path-total-squares');
    const pathColorBreakdownDiv = document.getElementById('path-color-breakdown');
    const pathColorSequenceContainer = document.getElementById('path-color-sequence-container'); // 引用新的容器
    const pathColorSequenceDiv = document.getElementById('path-color-sequence');

    let longPressTimer = null; // 用于长按的定时器
    const LONG_PRESS_DELAY = 300; // 长按延迟 (ms)
    const REPEAT_INTERVAL = 100; // 重复间隔 (ms)

    let highlightedSequenceItem = null; // 用于存储当前被选中的编织顺序项（用于高亮主网格）
    let highlightedSequenceRange = null; // 用于存储被选中序列项的索引范围 {startIndex, endIndex}

    // 启用/禁用手动路径选择
    function toggleManualPathSelection(enable) {
        if (enable) {
            mainGridCanvas.addEventListener('mousedown', onMouseDown);
            mainGridCanvas.addEventListener('mousemove', onMouseMove);
            mainGridCanvas.addEventListener('mouseup', onMouseUp);
        } else {
            mainGridCanvas.removeEventListener('mousedown', onMouseDown);
            mainGridCanvas.removeEventListener('mousemove', onMouseMove);
            mainGridCanvas.removeEventListener('mouseup', onMouseUp);
        }
    }

    // 将之前的事件处理函数提取出来，方便添加和移除
    function onMouseDown(event) {
        if (isDiagonalMode) return; // 对角模式下禁用手动选择
        const coords = getGridCoords(event);
        if (coords) {
            startPoint = coords;
            isDrawing = true;
            endPoint = null; // 重置终点
            currentPath = []; // 清空当前路径
            currentPathSet = new Set(); // 同步清空 Set
            currentPathIndexMap = new Map(); // 同步清空 Map
            currentPathData = null;
            updateMarkCompletedButtonState(); // 更新标记完成按钮状态
            drawMainGrid(); // 重新绘制以清除旧路径
        }
    }

    // 性能优化：节流鼠标移动事件
    let mouseMoveUpdateScheduled = false;
    
    function onMouseMove(event) {
        if (isDiagonalMode) return; // 对角模式下禁用手动选择
        if (!isDrawing || !startPoint) return;
        
        // 性能优化：使用 requestAnimationFrame 节流
        if (!mouseMoveUpdateScheduled) {
            mouseMoveUpdateScheduled = true;
            requestAnimationFrame(() => {
                const coords = getGridCoords(event);
                if (coords) {
                    endPoint = coords;
                    currentPath = calculatePath(startPoint, endPoint);
                    currentPathSet = new Set(currentPath.map(coord => `${coord.row},${coord.col}`)); // 同步更新 Set
                    currentPathIndexMap = new Map();
                    currentPath.forEach((coord, index) => {
                        currentPathIndexMap.set(`${coord.row},${coord.col}`, index);
                    });
                    drawMainGrid();
                }
                mouseMoveUpdateScheduled = false;
            });
        }
    }

    function onMouseUp() {
        if (isDiagonalMode) return; // 对角模式下禁用手动选择
        isDrawing = false;
        if (startPoint && endPoint) {
            currentPath = calculatePath(startPoint, endPoint);
            currentPathSet = new Set(currentPath.map(coord => `${coord.row},${coord.col}`)); // 同步更新 Set
            updateMarkCompletedButtonState(); // 更新标记完成按钮状态
            currentPathIndexMap = new Map();
            currentPath.forEach((coord, index) => {
                currentPathIndexMap.set(`${coord.row},${coord.col}`, index);
            });
            drawMainGrid();
        } else {
            currentPath = [];
            currentPathSet = new Set();
            currentPathIndexMap = new Map();
            updateMarkCompletedButtonState(); // 更新标记完成按钮状态
            if (startPoint) {
                currentPath.push(startPoint);
                currentPathSet.add(`${startPoint.row},${startPoint.col}`); // 同步更新 Set
                currentPathIndexMap.set(`${startPoint.row},${startPoint.col}`, 0); // 同步更新 Map
                endPoint = startPoint; // 确保终点也被设置
            }
            drawMainGrid();
        }
    }

    // 从鼠标事件获取网格坐标
    function getGridCoords(event) {
        if (!pixelMapData) return null;

        const rect = mainGridCanvas.getBoundingClientRect();
        const scaleX = mainGridCanvas.width / rect.width;
        const scaleY = mainGridCanvas.height / rect.height;

        // 获取鼠标在 canvas 中的坐标（考虑缩放）
        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;

        // 计算网格坐标，考虑边框偏移
        const col = Math.floor((x - BORDER_WIDTH_MAIN) / (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN));
        const row = Math.floor((y - BORDER_WIDTH_MAIN) / (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN));

        // 检查坐标是否在有效范围内
        const height = pixelMapData.grid_data.length;
        const width = pixelMapData.grid_data[0].length;

        if (row >= 0 && row < height && col >= 0 && col < width) {
            return { row: row, col: col };
        }

        return null;
    }

    // 使用 Bresenham 算法计算两点之间的路径
    function calculatePath(p1, p2) {
        const path = [];
        let x0 = p1.col;
        let y0 = p1.row;
        let x1 = p2.col;
        let y1 = p2.row;

        const dx = Math.abs(x1 - x0);
        const dy = Math.abs(y1 - y0);
        const sx = x0 < x1 ? 1 : -1;
        const sy = y0 < y1 ? 1 : -1;
        let err = dx - dy;

        while (true) {
            path.push({ row: y0, col: x0 });

            if (x0 === x1 && y0 === y1) break;

            const e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
        }

        return path;
    }

    // 初始时启用手动路径选择
    toggleManualPathSelection(true);

    // 初始化放大镜
    magnifierCanvas.width = MAGNIFIER_SIZE;
    magnifierCanvas.height = MAGNIFIER_SIZE;
    magnifierContainer.style.width = `${MAGNIFIER_SIZE}px`;
    magnifierContainer.style.height = `${MAGNIFIER_SIZE}px`;
    
    // 禁用图像平滑以保持像素清晰（放大镜需要清晰的像素）
    magnifierCtx.imageSmoothingEnabled = false;

    // 初始化：加载地图列表
    loadMapsList();

    // 加载地图列表
    function loadMapsList() {
        fetch('data/json/maps_list.json')
            .then(response => response.json())
            .then(data => {
                mapsList = data.maps || [];
                console.log("加载的地图列表:", mapsList);
                
                // 创建地图列表 UI
                createMapsListUI();
                
                // 默认加载第一张图
                if (mapsList.length > 0) {
                    switchMap(mapsList[0].name, mapsList[0].file);
                }
            })
            .catch(error => {
                console.error('加载地图列表失败:', error);
                // 如果列表文件不存在，尝试加载默认地图
                loadDefaultMap();
            });
    }

    // 加载默认地图（向后兼容）
    function loadDefaultMap() {
        fetch('data/json/pixel_map_data.json')
            .then(response => response.json())
            .then(data => {
                mapsList = [{ name: 'pixel_map_data', file: 'pixel_map_data.json', displayName: '像素地图' }];
                createMapsListUI();
                switchMap('pixel_map_data', 'pixel_map_data.json');
            })
            .catch(error => console.error('加载默认地图失败:', error));
    }

    // 创建地图列表 UI
    function createMapsListUI() {
        mapsListContainer.innerHTML = '';
        
        mapsList.forEach((map, index) => {
            const mapItem = document.createElement('div');
            mapItem.className = 'map-item';
            mapItem.dataset.mapName = map.name;
            mapItem.dataset.mapFile = map.file;
            
            const mapName = document.createElement('div');
            mapName.className = 'map-item-name';
            mapName.textContent = map.displayName || map.name;
            
            const mapCanvas = document.createElement('canvas');
            mapCanvas.className = 'map-item-canvas';
            mapCanvas.dataset.mapName = map.name;
            
            // 创建统计信息容器
            const statsContainer = document.createElement('div');
            statsContainer.className = 'map-item-stats';
            statsContainer.dataset.mapName = map.name;
            
            mapItem.appendChild(mapName);
            mapItem.appendChild(mapCanvas);
            mapItem.appendChild(statsContainer);
            
            // 性能优化：缓存DOM元素引用
            mapItemCache.set(map.name, { mapItem, canvas: mapCanvas, statsContainer });
            
            // 点击切换地图（添加防抖，防止双击误触）
            let clickTimer = null;
            mapItem.addEventListener('click', () => {
                // 清除之前的定时器
                if (clickTimer) {
                    clearTimeout(clickTimer);
                }
                
                // 设置新的定时器，延迟执行切换（防抖）
                clickTimer = setTimeout(() => {
                    switchMap(map.name, map.file);
                    clickTimer = null;
                }, 200); // 200ms 防抖延迟，防止双击误触
            });
            
            mapsListContainer.appendChild(mapItem);
            
            // 异步加载并绘制小地图
            loadMapDataForMiniMap(map.name, map.file, mapCanvas);
        });
    }

    // 为地图卡片加载数据并绘制小地图
    function loadMapDataForMiniMap(mapName, mapFile, canvas) {
        // 获取该地图的保存状态
        const storageKey = `weavingProgressState_${mapName}`;
        const savedProgressState = localStorage.getItem(storageKey);
        let completedSquaresForMap = [];
        
        if (savedProgressState) {
            try {
                const progressState = JSON.parse(savedProgressState);
                completedSquaresForMap = progressState.completedSquares || [];
            } catch (e) {
                console.error(`加载地图 ${mapName} 的状态失败:`, e);
            }
        }
        
        const completedSet = new Set(completedSquaresForMap.map(sq => `${sq.row},${sq.col}`));
        
        // 如果已缓存，直接使用
        if (mapsDataCache[mapName]) {
            drawMiniMapForItem(canvas, mapsDataCache[mapName], completedSet);
            updateMapItemStats(mapName, mapsDataCache[mapName]);
            return;
        }
        
        // 否则加载数据
        fetch(`data/json/${mapFile}`)
            .then(response => response.json())
            .then(data => {
                mapsDataCache[mapName] = data;
                drawMiniMapForItem(canvas, data, completedSet);
                updateMapItemStats(mapName, data);
            })
            .catch(error => console.error(`加载地图 ${mapName} 失败:`, error));
    }

    // 为地图卡片绘制小地图
    function drawMiniMapForItem(canvas, data, completedSet = null) {
        if (!data) return;
        
        const gridData = data.grid_data;
        const colorMap = data.color_map;
        const height = gridData.length;
        const width = gridData[0].length;
        
        // 设置 canvas 尺寸
        const scale = 0.3; // 缩小比例
        canvas.width = width * SQUARE_SIZE_MINI * scale;
        canvas.height = height * SQUARE_SIZE_MINI * scale;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // 绘制网格（如果提供了完成状态，显示边框）
        for (let r_idx = 0; r_idx < height; r_idx++) {
            for (let c_idx = 0; c_idx < width; c_idx++) {
                const pixelValue = gridData[r_idx][c_idx];
                const originalColor = colorMap[String(pixelValue)];
                
                const x = c_idx * SQUARE_SIZE_MINI * scale;
                const y = r_idx * SQUARE_SIZE_MINI * scale;
                
                if (originalColor) {
                    ctx.fillStyle = `rgb(${originalColor[0]}, ${originalColor[1]}, ${originalColor[2]})`;
                } else {
                    ctx.fillStyle = '#000000';
                }
                
                ctx.fillRect(x, y, SQUARE_SIZE_MINI * scale, SQUARE_SIZE_MINI * scale);
            }
        }
        
        // 第二遍绘制：为已完成的方块添加渐变蓝色透明覆盖层
        if (completedSet && completedSet.size > 0) {
            for (let r_idx = 0; r_idx < height; r_idx++) {
                for (let c_idx = 0; c_idx < width; c_idx++) {
                    const coordKey = `${r_idx},${c_idx}`;
                    if (completedSet.has(coordKey)) {
                        const x = c_idx * SQUARE_SIZE_MINI * scale;
                        const y = r_idx * SQUARE_SIZE_MINI * scale;
                        const squareSize = SQUARE_SIZE_MINI * scale;
                        
                        // 创建径向渐变（从中心到边缘）
                        const centerX = x + squareSize / 2;
                        const centerY = y + squareSize / 2;
                        const radius = squareSize / 2;
                        
                        const gradient = ctx.createRadialGradient(
                            centerX, centerY, 0,           // 渐变中心（方块中心）
                            centerX, centerY, radius        // 渐变边缘（方块边缘）
                        );
                        
                        // 渐变：中心较透明，边缘较不透明，形成柔和的覆盖效果
                        gradient.addColorStop(0, 'rgba(66, 153, 225, 0.3)');   // 中心：30% 不透明度的蓝色
                        gradient.addColorStop(0.5, 'rgba(66, 153, 225, 0.4)'); // 中间：40% 不透明度
                        gradient.addColorStop(1, 'rgba(66, 153, 225, 0.5)');   // 边缘：50% 不透明度
                        
                        ctx.fillStyle = gradient;
                        ctx.fillRect(x, y, squareSize, squareSize);
                    }
                }
            }
        }
        
        // 第三遍绘制：如果地图完成，在右上角绘制完成标志
        if (completedSet) {
            const totalSquares = height * width;
            const completedCount = completedSet.size;
            
            // 检查是否所有方块都已完成（允许1-2个误差，避免浮点数问题）
            if (completedCount >= totalSquares - 1) {
                const canvasWidth = canvas.width;
                const canvasHeight = canvas.height;
                
                // 完成标志的位置（右上角）
                // 标志大小：取画布较小边的20%，确保不会太大
                const badgeSize = Math.min(canvasWidth, canvasHeight) * 0.2;
                const badgeRadius = badgeSize / 2;
                
                // 边距：确保标志完全在canvas内部
                const padding = Math.max(2, badgeRadius * 0.3); // 至少2px边距
                
                // 右上角位置：距离右边和顶部都有padding
                const badgeX = canvasWidth - badgeRadius - padding;
                const badgeY = badgeRadius + padding;
                
                // 设置阴影效果（在绘制之前设置）
                ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
                ctx.shadowBlur = badgeSize * 0.1;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = badgeSize * 0.05;
                
                // 绘制圆形背景（绿色渐变）
                const badgeGradient = ctx.createRadialGradient(
                    badgeX, badgeY, 0,
                    badgeX, badgeY, badgeRadius
                );
                badgeGradient.addColorStop(0, '#22C55E'); // 浅绿色
                badgeGradient.addColorStop(1, '#16A34A'); // 深绿色
                
                ctx.fillStyle = badgeGradient;
                ctx.beginPath();
                ctx.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);
                ctx.fill();
                
                // 清除阴影，准备绘制边框和对勾
                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
                
                // 绘制白色边框
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.lineWidth = Math.max(1, badgeSize * 0.08);
                ctx.beginPath();
                ctx.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);
                ctx.stroke();
                
                // 绘制对勾（✓）
                ctx.strokeStyle = 'white';
                ctx.lineWidth = Math.max(2, badgeSize * 0.12);
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                
                // 对勾的路径（从左上到右下，再到右上）
                const checkmarkSize = badgeRadius * 0.5; // 稍微缩小对勾，确保在圆内
                const checkmarkX = badgeX;
                const checkmarkY = badgeY;
                
                // 对勾路径：从左上到中间，再到右上
                ctx.beginPath();
                ctx.moveTo(checkmarkX - checkmarkSize * 0.35, checkmarkY - checkmarkSize * 0.1);
                ctx.lineTo(checkmarkX - checkmarkSize * 0.1, checkmarkY + checkmarkSize * 0.25);
                ctx.lineTo(checkmarkX + checkmarkSize * 0.4, checkmarkY - checkmarkSize * 0.3);
                ctx.stroke();
            }
        }
    }

    // 切换地图
    function switchMap(mapName, mapFile) {
        // 如果切换到相同的地图，直接返回，避免重置进度
        if (currentMapName === mapName) {
            console.log(`地图 ${mapName} 已经是当前地图，无需切换`);
            return;
        }
        
        // 保存当前地图的状态
        if (currentMapName) {
            saveProgressForMap(currentMapName);
        }
        
        // 更新当前地图名称
        currentMapName = mapName;
        
        // 性能优化：使用缓存更新UI，减少DOM查询
        mapItemCache.forEach((data, cachedMapName) => {
            if (cachedMapName === mapName) {
                data.mapItem.classList.add('active');
            } else {
                data.mapItem.classList.remove('active');
            }
        });
        
        // 加载地图数据
        if (mapsDataCache[mapName]) {
            // 使用缓存
            pixelMapData = mapsDataCache[mapName];
            initializeMapData();
            // 切换地图后，更新所有小地图显示和统计信息
            updateAllMapItemsMiniMap();
            updateMapItemStats(mapName, mapsDataCache[mapName]);
        } else {
            // 加载新数据
            fetch(`data/json/${mapFile}`)
                .then(response => response.json())
                .then(data => {
                    mapsDataCache[mapName] = data;
                    pixelMapData = data;
                    initializeMapData();
                    // 切换地图后，更新所有小地图显示和统计信息
                    updateAllMapItemsMiniMap();
                    updateMapItemStats(mapName, data);
                })
                .catch(error => console.error(`加载地图 ${mapName} 失败:`, error));
        }
    }

    // 初始化地图数据（加载数据后调用）
    function initializeMapData() {
        // 重置状态
        completedSquares = [];
        completedSquaresSet = new Set();
        currentPath = [];
        currentPathSet = new Set();
        currentPathIndexMap = new Map();
        currentPathData = null;
        startPoint = null;
        endPoint = null;
        isDiagonalMode = false;
        currentDiagonalIndex = -1;
        highlightedSequenceRange = null;
        
        // 清除颜色序列的选中状态
        clearSequenceSelection();
        
        // 性能优化：标记静态网格需要重新绘制
        staticGridDirty = true;
        
        // 计算对角线路径
        calculateAllDiagonalPaths();
        
        // 不自动加载进度，只更新按钮状态
        // 用户需要手动点击"加载上次进度"按钮来加载保存的状态
        updateLoadButtonState(); // 更新加载按钮状态（检查是否有保存的数据）
        
        // 更新 UI
        drawMainGrid();
        updateProgressBar();
        updateCurrentPathStatus();
        updateModeUI();
    }

    // 全局变量：存储所有路径中的最大颜色数量，用于计算固定高度
    let maxColorCount = 0;

    // 计算所有从右下角开始，从右向左推进的对角线路径，并预计算所有相关数据
    function calculateAllDiagonalPaths() {
        if (!pixelMapData) return;

        const height = pixelMapData.grid_data.length;
        const width = pixelMapData.grid_data[0].length;
        const gridData = pixelMapData.grid_data;
        const colorMap = pixelMapData.color_map;
        diagonalPaths = [];
        maxColorCount = 0; // 重置最大颜色数量

        // 从右下角开始 (r = height - 1, c = width - 1)
        // 对角线总数 = width + height - 1

        // 先处理从右下角向左上方移动的对角线
        for (let sum = (width - 1) + (height - 1); sum >= 0; sum--) {
            const path = [];
            const colorBreakdown = {}; // 预计算颜色划分
            const colorSequence = []; // 预计算颜色序列
            let lastColorKey = null;
            let count = 0;
            let lastPixelValue = null;
            let sequenceStartIndex = 0; // 当前序列在路径中的起始索引

            for (let r = height - 1; r >= 0; r--) {
                const c = sum - r;
                if (c >= 0 && c < width) {
                    const coord = { row: r, col: c };
                    const pathIndex = path.length; // 当前方块在路径中的索引
                    path.push(coord);

                    // 预计算颜色信息
                    const pixelValue = gridData[r][c];
                    const color = colorMap[String(pixelValue)];
                    const currentColorKey = color ? `rgb(${color[0]},${color[1]},${color[2]})` : '#000000';
                    
                    // 更新颜色划分
                    colorBreakdown[currentColorKey] = (colorBreakdown[currentColorKey] || 0) + 1;

                    // 统计连续颜色序列（从右往左）
                    if (currentColorKey === lastColorKey) {
                        count++;
                    } else {
                        if (lastColorKey !== null) {
                            // 记录前一个序列的结束索引（当前索引的前一个）
                            const sequenceEndIndex = pathIndex - 1;
                            colorSequence.push({ 
                                color: lastColorKey, 
                                count: count, 
                                pixelValue: lastPixelValue,
                                startIndex: sequenceStartIndex, // 序列在路径中的起始索引
                                endIndex: sequenceEndIndex // 序列在路径中的结束索引
                            });
                        }
                        lastColorKey = currentColorKey;
                        count = 1;
                        lastPixelValue = pixelValue;
                        sequenceStartIndex = pathIndex; // 新序列的起始索引
                    }
                }
            }
            
            if (path.length > 0) {
                // 添加最后一个连续块
                if (lastColorKey !== null) {
                    const sequenceEndIndex = path.length - 1;
                    colorSequence.push({ 
                        color: lastColorKey, 
                        count: count, 
                        pixelValue: lastPixelValue,
                        startIndex: sequenceStartIndex, // 序列在路径中的起始索引
                        endIndex: sequenceEndIndex // 序列在路径中的结束索引
                    });
                }

                // 统计当前路径的颜色数量
                const currentColorCount = Object.keys(colorBreakdown).length;
                maxColorCount = Math.max(maxColorCount, currentColorCount);

                // 存储路径及其预计算的数据
                diagonalPaths.push({
                    path: path,
                    totalSquares: path.length,
                    colorBreakdown: colorBreakdown,
                    colorSequence: colorSequence
                });
            }
        }

        console.log("生成的所有对角线路径（含预计算数据）:", diagonalPaths);
        console.log("最大颜色数量:", maxColorCount);
    }

    // 根据索引设置当前对角线路径并更新网格
    function setDiagonalPath(index) {
        console.log(`[setDiagonalPath] called with index: ${index}`);
        if (index >= 0 && index < diagonalPaths.length) {
            // 切换路径时清除选中状态
            clearSequenceSelection();
            
            currentDiagonalIndex = index;
            const diagonalData = diagonalPaths[currentDiagonalIndex];
            currentPath = diagonalData.path; // 使用预计算的路径
            currentPathSet = new Set(currentPath.map(coord => `${coord.row},${coord.col}`)); // 创建 Set 用于快速查找
            // 创建坐标到索引的映射
            currentPathIndexMap = new Map();
            currentPath.forEach((coord, index) => {
                currentPathIndexMap.set(`${coord.row},${coord.col}`, index);
            });
            currentPathData = diagonalData; // 存储预计算的数据
            console.log(`[setDiagonalPath] currentDiagonalIndex: ${currentDiagonalIndex}, currentPath length: ${currentPath.length}`);
            drawMainGrid();
            updateNavigationButtons();
            updateCurrentPathStatus(); // 更新路径状态显示（现在使用预计算数据）
            updateMarkCompletedButtonState(); // 更新标记完成按钮状态
        } else {
            console.warn(`[setDiagonalPath] 尝试设置无效的对角线路径索引: ${index} (有效范围: 0-${diagonalPaths.length - 1})。保持当前索引: ${currentDiagonalIndex}`);
            updateNavigationButtons();
        }
    }

    // 检查是否所有方块都已完成
    function checkIfAllCompleted() {
        if (!pixelMapData) return false;
        const totalSquares = pixelMapData.grid_data.length * pixelMapData.grid_data[0].length;
        const completedCount = completedSquares.length;
        // 允许1-2个误差，避免浮点数问题
        return completedCount >= totalSquares - 1;
    }

    // 更新“上一条”和“下一条”按钮的状态
    function updateNavigationButtons() {
        console.log(`[updateNavigationButtons] called. isDiagonalMode: ${isDiagonalMode}, currentDiagonalIndex: ${currentDiagonalIndex}, diagonalPaths.length: ${diagonalPaths.length}, isConfirmedCompleted: ${isConfirmedCompleted}`);
        
        // 如果已确认完成，禁用两个按钮
        if (isConfirmedCompleted) {
            console.log("[updateNavigationButtons] Confirmed completed. Disabling navigation buttons.");
            prevDiagonalBtn.disabled = true;
            nextDiagonalBtn.disabled = true;
            return;
        }
        
        if (!pixelMapData || diagonalPaths.length === 0) { // 如果没有数据或没有路径，禁用所有按钮
            console.log("[updateNavigationButtons] No pixelMapData or diagonalPaths are empty. Disabling buttons.");
            prevDiagonalBtn.disabled = true;
            nextDiagonalBtn.disabled = true;
            return;
        }
        prevDiagonalBtn.disabled = (currentDiagonalIndex <= 0);
        nextDiagonalBtn.disabled = (currentDiagonalIndex >= diagonalPaths.length - 1);
        console.log(`[updateNavigationButtons] prevDisabled: ${prevDiagonalBtn.disabled}, nextDisabled: ${nextDiagonalBtn.disabled}`);
    }

    // 清除颜色序列的选中状态
    function clearSequenceSelection() {
        if (pathColorSequenceDiv) {
            const selectedItem = pathColorSequenceDiv.querySelector('.color-sequence-item.selected');
            if (selectedItem) {
                selectedItem.classList.remove('selected');
            }
        }
        highlightedSequenceItem = null;
        highlightedSequenceRange = null;
    }
    
    // 更新当前路径状态的显示（使用预计算数据）
    function updateCurrentPathStatus() {
        if (!isDiagonalMode || !pixelMapData || !currentPathData) {
            currentPathStatusDiv.style.display = 'none';
            pathColorSequenceContainer.classList.add('hidden'); // 隐藏横条
            return;
        }

        currentPathStatusDiv.style.display = 'block';
        pathColorSequenceContainer.classList.remove('hidden'); // 显示横条

        // 使用预计算的总方块数
        pathTotalSquaresSpan.textContent = currentPathData.totalSquares;

        // 使用预计算的颜色划分
        const colorBreakdown = currentPathData.colorBreakdown;
        pathColorBreakdownDiv.innerHTML = ''; // 清空之前的显示
        
        // 使用全局最大颜色数量计算固定高度，确保所有路径使用相同高度
        // 这样在切换路径时，高度不会变化，按钮位置保持稳定，长按不会被中断
        const colorCount = maxColorCount > 0 ? maxColorCount : Object.keys(colorBreakdown).length;
        
        // 根据最大颜色数量计算固定高度（在创建颜色项之前计算并设置）
        // 计算参数（基于实际 CSS 样式）：
        // - path-info: font-size 0.9rem (约 14.4px) + margin-bottom 0.5rem (8px) = 约 22.4px，加上行高约 30px
        // - color-item: 高度约 20px (16px color-swatch + gap 0.25rem)
        // - gap: 0.5rem = 8px (颜色项之间的间距)
        // - 每行颜色项数量：容器宽度约 200px，每个 color-item 约 50% 宽度 (max-width: calc(50% - 0.25rem))，所以每行 2 个
        // - padding: 1rem = 16px (上下各 16px)
        // - margin-top: 0.5rem = 8px (color-breakdown-container)
        
        const itemsPerRow = 2; // 每行固定 2 个颜色项
        const rows = Math.ceil(colorCount / itemsPerRow); // 需要的行数（基于最大颜色数量）
        
        const pathInfoHeight = 30; // path-info 高度（包含 margin-bottom）
        const itemHeight = 20; // 每个 color-item 高度
        const gap = 8; // gap: 0.5rem = 8px
        const colorContainerMarginTop = 8; // margin-top: 0.5rem
        const padding = 32; // padding: 1rem (上下各 16px)
        
        // 计算颜色容器高度（基于最大颜色数量）
        // 如果只有一行，高度 = itemHeight
        // 如果多行，高度 = rows * itemHeight + (rows - 1) * gap
        const colorContainerHeight = rows > 0 
            ? (rows === 1 ? itemHeight : rows * itemHeight + (rows - 1) * gap) + colorContainerMarginTop
            : 0;
        
        // 计算总高度（固定高度，所有路径使用相同高度）
        const totalHeight = pathInfoHeight + colorContainerHeight + padding;
        
        // 设置固定高度（在创建颜色项之前设置，确保布局稳定）
        // 使用固定高度可以确保按钮位置不会因路径切换而变化
        currentPathStatusDiv.style.height = `${totalHeight}px`;
        
        // 创建颜色项（在设置高度之后）
        for (const colorKey in colorBreakdown) {
            const count = colorBreakdown[colorKey];
            const colorItemDiv = document.createElement('div');
            colorItemDiv.className = 'color-item';

            const colorSwatchDiv = document.createElement('div');
            colorSwatchDiv.className = 'color-swatch';
            colorSwatchDiv.style.backgroundColor = colorKey;
            colorItemDiv.appendChild(colorSwatchDiv);

            const countSpan = document.createElement('span');
            countSpan.textContent = count;
            colorItemDiv.appendChild(countSpan);

            pathColorBreakdownDiv.appendChild(colorItemDiv);
        }

        // 使用预计算的颜色序列
        pathColorSequenceDiv.innerHTML = ''; // 清空之前的显示
        const colorSequence = currentPathData.colorSequence;
        
        colorSequence.forEach(item => {
            const colorSequenceItemDiv = document.createElement('div');
            colorSequenceItemDiv.className = 'color-sequence-item';
            
            // 检查是否为当前选中的项
            const isSelected = highlightedSequenceItem === item;
            if (isSelected) {
                colorSequenceItemDiv.classList.add('selected');
            }

            // 添加计数显示
            const countSpan = document.createElement('span');
            countSpan.className = 'sequence-count';
            countSpan.textContent = item.count;
            colorSequenceItemDiv.appendChild(countSpan);

            const colorSwatchDiv = document.createElement('div');
            colorSwatchDiv.className = 'color-swatch';
            colorSwatchDiv.style.backgroundColor = item.color;
            colorSequenceItemDiv.appendChild(colorSwatchDiv);

            // 添加点击事件监听器
            colorSequenceItemDiv.addEventListener('click', () => {
                if (isDiagonalMode) {
                    if (highlightedSequenceItem === item) {
                        highlightedSequenceItem = null; // 取消高亮
                        highlightedSequenceRange = null; // 清除索引范围
                        colorSequenceItemDiv.classList.remove('selected'); // 移除选中状态
                    } else {
                        // 移除之前选中项的样式
                        const previousSelected = pathColorSequenceDiv.querySelector('.color-sequence-item.selected');
                        if (previousSelected) {
                            previousSelected.classList.remove('selected');
                        }
                        
                        highlightedSequenceItem = item; // 设置新的高亮项
                        highlightedSequenceRange = { 
                            startIndex: item.startIndex, 
                            endIndex: item.endIndex 
                        }; // 存储索引范围
                        colorSequenceItemDiv.classList.add('selected'); // 添加选中状态
                    }
                    drawMainGrid(); // 重新绘制主网格以应用高亮
                }
            });

            pathColorSequenceDiv.appendChild(colorSequenceItemDiv);
        });
    }

    // 将编织进度保存到 localStorage（按地图名称）
    function saveProgress() {
        if (!currentMapName) return;
        saveProgressForMap(currentMapName);
    }

    // 为指定地图保存进度
    function saveProgressForMap(mapName) {
        if (!mapName) {
            console.error('保存进度失败：地图名称为空');
            return false;
        }
        
        // 检查是否完成，如果完成则设置确认完成状态
        const allCompleted = checkIfAllCompleted();
        if (allCompleted) {
            isConfirmedCompleted = true;
            console.log(`[saveProgressForMap] 地图 ${mapName} 已完成，设置确认完成状态`);
        }
        
        const progressState = {
            completedSquares: completedSquares,
            currentDiagonalIndex: currentDiagonalIndex,
            isDiagonalMode: isDiagonalMode,
            isConfirmedCompleted: isConfirmedCompleted // 保存确认完成状态
        };
        
        try {
            const storageKey = `weavingProgressState_${mapName}`;
            localStorage.setItem(storageKey, JSON.stringify(progressState));
            console.log(`地图 ${mapName} 的编织进度和模式状态已保存。`, {
                completedSquares: completedSquares.length,
                currentDiagonalIndex,
                isDiagonalMode,
                isConfirmedCompleted
            });
            updateLoadButtonState(); // 更新按钮状态
            updateMapItemMiniMap(mapName); // 更新该地图的小地图显示
            updateNavigationButtons(); // 更新导航按钮状态（如果已确认完成，会禁用按钮）
            return true;
        } catch (e) {
            console.error(`保存地图 ${mapName} 的编织进度失败:`, e);
            alert(`保存进度失败：${e.message}`);
            return false;
        }
    }

    // 更新标记完成按钮的状态
    function updateMarkCompletedButtonState() {
        if (currentPath.length > 0) {
            markCompletedBtn.disabled = false;
        } else {
            markCompletedBtn.disabled = true;
        }
    }
    
    // 更新模式切换的 UI 状态
    function updateModeUI() {
        if (isDiagonalMode) {
            // 对角模式：更新按钮文本、禁用手动选择、显示横条等
            enterDiagonalModeBtn.textContent = "退出对角预设模式";
            enterDiagonalModeBtn.classList.add('active'); // 添加激活状态样式
            toggleManualPathSelection(false); // 禁用手动选择
            markCompletedBtn.textContent = "标记当前对角线为已完成";
            pathColorSequenceContainer.classList.remove('hidden'); // 显示编织顺序横条
            updateNavigationButtons(); // 更新导航按钮状态
        } else {
            // 普通模式：更新按钮文本、启用手动选择、隐藏横条等
            enterDiagonalModeBtn.textContent = "进入对角预设模式";
            enterDiagonalModeBtn.classList.remove('active'); // 移除激活状态样式
            toggleManualPathSelection(true); // 启用手动选择
            markCompletedBtn.textContent = "标记为已完成";
            pathColorSequenceContainer.classList.add('hidden'); // 隐藏编织顺序横条
            currentPathStatusDiv.style.display = 'none'; // 隐藏路径状态
            updateNavigationButtons(); // 更新导航按钮状态（禁用）
        }
        
        // 更新标记完成按钮的状态
        updateMarkCompletedButtonState();
    }

    // 从 localStorage 加载编织进度（当前地图）
    function loadProgress() {
        if (!currentMapName) return;
        loadProgressForMap(currentMapName);
    }

    // 为指定地图加载进度
    function loadProgressForMap(mapName) {
        const storageKey = `weavingProgressState_${mapName}`;
        const savedProgressState = localStorage.getItem(storageKey);
        
        if (savedProgressState) {
            try {
                const progressState = JSON.parse(savedProgressState);
                completedSquares = progressState.completedSquares || [];
                // 同步更新 Set
                completedSquaresSet = new Set(completedSquares.map(sq => `${sq.row},${sq.col}`));
                currentDiagonalIndex = progressState.currentDiagonalIndex !== undefined ? progressState.currentDiagonalIndex : -1;
                isDiagonalMode = progressState.isDiagonalMode !== undefined ? progressState.isDiagonalMode : false;
                isConfirmedCompleted = progressState.isConfirmedCompleted !== undefined ? progressState.isConfirmedCompleted : false;
                console.log(`地图 ${mapName} 的编织进度和模式状态已加载。`, {
                    completedSquares: completedSquares.length,
                    currentDiagonalIndex,
                    isDiagonalMode,
                    isConfirmedCompleted
                });
            } catch (e) {
                console.error(`加载地图 ${mapName} 的编织进度失败，JSON 解析错误:`, e);
                alert(`加载进度失败：数据格式错误。错误信息：${e.message}`);
                completedSquares = []; // 解析失败则清空
                completedSquaresSet = new Set(); // 同步清空 Set
                currentDiagonalIndex = -1;
                isDiagonalMode = false;
                isConfirmedCompleted = false;
            }
        } else {
            console.log(`地图 ${mapName} 没有找到保存的编织进度和模式状态。`);
            completedSquares = [];
            completedSquaresSet = new Set(); // 同步清空 Set
            currentDiagonalIndex = -1;
            isDiagonalMode = false;
            isConfirmedCompleted = false;
        }

        // 根据加载的状态初始化路径数据
        // 确保 diagonalPaths 已经计算完成
        if (isDiagonalMode) {
            if (diagonalPaths.length === 0) {
                console.warn(`加载进度时，diagonalPaths 尚未计算完成。正在重新计算...`);
                calculateAllDiagonalPaths();
            }
            
            if (currentDiagonalIndex !== -1 && diagonalPaths.length > 0) {
                // 验证索引是否有效
                if (currentDiagonalIndex >= 0 && currentDiagonalIndex < diagonalPaths.length) {
                    const diagonalData = diagonalPaths[currentDiagonalIndex];
                    currentPath = diagonalData.path; // 使用预计算的路径
                    currentPathSet = new Set(currentPath.map(coord => `${coord.row},${coord.col}`));
                    currentPathIndexMap = new Map();
                    currentPath.forEach((coord, index) => {
                        currentPathIndexMap.set(`${coord.row},${coord.col}`, index);
                    });
                    currentPathData = diagonalData; // 存储预计算的数据
                    console.log(`成功恢复对角模式路径，索引：${currentDiagonalIndex}，路径长度：${currentPath.length}`);
                } else {
                    console.warn(`加载的 currentDiagonalIndex (${currentDiagonalIndex}) 超出有效范围 (0-${diagonalPaths.length - 1})，重置为普通模式。`);
                    currentPath = [];
                    currentPathSet = new Set();
                    currentPathIndexMap = new Map();
                    currentPathData = null;
                    currentDiagonalIndex = -1;
                    isDiagonalMode = false;
                }
            } else {
                console.warn(`无法恢复对角模式：currentDiagonalIndex=${currentDiagonalIndex}, diagonalPaths.length=${diagonalPaths.length}`);
                currentPath = [];
                currentPathSet = new Set();
                currentPathIndexMap = new Map();
                currentPathData = null;
                currentDiagonalIndex = -1;
                isDiagonalMode = false; // 如果索引无效，退出对角模式
            }
        } else {
            currentPath = []; // 加载普通模式时，清空路径
            currentPathSet = new Set();
            currentPathIndexMap = new Map();
            currentPathData = null;
            startPoint = null;
            endPoint = null;
        }

        // 使用统一的函数更新模式 UI
        updateModeUI();

        // 在加载进度后立即更新 UI
        drawMainGrid();
        updateProgressBar();
        updateCurrentPathStatus();
        updateLoadButtonState(); // 更新按钮状态
        updateNavigationButtons(); // 更新导航按钮状态（如果已确认完成，会禁用按钮）
    }

    // 更新加载按钮状态
    function updateLoadButtonState() {
        if (!currentMapName) {
            loadProgressBtn.disabled = true;
            return;
        }
        
        const storageKey = `weavingProgressState_${currentMapName}`;
        const savedProgressState = localStorage.getItem(storageKey);
        
        // 如果没有保存的数据，禁用按钮
        if (!savedProgressState) {
            loadProgressBtn.disabled = true;
            return;
        }
        
        // 检查保存的进度是否有效（有完成的方块或处于对角模式）
        try {
            const progressState = JSON.parse(savedProgressState);
            const hasCompletedSquares = progressState.completedSquares && progressState.completedSquares.length > 0;
            const isInDiagonalMode = progressState.isDiagonalMode === true;
            
            // 只有当有完成的方块或处于对角模式时，才启用按钮
            // 如果进度为空（没有完成的方块且不是对角模式），禁用按钮
            loadProgressBtn.disabled = !(hasCompletedSquares || isInDiagonalMode);
        } catch (e) {
            // 如果解析失败，禁用按钮
            console.error('解析保存的进度状态失败:', e);
            loadProgressBtn.disabled = true;
        }
    }

    // 处理“下一条”按钮的逻辑，包括标记完成和前进
    function handleNextDiagonal() {
        console.log(`[handleNextDiagonal] start. currentDiagonalIndex: ${currentDiagonalIndex}, isDiagonalMode: ${isDiagonalMode}`);
        if (!isDiagonalMode) return;

        if (currentDiagonalIndex < diagonalPaths.length - 1) {
            markPathSquaresAsCompleted(currentPath); // 标记当前对角线为已完成
            setDiagonalPath(currentDiagonalIndex + 1); // 前进到下一条对角线
            updateProgressBar(); // 更新进度条
            updateCurrentPathStatus(); // 更新路径状态显示
        } else if (currentDiagonalIndex === diagonalPaths.length - 1) {
            // 如果是最后一条，标记完成，然后清除路径，提示完成
            markPathSquaresAsCompleted(currentPath); 
            currentPath = []; 
            currentPathSet = new Set();
            currentPathData = null;
            currentDiagonalIndex = -1;
            isDiagonalMode = false; // 退出对角模式
            
            // 使用统一的函数更新模式 UI
            updateModeUI();
            
            drawMainGrid();
            updateProgressBar(); // 更新进度条
            updateCurrentPathStatus(); // 更新路径状态显示
            alert("所有对角线已完成！");
        }
        console.log(`[handleNextDiagonal] end. currentDiagonalIndex: ${currentDiagonalIndex}`);
    }

    // 处理“上一条”按钮的逻辑，只导航
    function handlePrevDiagonal() {
        console.log(`[handlePrevDiagonal] start. currentDiagonalIndex: ${currentDiagonalIndex}, isDiagonalMode: ${isDiagonalMode}`);
        if (!isDiagonalMode || currentDiagonalIndex <= 0) return; // 如果不是对角模式或已经是第一条，则不执行

        // 撤销将要导航到的前一条对角线的“已完成”状态
        const pathToUndoCompletion = diagonalPaths[currentDiagonalIndex - 1].path;
        const pathToUndoSet = new Set(pathToUndoCompletion.map(coord => `${coord.row},${coord.col}`));
        
        // 从数组中移除，同时更新 Set（使用 Set 优化）
        completedSquares = completedSquares.filter(sq => {
            const coordKey = `${sq.row},${sq.col}`;
            if (pathToUndoSet.has(coordKey)) {
                completedSquaresSet.delete(coordKey); // 从 Set 中删除
                return false; // 从数组中移除
            }
            return true;
        });
        
        // 旧的 filter 代码（已替换为上面的 Set 优化版本）
        /* completedSquares = completedSquares.filter(sq => 
            !pathToUndoCompletion.some(pSq => pSq.row === sq.row && pSq.col === sq.col)
        ); */

        setDiagonalPath(currentDiagonalIndex - 1); // 导航到上一条对角线
        updateProgressBar(); // 更新进度条
        console.log(`[handlePrevDiagonal] end. currentDiagonalIndex: ${currentDiagonalIndex}`);
    }

    enterDiagonalModeBtn.addEventListener('click', () => {
        if (isDiagonalMode) {
            // 退出对角模式
            isDiagonalMode = false;
            currentPath = [];
            currentPathSet = new Set();
            currentPathIndexMap = new Map();
            currentPathData = null;
            currentDiagonalIndex = -1;
            startPoint = null;
            endPoint = null;
            highlightedSequenceItem = null;
            highlightedSequenceRange = null;
            
            // 清除颜色序列的选中状态
            clearSequenceSelection();
            
            // 使用统一的函数更新模式 UI
            updateModeUI();
            
            // 更新 UI
            drawMainGrid();
            updateProgressBar();
            updateCurrentPathStatus();
        } else {
            // 进入对角模式
            isDiagonalMode = true;
            currentDiagonalIndex = 0;
            setDiagonalPath(currentDiagonalIndex);
            
            // 使用统一的函数更新模式 UI
            updateModeUI();
            
            // 更新 UI
            updateProgressBar();
            updateCurrentPathStatus();
        }
    });

    // “上一条”按钮的长按和点击事件
    prevDiagonalBtn.addEventListener('mousedown', (event) => {
        if (event.button === 0 && !prevDiagonalBtn.disabled) { // 鼠标左键按下且按钮未禁用
            handlePrevDiagonal(); // 立即执行一次
            longPressTimer = setInterval(handlePrevDiagonal, REPEAT_INTERVAL);
        }
    });
    prevDiagonalBtn.addEventListener('mouseup', () => clearInterval(longPressTimer));
    prevDiagonalBtn.addEventListener('mouseleave', () => clearInterval(longPressTimer));

    // “下一条”按钮的长按和点击事件
    nextDiagonalBtn.addEventListener('mousedown', (event) => {
        if (event.button === 0 && !nextDiagonalBtn.disabled) { // 鼠标左键按下且按钮未禁用
            handleNextDiagonal(); // 立即执行一次
            longPressTimer = setInterval(handleNextDiagonal, REPEAT_INTERVAL);
        }
    });
    nextDiagonalBtn.addEventListener('mouseup', () => clearInterval(longPressTimer));
    nextDiagonalBtn.addEventListener('mouseleave', () => clearInterval(longPressTimer));

    // 处理完成当前编织的逻辑
    function handleCompleteCurrentWeaving() {
        console.log(`[handleCompleteCurrentWeaving] called. isDiagonalMode: ${isDiagonalMode}, currentPath length: ${currentPath.length}`);
        
        if (currentPath.length === 0) {
            // 如果没有当前路径，提示用户
            console.log('没有当前路径可完成');
            return;
        }
        
        // 标记当前路径为已完成
        markPathSquaresAsCompleted(currentPath);
        
        if (isDiagonalMode) {
            // 对角模式：标记完成后自动前进到下一条
            const currentIndex = currentDiagonalIndex; // 保存当前索引
            
            if (currentIndex < diagonalPaths.length - 1) {
                // 还有下一条，自动前进
                // setDiagonalPath 内部已经会调用 updateCurrentPathStatus 和 drawMainGrid
                setDiagonalPath(currentIndex + 1);
                updateProgressBar(); // 更新进度条
                console.log(`已完成第 ${currentIndex} 条对角线，自动前进到第 ${currentIndex + 1} 条`);
            } else if (currentIndex === diagonalPaths.length - 1) {
                // 这是最后一条，标记完成后退出对角模式
                currentPath = [];
                currentPathSet = new Set();
                currentPathIndexMap = new Map();
                currentPathData = null;
                currentDiagonalIndex = -1;
                isDiagonalMode = false;
                
                // 更新模式 UI（内部会调用 updateMarkCompletedButtonState）
                updateModeUI();
                
                drawMainGrid();
                updateProgressBar();
                updateCurrentPathStatus();
                
                // 提示完成
                alert("🎉 所有对角线已完成！");
                console.log('所有对角线已完成');
            }
        } else {
            // 普通模式：标记完成后清空当前路径
            currentPath = [];
            currentPathSet = new Set();
            currentPathIndexMap = new Map();
            currentPathData = null;
            startPoint = null;
            endPoint = null;
            
            drawMainGrid();
            updateProgressBar();
            updateCurrentPathStatus();
            updateMarkCompletedButtonState(); // 更新标记完成按钮状态
            
            console.log('普通模式：当前路径已标记为已完成');
        }
        
        // 自动保存进度
        if (currentMapName) {
            saveProgressForMap(currentMapName);
        }
    }
    
    // 标记完成按钮的长按和点击事件
    let markCompletedLongPressTimer = null;
    let markCompletedClickTimer = null;
    
    markCompletedBtn.addEventListener('mousedown', (event) => {
        if (event.button === 0 && !markCompletedBtn.disabled) { // 鼠标左键按下且按钮未禁用
            if (currentPath.length === 0) {
                return; // 没有路径可完成
            }
            
            // 添加按下状态的视觉反馈
            markCompletedBtn.classList.add('pressing');
            
            // 清除之前的定时器（防止重复触发）
            if (markCompletedClickTimer) {
                clearTimeout(markCompletedClickTimer);
                markCompletedClickTimer = null;
            }
            
            // 设置长按定时器
            markCompletedLongPressTimer = setTimeout(() => {
                // 长按触发：完成当前编织
                markCompletedBtn.classList.remove('pressing');
                markCompletedBtn.classList.add('completing');
                handleCompleteCurrentWeaving();
                
                // 完成动画后保持绿色状态一段时间，然后恢复
                setTimeout(() => {
                    markCompletedBtn.classList.remove('completing');
                }, 1200); // 动画0.8s + 保持0.4s = 1.2s
                
                markCompletedLongPressTimer = null;
            }, LONG_PRESS_DELAY);
        }
    });
    
    markCompletedBtn.addEventListener('mouseup', () => {
        // 移除按下状态
        markCompletedBtn.classList.remove('pressing');
        
        if (markCompletedLongPressTimer) {
            // 如果长按定时器还在，说明是短按，取消长按
            clearTimeout(markCompletedLongPressTimer);
            markCompletedLongPressTimer = null;
            
            // 短按也触发完成（保持向后兼容）
            if (currentPath.length > 0) {
                markCompletedBtn.classList.add('completing');
                handleCompleteCurrentWeaving();
                
                // 完成动画后保持绿色状态一段时间，然后恢复
                setTimeout(() => {
                    markCompletedBtn.classList.remove('completing');
                }, 1200); // 动画0.8s + 保持0.4s = 1.2s
            }
        }
    });
    
    markCompletedBtn.addEventListener('mouseleave', () => {
        // 移除按下状态
        markCompletedBtn.classList.remove('pressing');
        
        // 鼠标离开时取消长按
        if (markCompletedLongPressTimer) {
            clearTimeout(markCompletedLongPressTimer);
            markCompletedLongPressTimer = null;
        }
    });

    saveProgressBtn.addEventListener('click', () => {
        if (!currentMapName) {
            alert("无法保存进度：未选择地图！");
            return;
        }
        
        const success = saveProgress();
        if (success) {
            // 使用更友好的提示
            const completedCount = completedSquares.length;
            const totalSquares = pixelMapData ? pixelMapData.grid_data.length * pixelMapData.grid_data[0].length : 0;
            const progressPercent = totalSquares > 0 ? Math.round((completedCount / totalSquares) * 100) : 0;
            alert(`✅ 进度已保存！\n已完成：${completedCount} 个方块 (${progressPercent}%)`);
        }
    });

    loadProgressBtn.addEventListener('click', () => {
        if (loadProgressBtn.disabled) {
            alert("⚠️ 当前地图没有保存的进度！");
            return;
        }
        
        if (!currentMapName) {
            alert("无法加载进度：未选择地图！");
            return;
        }
        
        // 确认是否要覆盖当前进度
        if (completedSquares.length > 0) {
            if (!confirm("⚠️ 加载进度将覆盖当前的编织进度，是否继续？")) {
                return;
            }
        }
        
        loadProgress();
        
        // 显示加载结果
        const completedCount = completedSquares.length;
        const totalSquares = pixelMapData ? pixelMapData.grid_data.length * pixelMapData.grid_data[0].length : 0;
        const progressPercent = totalSquares > 0 ? Math.round((completedCount / totalSquares) * 100) : 0;
        const modeText = isDiagonalMode ? `对角模式 (第 ${currentDiagonalIndex + 1} 条)` : '普通模式';
        alert(`✅ 进度已加载！\n已完成：${completedCount} 个方块 (${progressPercent}%)\n模式：${modeText}`);
    });

    resetProgressBtn.addEventListener('click', () => {
        if (!currentMapName) {
            alert("无法重置进度：未选择地图！");
            return;
        }
        
        if (confirm("⚠️ 确定要重置所有编织进度吗？\n这将清除所有已完成的方块和当前模式状态。\n此操作不可撤销！")) {
            completedSquares = [];
            completedSquaresSet = new Set(); // 同步清空 Set
            currentDiagonalIndex = -1;
            isDiagonalMode = false;
            isConfirmedCompleted = false; // 清除确认完成状态
            currentPath = [];
            currentPathSet = new Set();
            currentPathData = null;
            startPoint = null;
            endPoint = null;

            // 使用统一的函数更新模式 UI
            updateModeUI();
            
            drawMainGrid();
            updateProgressBar();
            updateCurrentPathStatus();
            
            // 保存重置后的状态（清空进度）
            const success = saveProgress();
            // 重置后，按钮状态应该更新为禁用（因为没有有效进度）
            updateLoadButtonState();
            updateNavigationButtons(); // 更新导航按钮状态（重置后应该恢复可用）
            
            if (success) {
                alert("✅ 编织进度已重置！");
            } else {
                alert("⚠️ 进度已重置，但保存失败。请手动保存。");
            }
        }
    });

    // 更新进度条显示
    function updateProgressBar() {
        if (!pixelMapData) return;

        const totalSquares = pixelMapData.grid_data.length * pixelMapData.grid_data[0].length;
        const completedCount = completedSquares.length;
        const progress = (completedCount / totalSquares) * 100;

        progressBarFill.style.width = `${progress.toFixed(2)}%`;
        progressText.textContent = `${progress.toFixed(0)}%`;
    }

    // 辅助函数：将路径上的方块标记为已完成（使用 Set 优化）
    function markPathSquaresAsCompleted(path) {
        // 性能优化：批量添加，减少数组操作
        const newSquares = [];
        path.forEach(pathCoord => {
            const coordKey = `${pathCoord.row},${pathCoord.col}`;
            if (!completedSquaresSet.has(coordKey)) {
                newSquares.push(pathCoord);
                completedSquaresSet.add(coordKey); // 同步更新 Set
            }
        });
        // 批量添加到数组
        if (newSquares.length > 0) {
            completedSquares.push(...newSquares);
            // 如果之前已确认完成，但现在又添加了新的方块，清除确认完成状态
            // 因为用户可能继续编织（虽然理论上不应该发生，但为了安全起见）
            if (isConfirmedCompleted) {
                isConfirmedCompleted = false;
                console.log(`[markPathSquaresAsCompleted] 检测到新的方块，清除确认完成状态`);
                updateNavigationButtons(); // 更新导航按钮状态
            }
            // 标记完成后，更新当前地图的小地图显示
            if (currentMapName) {
                updateMapItemMiniMap(currentMapName);
            }
        }
    }

    // 更新地图列表中指定地图的小地图（显示完成状态）
    function updateMapItemMiniMap(mapName) {
        if (!mapName || !mapsDataCache[mapName]) return;
        
        // 性能优化：使用缓存获取DOM元素
        let mapItemData = mapItemCache.get(mapName);
        if (!mapItemData) {
            const mapItem = document.querySelector(`.map-item[data-map-name="${mapName}"]`);
            if (!mapItem) return;
            const canvas = mapItem.querySelector('.map-item-canvas');
            if (!canvas) return;
            mapItemData = { mapItem, canvas };
            mapItemCache.set(mapName, mapItemData);
        }
        
        const { canvas } = mapItemData;
        
        // 获取该地图的保存状态
        const storageKey = `weavingProgressState_${mapName}`;
        const savedProgressState = localStorage.getItem(storageKey);
        let completedSquaresForMap = [];
        
        if (savedProgressState) {
            try {
                const progressState = JSON.parse(savedProgressState);
                completedSquaresForMap = progressState.completedSquares || [];
            } catch (e) {
                console.error(`加载地图 ${mapName} 的状态失败:`, e);
            }
        }
        
        const completedSet = new Set(completedSquaresForMap.map(sq => `${sq.row},${sq.col}`));
        drawMiniMapForItem(canvas, mapsDataCache[mapName], completedSet);
        
        // 更新统计信息
        updateMapItemStats(mapName, mapsDataCache[mapName]);
    }
    
    // 计算并更新地图项的色块统计信息
    function updateMapItemStats(mapName, data) {
        if (!mapName || !data) return;
        
        // 获取统计容器
        let mapItemData = mapItemCache.get(mapName);
        if (!mapItemData || !mapItemData.statsContainer) {
            // 如果缓存中没有，尝试查找
            const mapItem = document.querySelector(`.map-item[data-map-name="${mapName}"]`);
            if (!mapItem) return;
            const statsContainer = mapItem.querySelector('.map-item-stats');
            if (!statsContainer) return;
            if (!mapItemData) {
                mapItemData = {};
            }
            mapItemData.statsContainer = statsContainer;
            mapItemCache.set(mapName, mapItemData);
        }
        
        const statsContainer = mapItemData.statsContainer;
        
        // 统计颜色数量
        const gridData = data.grid_data;
        const colorMap = data.color_map;
        const colorCount = new Map(); // {colorKey: count}
        
        for (let r_idx = 0; r_idx < gridData.length; r_idx++) {
            for (let c_idx = 0; c_idx < gridData[r_idx].length; c_idx++) {
                const pixelValue = gridData[r_idx][c_idx];
                const colorKey = String(pixelValue);
                colorCount.set(colorKey, (colorCount.get(colorKey) || 0) + 1);
            }
        }
        
        // 转换为数组并按数量排序（降序）
        const allColorStats = Array.from(colorCount.entries())
            .map(([colorKey, count]) => ({
                colorKey,
                color: colorMap[colorKey],
                count
            }))
            .sort((a, b) => b.count - a.count);
        
        // 清空并重新填充统计容器
        statsContainer.innerHTML = '';
        
        if (allColorStats.length === 0) {
            return;
        }
        
        // 创建主要统计项容器（前6种，显示数字）
        const primaryStatsContainer = document.createElement('div');
        primaryStatsContainer.className = 'map-stat-primary';
        
        // 创建次要统计项容器（其他颜色，不显示数字）
        const secondaryStatsContainer = document.createElement('div');
        secondaryStatsContainer.className = 'map-stat-secondary';
        
        // 创建统计项（前6种显示数字）
        allColorStats.slice(0, 6).forEach(({ color, count }) => {
            const statItem = createStatItem(color, count, true);
            primaryStatsContainer.appendChild(statItem);
        });
        
        // 创建统计项（其他颜色不显示数字）
        allColorStats.slice(6).forEach(({ color, count }) => {
            const statItem = createStatItem(color, count, false);
            secondaryStatsContainer.appendChild(statItem);
        });
        
        statsContainer.appendChild(primaryStatsContainer);
        if (allColorStats.length > 6) {
            statsContainer.appendChild(secondaryStatsContainer);
        }
    }
    
    // 创建统计项的辅助函数
    function createStatItem(color, count, showCount) {
        const statItem = document.createElement('div');
        statItem.className = 'map-stat-item';
        if (!showCount) {
            statItem.classList.add('map-stat-item-no-count');
        }
        
        const colorSwatch = document.createElement('div');
        colorSwatch.className = 'map-stat-swatch';
        if (color) {
            colorSwatch.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        } else {
            colorSwatch.style.backgroundColor = '#000000';
        }
        
        statItem.appendChild(colorSwatch);
        
        // 只在需要时显示数字
        if (showCount) {
            const countText = document.createElement('span');
            countText.className = 'map-stat-count';
            countText.textContent = count;
            statItem.appendChild(countText);
        }
        
        // 创建悬停提示工具（追加到body，避免被overflow截断）
        const tooltip = document.createElement('div');
        tooltip.className = 'map-stat-tooltip';
        if (color) {
            tooltip.textContent = `RGB(${color[0]}, ${color[1]}, ${color[2]}) | 数量: ${count}`;
        } else {
            tooltip.textContent = `RGB(0, 0, 0) | 数量: ${count}`;
        }
        document.body.appendChild(tooltip);
        
        // 添加鼠标事件，显示/隐藏提示工具
        statItem.addEventListener('mouseenter', (e) => {
            const rect = statItem.getBoundingClientRect();
            tooltip.style.left = `${rect.left + rect.width / 2}px`;
            tooltip.style.top = `${rect.top - 10}px`;
            tooltip.style.transform = 'translateX(-50%) translateY(-100%)';
            tooltip.style.opacity = '1';
            tooltip.style.visibility = 'visible';
        });
        
        statItem.addEventListener('mouseleave', () => {
            tooltip.style.opacity = '0';
            tooltip.style.visibility = 'hidden';
        });
        
        return statItem;
    }

    // 更新所有地图列表项的小地图
    function updateAllMapItemsMiniMap() {
        mapsList.forEach(map => {
            updateMapItemMiniMap(map.name);
        });
    }
    
    // 性能优化：初始化离屏Canvas
    function initStaticGridCanvas() {
        if (!pixelMapData) return;
        
        const gridData = pixelMapData.grid_data;
        const height = gridData.length;
        const width = gridData[0].length;
        
        // 创建离屏Canvas
        staticGridCanvas = document.createElement('canvas');
        staticGridCanvas.width = width * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
        staticGridCanvas.height = height * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
        staticGridCtx = staticGridCanvas.getContext('2d');
        staticGridDirty = true;
    }
    
    // 性能优化：绘制静态网格到离屏Canvas（只绘制基础网格，不包含动态状态）
    function drawStaticGrid() {
        if (!pixelMapData || !staticGridCanvas || !staticGridCtx) return;
        if (!staticGridDirty) return; // 如果静态网格没有变化，不需要重绘
        
        const gridData = pixelMapData.grid_data;
        const colorMap = pixelMapData.color_map;
        const height = gridData.length;
        const width = gridData[0].length;
        
        // 清除并绘制背景
        staticGridCtx.clearRect(0, 0, staticGridCanvas.width, staticGridCanvas.height);
        staticGridCtx.fillStyle = 'white';
        staticGridCtx.fillRect(0, 0, staticGridCanvas.width, staticGridCanvas.height);
        
        // 只绘制基础网格（未完成状态的方块）
        for (let r_idx = 0; r_idx < height; r_idx++) {
            for (let c_idx = 0; c_idx < width; c_idx++) {
                const pixelValue = gridData[r_idx][c_idx];
                const originalColor = colorMap[String(pixelValue)];
                
                const start_x = c_idx * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
                const start_y = r_idx * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
                
                if (originalColor) {
                    // 绘制柔和的半透明色块（未完成状态）
                    const softenedR = originalColor[0] * 0.7 + 128 * 0.3;
                    const softenedG = originalColor[1] * 0.7 + 128 * 0.3;
                    const softenedB = originalColor[2] * 0.7 + 128 * 0.3;
                    
                    staticGridCtx.fillStyle = `rgba(${Math.round(softenedR)}, ${Math.round(softenedG)}, ${Math.round(softenedB)}, 0.35)`;
                    staticGridCtx.fillRect(start_x, start_y, SQUARE_SIZE_MAIN, SQUARE_SIZE_MAIN);
                    
                    // 绘制边框
                    staticGridCtx.strokeStyle = 'rgba(128, 128, 128, 0.4)';
                    staticGridCtx.lineWidth = BORDER_WIDTH_MAIN;
                    staticGridCtx.strokeRect(start_x, start_y, SQUARE_SIZE_MAIN, SQUARE_SIZE_MAIN);
                }
            }
        }
        
        staticGridDirty = false; // 标记为已绘制
    }

    function drawMainGrid() {
        if (!pixelMapData) return;

        const gridData = pixelMapData.grid_data;
        const colorMap = pixelMapData.color_map;

        const height = gridData.length;
        const width = gridData[0].length;

        // 调整 canvas 尺寸以包含所有方块和边框
        mainGridCanvas.width = width * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
        mainGridCanvas.height = height * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
        
        // 性能优化：如果离屏Canvas尺寸不匹配，重新初始化
        if (!staticGridCanvas || staticGridCanvas.width !== mainGridCanvas.width || staticGridCanvas.height !== mainGridCanvas.height) {
            initStaticGridCanvas();
        }
        
        // 性能优化：先绘制静态网格（从离屏Canvas复制）
        drawStaticGrid();
        mainGridCtx.clearRect(0, 0, mainGridCanvas.width, mainGridCanvas.height);
        mainGridCtx.drawImage(staticGridCanvas, 0, 0);
        
        // 性能优化：只绘制动态部分（已完成方块和路径高亮）
        // 使用更高效的方式：只遍历需要更新的方块
        const squaresToDraw = new Set();
        
        // 收集需要绘制的方块（已完成和路径上的）
        completedSquaresSet.forEach(coordKey => {
            squaresToDraw.add(coordKey);
        });
        currentPathSet.forEach(coordKey => {
            squaresToDraw.add(coordKey);
        });
        
        // 批量绘制已完成的方块和路径高亮
        for (const coordKey of squaresToDraw) {
            const [r_idx, c_idx] = coordKey.split(',').map(Number);
            if (r_idx < 0 || r_idx >= height || c_idx < 0 || c_idx >= width) continue;
            
            const pixelValue = gridData[r_idx][c_idx];
            const originalColor = colorMap[String(pixelValue)];
            if (!originalColor) continue;
            
            const start_x = c_idx * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
            const start_y = r_idx * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
            
            const isCompleted = completedSquaresSet.has(coordKey);
            const isOnPath = currentPathSet.has(coordKey);
            
            // 检查当前方块是否在被选中的序列范围内
            let isHighlightedBySequence = false;
            if (highlightedSequenceRange && isOnPath) {
                const pathIndex = currentPathIndexMap.get(coordKey);
                if (pathIndex !== undefined && pathIndex >= highlightedSequenceRange.startIndex && pathIndex <= highlightedSequenceRange.endIndex) {
                    isHighlightedBySequence = true;
                }
            }

            if (isCompleted) {
                    // 已完成方块: 增强视觉效果
                    // 1. 轻微提高颜色饱和度（增强10%）
                    const enhancedR = Math.min(255, originalColor[0] * 1.1);
                    const enhancedG = Math.min(255, originalColor[1] * 1.1);
                    const enhancedB = Math.min(255, originalColor[2] * 1.1);
                    
                    // 2. 绘制主色块
                    mainGridCtx.fillStyle = `rgb(${Math.round(enhancedR)}, ${Math.round(enhancedG)}, ${Math.round(enhancedB)})`;
                    mainGridCtx.fillRect(start_x, start_y, SQUARE_SIZE_MAIN, SQUARE_SIZE_MAIN);
                    
                    // 3. 绘制底部和右侧内阴影（模拟立体感，光源在左上）
                    const shadowSize = 2;
                    const shadowAlpha = 0.2;
                    mainGridCtx.fillStyle = `rgba(0, 0, 0, ${shadowAlpha})`;
                    // 底部阴影
                    mainGridCtx.fillRect(
                        start_x, 
                        start_y + SQUARE_SIZE_MAIN - shadowSize, 
                        SQUARE_SIZE_MAIN, 
                        shadowSize
                    );
                    // 右侧阴影
                    mainGridCtx.fillRect(
                        start_x + SQUARE_SIZE_MAIN - shadowSize, 
                        start_y, 
                        shadowSize, 
                        SQUARE_SIZE_MAIN
                    );
                    
                    // 4. 绘制顶部高光（模拟光照效果，光源在左上）
                    const highlightSize = Math.max(2, SQUARE_SIZE_MAIN / 3);
                    mainGridCtx.fillStyle = 'rgba(255, 255, 255, 0.25)';
                    mainGridCtx.fillRect(start_x + 1, start_y + 1, highlightSize, highlightSize);
                    
                    // 5. 绘制双层边框（内层浅色，外层深色，增加层次感）
                    // 外层深色边框
                    mainGridCtx.strokeStyle = COMPLETED_BORDER_COLOR;
                    mainGridCtx.lineWidth = BORDER_WIDTH_MAIN + 0.5;
                    mainGridCtx.strokeRect(start_x, start_y, SQUARE_SIZE_MAIN, SQUARE_SIZE_MAIN);
                    
                    // 内层浅色边框（稍微缩小）
                    const innerBorderOffset = 1;
                    mainGridCtx.strokeStyle = `rgba(${Math.round(enhancedR * 0.7)}, ${Math.round(enhancedG * 0.7)}, ${Math.round(enhancedB * 0.7)}, 0.5)`;
                    mainGridCtx.lineWidth = 0.5;
                    mainGridCtx.strokeRect(
                        start_x + innerBorderOffset, 
                        start_y + innerBorderOffset, 
                        SQUARE_SIZE_MAIN - innerBorderOffset * 2, 
                        SQUARE_SIZE_MAIN - innerBorderOffset * 2
                    );

            } else if (isOnPath) {
                // 路径高亮方块: 完整颜色，深红色加粗边框
                mainGridCtx.fillStyle = `rgb(${originalColor[0]}, ${originalColor[1]}, ${originalColor[2]})`;
                mainGridCtx.fillRect(start_x, start_y, SQUARE_SIZE_MAIN, SQUARE_SIZE_MAIN);
                
                // 绘制高亮边框
                mainGridCtx.strokeStyle = isHighlightedBySequence ? '#00FFFF' : PATH_HIGHLIGHT_COLOR; // 如果被顺序项高亮，则使用青色
                mainGridCtx.lineWidth = isHighlightedBySequence ? PATH_HIGHLIGHT_WIDTH + 1 : PATH_HIGHLIGHT_WIDTH; // 更粗的边框
                mainGridCtx.strokeRect(start_x, start_y, SQUARE_SIZE_MAIN, SQUARE_SIZE_MAIN);
            }
            // 注意：未完成方块已经在静态网格中绘制，不需要重复绘制
        }
        
        // 绘制悬停方格的高亮（在所有方格绘制完成后，确保在最上层）
        if (hoveredSquare !== null) {
            const hoverRow = hoveredSquare.row;
            const hoverCol = hoveredSquare.col;
            const hoverStartX = hoverCol * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
            const hoverStartY = hoverRow * (SQUARE_SIZE_MAIN + BORDER_WIDTH_MAIN) + BORDER_WIDTH_MAIN;
            
            // 检查悬停方格的状态，决定高亮样式
            const hoverCoordKey = `${hoverRow},${hoverCol}`;
            const isHoverCompleted = completedSquaresSet.has(hoverCoordKey);
            const isHoverOnPath = currentPathSet.has(hoverCoordKey);
            
            // 绘制悬停高亮边框（使用主题色，加粗边框）
            // 如果方格已经在路径中，使用更明显的颜色
            const hoverColor = isHoverOnPath ? '#00D9FF' : HOVER_HIGHLIGHT_COLOR;
            mainGridCtx.strokeStyle = hoverColor;
            mainGridCtx.lineWidth = HOVER_HIGHLIGHT_WIDTH;
            
            // 计算高亮边框的位置（稍微向外扩展，不覆盖原有边框）
            const offset = (HOVER_HIGHLIGHT_WIDTH - BORDER_WIDTH_MAIN) / 2;
            mainGridCtx.strokeRect(
                hoverStartX - offset - 1,
                hoverStartY - offset - 1,
                SQUARE_SIZE_MAIN + offset * 2 + 2,
                SQUARE_SIZE_MAIN + offset * 2 + 2
            );
        }
        
        // 如果放大镜可见，更新放大镜内容
        updateMagnifierIfVisible();
        
        // 注意：不再在每次 drawMainGrid 时更新所有小地图，因为这会严重影响性能
        // 小地图更新应该在特定操作时手动调用（如保存进度、切换地图等）
    }

    // 绘制放大镜内容
    function drawMagnifier(mouseX, mouseY) {
        if (!pixelMapData) return;

        // 获取 canvas 相对于视口的位置（使用 getBoundingClientRect，不受滚动影响）
        const rect = mainGridCanvas.getBoundingClientRect();
        const scaleX = mainGridCanvas.width / rect.width;
        const scaleY = mainGridCanvas.height / rect.height;

        // 获取鼠标在 canvas 中的坐标（考虑缩放）
        // mouseX/Y 是相对于视口的坐标（event.clientX/Y），rect 也是相对于视口的
        // 所以计算是正确的，不受页面滚动影响
        const canvasX = (mouseX - rect.left) * scaleX;
        const canvasY = (mouseY - rect.top) * scaleY;

        // 计算要放大的区域（以鼠标位置为中心）
        const sourceRadius = MAGNIFIER_RADIUS / MAGNIFIER_SCALE;
        let sourceX = canvasX - sourceRadius;
        let sourceY = canvasY - sourceRadius;
        const sourceSize = sourceRadius * 2;

        // 确保源区域不超出 canvas 边界
        sourceX = Math.max(0, Math.min(sourceX, mainGridCanvas.width - sourceSize));
        sourceY = Math.max(0, Math.min(sourceY, mainGridCanvas.height - sourceSize));

        // 清空放大镜 canvas
        magnifierCtx.clearRect(0, 0, MAGNIFIER_SIZE, MAGNIFIER_SIZE);

        // 创建圆形裁剪路径（先创建路径，再填充和绘制）
        magnifierCtx.save();
        magnifierCtx.beginPath();
        magnifierCtx.arc(MAGNIFIER_RADIUS, MAGNIFIER_RADIUS, MAGNIFIER_RADIUS, 0, Math.PI * 2);
        magnifierCtx.clip();

        // 填充白色背景（在裁剪区域内）
        magnifierCtx.fillStyle = '#ffffff';
        magnifierCtx.fillRect(0, 0, MAGNIFIER_SIZE, MAGNIFIER_SIZE);

        // 绘制放大后的区域（确保完全覆盖圆形区域）
        // 使用稍微大一点的绘制区域，确保完全覆盖圆形边缘
        const drawSize = MAGNIFIER_SIZE + 2; // 稍微大一点，确保覆盖边缘
        const drawOffset = -1; // 偏移，使绘制区域居中
        magnifierCtx.drawImage(
            mainGridCanvas,
            sourceX, sourceY, sourceSize, sourceSize, // 源区域
            drawOffset, drawOffset, drawSize, drawSize // 目标区域（稍微大一点）
        );

        magnifierCtx.restore();

        // 绘制中心十字线
        magnifierCtx.strokeStyle = '#ff0000';
        magnifierCtx.lineWidth = 2;
        magnifierCtx.beginPath();
        magnifierCtx.moveTo(MAGNIFIER_RADIUS - 10, MAGNIFIER_RADIUS);
        magnifierCtx.lineTo(MAGNIFIER_RADIUS + 10, MAGNIFIER_RADIUS);
        magnifierCtx.moveTo(MAGNIFIER_RADIUS, MAGNIFIER_RADIUS - 10);
        magnifierCtx.lineTo(MAGNIFIER_RADIUS, MAGNIFIER_RADIUS + 10);
        magnifierCtx.stroke();
    }

    // 放大镜更新节流
    let magnifierUpdateScheduled = false;
    let lastMagnifierUpdateTime = 0;
    const MAGNIFIER_UPDATE_THROTTLE = 16; // 约 60fps (1000ms / 60)

    // 鼠标移动事件处理（放大镜）- 添加节流优化
    function onMagnifierMouseMove(event) {
        const rect = mainGridCanvas.getBoundingClientRect();

        // 检查鼠标是否在主网格 canvas 区域内
        if (event.clientX >= rect.left && event.clientX <= rect.right &&
            event.clientY >= rect.top && event.clientY <= rect.bottom) {
            
            // 显示放大镜
            magnifierContainer.classList.remove('hidden');

            // 节流：限制更新频率
            const now = performance.now();
            if (now - lastMagnifierUpdateTime < MAGNIFIER_UPDATE_THROTTLE) {
                // 如果距离上次更新太近，只更新位置，不更新内容
                if (!magnifierUpdateScheduled) {
                    magnifierUpdateScheduled = true;
                    requestAnimationFrame(() => {
                        updateMagnifierPosition(event);
                        magnifierUpdateScheduled = false;
                    });
                }
                return;
            }
            lastMagnifierUpdateTime = now;

            // 更新放大镜位置和内容
            updateMagnifierPosition(event);
            drawMagnifier(event.clientX, event.clientY);
        } else {
            // 隐藏放大镜
            magnifierContainer.classList.add('hidden');
            lastMouseX = null;
            lastMouseY = null;
        }
    }

    // 更新放大镜位置（不更新内容，用于节流）
    function updateMagnifierPosition(event) {
        const CLOSE_OFFSET = 10;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let preferredLeft = event.clientX + CLOSE_OFFSET;
        let preferredTop = event.clientY - MAGNIFIER_SIZE - CLOSE_OFFSET;

        // 智能位置选择
        let left = preferredLeft;
        let top = preferredTop;
        let positionFound = false;

        if (preferredLeft + MAGNIFIER_SIZE <= viewportWidth && preferredTop >= 0) {
            positionFound = true;
        } else if (preferredLeft + MAGNIFIER_SIZE <= viewportWidth) {
            top = event.clientY + CLOSE_OFFSET;
            if (top + MAGNIFIER_SIZE <= viewportHeight) {
                positionFound = true;
            }
        } else if (preferredTop >= 0) {
            left = event.clientX - MAGNIFIER_SIZE - CLOSE_OFFSET;
            if (left >= 0) {
                positionFound = true;
            }
        } else {
            left = event.clientX - MAGNIFIER_SIZE - CLOSE_OFFSET;
            top = event.clientY + CLOSE_OFFSET;
            if (left >= 0 && top + MAGNIFIER_SIZE <= viewportHeight) {
                positionFound = true;
            }
        }

        // 边界检查
        if (!positionFound) {
            if (left + MAGNIFIER_SIZE > viewportWidth) left = viewportWidth - MAGNIFIER_SIZE;
            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (top + MAGNIFIER_SIZE > viewportHeight) top = viewportHeight - MAGNIFIER_SIZE;
        }

        left = Math.max(0, Math.min(left, viewportWidth - MAGNIFIER_SIZE));
        top = Math.max(0, Math.min(top, viewportHeight - MAGNIFIER_SIZE));

        magnifierContainer.style.left = `${left}px`;
        magnifierContainer.style.top = `${top}px`;
        magnifierContainer.style.position = 'fixed';

        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
    }

    // 鼠标离开事件处理（放大镜）
    function onMagnifierMouseLeave() {
        magnifierContainer.classList.add('hidden');
        lastMouseX = null;
        lastMouseY = null;
    }
    
    // 更新放大镜（如果可见）
    function updateMagnifierIfVisible() {
        if (!magnifierContainer.classList.contains('hidden') && lastMouseX !== null && lastMouseY !== null) {
            drawMagnifier(lastMouseX, lastMouseY);
        }
    }

    // 创建工具提示元素
    const tooltip = document.createElement('div');
    tooltip.id = 'square-tooltip';
    tooltip.className = 'square-tooltip';
    document.body.appendChild(tooltip);

    // 鼠标悬停处理函数（使用节流优化性能）
    function handleSquareHover(event) {
        const coords = getGridCoords(event);
        
        if (coords) {
            // 如果悬停的方格发生变化，更新并重绘
            if (!hoveredSquare || hoveredSquare.row !== coords.row || hoveredSquare.col !== coords.col) {
                hoveredSquare = coords;
                
                // 使用 requestAnimationFrame 节流重绘
                if (!hoverUpdateScheduled) {
                    hoverUpdateScheduled = true;
                    requestAnimationFrame(() => {
                        drawMainGrid(); // 重绘以显示悬停高亮
                        hoverUpdateScheduled = false;
                    });
                }
                
                updateTooltip(coords, event); // 更新工具提示（不需要节流，因为只是更新DOM）
            } else {
                // 同一方格，只更新工具提示位置（跟随鼠标）- 添加节流
                if (!tooltipPositionUpdateScheduled) {
                    tooltipPositionUpdateScheduled = true;
                    requestAnimationFrame(() => {
                        updateTooltipPosition(event);
                        tooltipPositionUpdateScheduled = false;
                    });
                }
            }
        } else {
            // 鼠标不在方格上，清除悬停状态
            if (hoveredSquare !== null) {
                hoveredSquare = null;
                
                // 使用 requestAnimationFrame 节流重绘
                if (!hoverUpdateScheduled) {
                    hoverUpdateScheduled = true;
                    requestAnimationFrame(() => {
                        drawMainGrid(); // 重绘以清除悬停高亮
                        hoverUpdateScheduled = false;
                    });
                }
                
                hideTooltip(); // 隐藏工具提示
            }
        }
    }

    // 更新工具提示位置（不更新内容，只更新位置）
    function updateTooltipPosition(event) {
        if (tooltip.classList.contains('hidden')) return;

        const tooltipOffset = 15;
        let left = event.clientX + tooltipOffset;
        let top = event.clientY + tooltipOffset;

        // 边界检查
        const rect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        if (left + rect.width > viewportWidth) {
            left = event.clientX - rect.width - tooltipOffset;
        }
        if (top + rect.height > viewportHeight) {
            top = event.clientY - rect.height - tooltipOffset;
        }

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    }

    // 更新工具提示内容
    function updateTooltip(coords, event) {
        if (!pixelMapData) return;

        const gridData = pixelMapData.grid_data;
        const colorMap = pixelMapData.color_map;
        const pixelValue = gridData[coords.row][coords.col];
        const color = colorMap[String(pixelValue)];
        const coordKey = `${coords.row},${coords.col}`;
        const isCompleted = completedSquaresSet.has(coordKey);
        const isOnPath = currentPathSet.has(coordKey);

        // 构建工具提示内容
        let tooltipContent = `
            <div class="tooltip-row"><strong>坐标:</strong> (${coords.row}, ${coords.col})</div>
            <div class="tooltip-row"><strong>颜色值:</strong> ${pixelValue}</div>
            <div class="tooltip-row">
                <strong>RGB:</strong> 
                <span class="color-preview" style="background-color: rgb(${color[0]}, ${color[1]}, ${color[2]});"></span>
                rgb(${color[0]}, ${color[1]}, ${color[2]})
            </div>
            <div class="tooltip-row"><strong>状态:</strong> ${isCompleted ? '✅ 已完成' : isOnPath ? '🟢 路径中' : '⚪ 未完成'}</div>
        `;

        tooltip.innerHTML = tooltipContent;
        tooltip.classList.remove('hidden');

        // 更新工具提示位置
        updateTooltipPosition(event);
    }

    // 隐藏工具提示
    function hideTooltip() {
        tooltip.classList.add('hidden');
    }

    // 合并鼠标移动事件处理，避免重复处理
    function handleCanvasMouseMove(event) {
        // 先处理放大镜
        onMagnifierMouseMove(event);
        // 再处理悬停
        handleSquareHover(event);
    }

    // 添加合并后的鼠标事件监听器
    mainGridCanvas.addEventListener('mousemove', handleCanvasMouseMove);
    mainGridCanvas.addEventListener('mouseleave', () => {
        // 隐藏放大镜
        onMagnifierMouseLeave();
        // 清除悬停状态
        hoveredSquare = null;
        if (!hoverUpdateScheduled) {
            hoverUpdateScheduled = true;
            requestAnimationFrame(() => {
                drawMainGrid();
                hoverUpdateScheduled = false;
            });
        }
        hideTooltip();
    });

    // 由于放大镜容器现在直接追加到 body，使用 position: fixed 定位
    // 它应该完全不受页面滚动影响，始终相对于视口定位
    // 因此不需要滚动事件监听器来更新位置
    // 但我们需要监听窗口大小变化，确保放大镜位置正确
    window.addEventListener('resize', () => {
        // 如果放大镜可见，重新计算位置以适应新的视口大小
        if (!magnifierContainer.classList.contains('hidden') && lastMouseX !== null && lastMouseY !== null) {
            // 触发一次鼠标移动事件来重新计算位置
            // 由于 lastMouseX/Y 是视口坐标，直接使用即可
            const event = new MouseEvent('mousemove', {
                clientX: lastMouseX,
                clientY: lastMouseY,
                bubbles: true
            });
            mainGridCanvas.dispatchEvent(event);
        }
    });
});

