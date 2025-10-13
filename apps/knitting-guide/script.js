// 针织引导应用 - 核心JavaScript

class KnittingGuide {
    constructor() {
        this.originalImage = null;
        this.patternData = [];
        this.colorPalette = [];
        this.currentStep = 0;
        this.isGuiding = false;
        this.gridSize = 30;
        this.colorCount = 8;
        this.completedSteps = new Set();
        this.zoomLevel = 1;
        
        this.init();
    }
    
    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.loadProgress();
        this.mode = 'convert'; // convert or pattern
        this.debugMessages = [];
    }
    
    // 添加调试信息
    addDebugMessage(message, type = 'info') {
        console.log(message);
        this.debugMessages.push({ message, type });
        this.updateDebugPanel();
    }
    
    // 更新调试面板
    updateDebugPanel() {
        const debugPanel = document.getElementById('debugPanel');
        const debugContent = document.getElementById('debugContent');
        
        if (this.debugMessages.length > 0) {
            debugPanel.style.display = 'block';
            debugContent.innerHTML = this.debugMessages.map(({ message, type }) => {
                const className = type === 'success' ? 'debug-success' : 
                                 type === 'warning' ? 'debug-warning' :
                                 type === 'error' ? 'debug-error' : '';
                return `<p class="${className}">${message}</p>`;
            }).join('');
            
            // 自动滚动到底部
            debugContent.scrollTop = debugContent.scrollHeight;
        }
    }
    
    // 清空调试信息
    clearDebugMessages() {
        this.debugMessages = [];
        document.getElementById('debugPanel').style.display = 'none';
    }
    
    setupCanvas() {
        this.originalCanvas = document.getElementById('originalCanvas');
        this.patternCanvas = document.getElementById('patternCanvas');
        this.originalCtx = this.originalCanvas.getContext('2d');
        this.patternCtx = this.patternCanvas.getContext('2d');
    }
    
    setupEventListeners() {
        // 模式切换
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.mode = e.target.value;
                document.getElementById('convertSettings').style.display = 
                    this.mode === 'convert' ? 'flex' : 'none';
                document.getElementById('patternSettings').style.display = 
                    this.mode === 'pattern' ? 'flex' : 'none';
            });
        });
        
        // 文件上传
        document.getElementById('imageUpload').addEventListener('change', (e) => {
            this.handleImageUpload(e);
        });
        
        // 生成图案
        document.getElementById('generateBtn').addEventListener('click', () => {
            this.generatePattern();
        });
        
        // 分析图案
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzePattern();
        });
        
        // 加载示例
        document.getElementById('loadExampleBtn').addEventListener('click', () => {
            this.loadExamplePattern();
        });
        
        // 控制按钮
        document.getElementById('startBtn').addEventListener('click', () => {
            this.startGuiding();
        });
        
        document.getElementById('pauseBtn').addEventListener('click', () => {
            this.pauseGuiding();
        });
        
        document.getElementById('prevBtn').addEventListener('click', () => {
            this.previousStep();
        });
        
        document.getElementById('nextBtn').addEventListener('click', () => {
            this.nextStep();
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.reset();
        });
        
        document.getElementById('saveBtn').addEventListener('click', () => {
            this.saveProgress();
        });
        
        document.getElementById('exportBtn').addEventListener('click', () => {
            this.exportPattern();
        });
        
        // 缩放控制
        document.getElementById('zoomIn').addEventListener('click', () => {
            this.zoom(1.2);
        });
        
        document.getElementById('zoomOut').addEventListener('click', () => {
            this.zoom(0.8);
        });
        
        document.getElementById('fitScreen').addEventListener('click', () => {
            this.fitToScreen();
        });
        
        // 画布点击
        this.patternCanvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e);
        });
        
        // 画布悬停
        this.patternCanvas.addEventListener('mousemove', (e) => {
            this.handleCanvasHover(e);
        });
        
        this.patternCanvas.addEventListener('mouseleave', () => {
            document.getElementById('tooltip').style.display = 'none';
        });
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            this.handleKeyPress(e);
        });
        
        // 参数变化
        document.getElementById('gridSize').addEventListener('change', (e) => {
            this.gridSize = parseInt(e.target.value);
        });
        
        document.getElementById('colorCount').addEventListener('change', (e) => {
            this.colorCount = parseInt(e.target.value);
        });
    }
    
    handleImageUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.displayOriginalImage();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    displayOriginalImage() {
        const maxSize = 400;
        const scale = Math.min(maxSize / this.originalImage.width, maxSize / this.originalImage.height);
        
        this.originalCanvas.width = this.originalImage.width * scale;
        this.originalCanvas.height = this.originalImage.height * scale;
        
        this.originalCtx.drawImage(this.originalImage, 0, 0, 
            this.originalCanvas.width, this.originalCanvas.height);
    }
    
    generatePattern() {
        if (!this.originalImage) {
            alert('请先上传图片！');
            return;
        }
        
        // 计算网格尺寸
        const cols = Math.floor(this.originalImage.width / this.gridSize);
        const rows = Math.floor(this.originalImage.height / this.gridSize);
        
        // 创建临时画布
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.originalImage.width;
        tempCanvas.height = this.originalImage.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(this.originalImage, 0, 0);
        
        // 提取颜色数据
        this.patternData = [];
        const colors = [];
        
        for (let row = 0; row < rows; row++) {
            this.patternData[row] = [];
            for (let col = 0; col < cols; col++) {
                const x = col * this.gridSize + this.gridSize / 2;
                const y = row * this.gridSize + this.gridSize / 2;
                const imageData = tempCtx.getImageData(x, y, 1, 1).data;
                const color = `rgb(${imageData[0]}, ${imageData[1]}, ${imageData[2]})`;
                colors.push(color);
                this.patternData[row][col] = { color, completed: false };
            }
        }
        
        // 颜色量化
        this.colorPalette = this.quantizeColors(colors, this.colorCount);
        
        // 将颜色映射到调色板
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const originalColor = this.patternData[row][col].color;
                this.patternData[row][col].color = this.findClosestColor(originalColor);
            }
        }
        
        // 绘制图案
        this.drawPattern();
        this.displayColorPalette();
        this.showControls();
    }
    
    quantizeColors(colors, count) {
        // 简单的K-means颜色量化
        const colorArray = colors.map(c => {
            const rgb = c.match(/\d+/g);
            return [parseInt(rgb[0]), parseInt(rgb[1]), parseInt(rgb[2])];
        });
        
        // 随机选择初始中心点
        let centroids = [];
        for (let i = 0; i < count; i++) {
            centroids.push(colorArray[Math.floor(Math.random() * colorArray.length)]);
        }
        
        // K-means迭代
        for (let iter = 0; iter < 10; iter++) {
            const clusters = Array(count).fill(null).map(() => []);
            
            // 分配到最近的中心
            colorArray.forEach(color => {
                let minDist = Infinity;
                let clusterIndex = 0;
                centroids.forEach((centroid, i) => {
                    const dist = this.colorDistance(color, centroid);
                    if (dist < minDist) {
                        minDist = dist;
                        clusterIndex = i;
                    }
                });
                clusters[clusterIndex].push(color);
            });
            
            // 更新中心点
            centroids = clusters.map(cluster => {
                if (cluster.length === 0) return centroids[0];
                const sum = cluster.reduce((acc, c) => [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]], [0, 0, 0]);
                return [
                    Math.round(sum[0] / cluster.length),
                    Math.round(sum[1] / cluster.length),
                    Math.round(sum[2] / cluster.length)
                ];
            });
        }
        
        return centroids.map(c => `rgb(${c[0]}, ${c[1]}, ${c[2]})`);
    }
    
    colorDistance(c1, c2) {
        return Math.sqrt(
            Math.pow(c1[0] - c2[0], 2) +
            Math.pow(c1[1] - c2[1], 2) +
            Math.pow(c1[2] - c2[2], 2)
        );
    }
    
    findClosestColor(color) {
        const rgb = color.match(/\d+/g).map(Number);
        let minDist = Infinity;
        let closestColor = this.colorPalette[0];
        
        this.colorPalette.forEach(paletteColor => {
            const paletteRgb = paletteColor.match(/\d+/g).map(Number);
            const dist = this.colorDistance(rgb, paletteRgb);
            if (dist < minDist) {
                minDist = dist;
                closestColor = paletteColor;
            }
        });
        
        return closestColor;
    }
    
    drawPattern() {
        const rows = this.patternData.length;
        const cols = this.patternData[0].length;
        const cellSize = 20 * this.zoomLevel;
        
        this.patternCanvas.width = cols * cellSize;
        this.patternCanvas.height = rows * cellSize;
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const cell = this.patternData[row][col];
                const x = col * cellSize;
                const y = row * cellSize;
                
                // 绘制方块
                this.patternCtx.fillStyle = cell.color;
                this.patternCtx.fillRect(x, y, cellSize, cellSize);
                
                // 绘制边框
                this.patternCtx.strokeStyle = '#ccc';
                this.patternCtx.strokeRect(x, y, cellSize, cellSize);
                
                // 标记已完成
                if (cell.completed) {
                    this.patternCtx.fillStyle = 'rgba(255, 255, 255, 0.6)';
                    this.patternCtx.fillRect(x, y, cellSize, cellSize);
                    this.patternCtx.fillStyle = '#000';
                    this.patternCtx.font = `${cellSize * 0.6}px Arial`;
                    this.patternCtx.textAlign = 'center';
                    this.patternCtx.textBaseline = 'middle';
                    this.patternCtx.fillText('✓', x + cellSize / 2, y + cellSize / 2);
                }
                
                // 高亮当前方块
                if (this.isGuiding && this.getCellIndex(row, col) === this.currentStep) {
                    this.patternCtx.strokeStyle = '#ff0000';
                    this.patternCtx.lineWidth = 4;
                    this.patternCtx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
                    this.patternCtx.lineWidth = 1;
                }
            }
        }
    }
    
    displayColorPalette() {
        const palette = document.getElementById('colorPalette');
        palette.innerHTML = '';
        
        const colorCounts = {};
        this.patternData.flat().forEach(cell => {
            colorCounts[cell.color] = (colorCounts[cell.color] || 0) + 1;
        });
        
        this.colorPalette.forEach(color => {
            const item = document.createElement('div');
            item.className = 'color-item';
            item.innerHTML = `
                <div class="color-swatch" style="background: ${color};"></div>
                <div class="color-info">
                    <div class="color-code">${color}</div>
                    <div class="color-count">${colorCounts[color] || 0} 个方块</div>
                </div>
            `;
            palette.appendChild(item);
        });
        
        document.getElementById('paletteSection').style.display = 'block';
    }
    
    showControls() {
        document.getElementById('actionButtons').style.display = 'flex';
        document.getElementById('progressInfo').style.display = 'block';
        this.updateProgress();
    }
    
    startGuiding() {
        this.isGuiding = true;
        document.getElementById('startBtn').style.display = 'none';
        document.getElementById('pauseBtn').style.display = 'inline-block';
        this.drawPattern();
        this.updateCurrentColorInfo();
    }
    
    pauseGuiding() {
        this.isGuiding = false;
        document.getElementById('startBtn').style.display = 'inline-block';
        document.getElementById('pauseBtn').style.display = 'none';
        this.drawPattern();
    }
    
    nextStep() {
        if (this.currentStep < this.getTotalStitches() - 1) {
            const { row, col } = this.getCellPosition(this.currentStep);
            this.patternData[row][col].completed = true;
            this.completedSteps.add(this.currentStep);
            this.currentStep++;
            this.drawPattern();
            this.updateProgress();
            this.updateCurrentColorInfo();
        }
    }
    
    previousStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            const { row, col } = this.getCellPosition(this.currentStep);
            this.patternData[row][col].completed = false;
            this.completedSteps.delete(this.currentStep);
            this.drawPattern();
            this.updateProgress();
            this.updateCurrentColorInfo();
        }
    }
    
    reset() {
        if (confirm('确定要重置所有进度吗？')) {
            this.currentStep = 0;
            this.completedSteps.clear();
            this.patternData.forEach(row => {
                row.forEach(cell => cell.completed = false);
            });
            this.drawPattern();
            this.updateProgress();
            this.updateCurrentColorInfo();
        }
    }
    
    getCellIndex(row, col) {
        return row * this.patternData[0].length + col;
    }
    
    getCellPosition(index) {
        const cols = this.patternData[0].length;
        return {
            row: Math.floor(index / cols),
            col: index % cols
        };
    }
    
    getTotalStitches() {
        return this.patternData.length * this.patternData[0].length;
    }
    
    updateProgress() {
        const total = this.getTotalStitches();
        const completed = this.completedSteps.size;
        const percentage = Math.round((completed / total) * 100);
        
        document.getElementById('currentStitch').textContent = completed;
        document.getElementById('totalStitches').textContent = total;
        document.getElementById('percentage').textContent = percentage + '%';
        document.getElementById('progressBar').style.width = percentage + '%';
    }
    
    updateCurrentColorInfo() {
        const { row, col } = this.getCellPosition(this.currentStep);
        if (row < this.patternData.length && col < this.patternData[0].length) {
            const color = this.patternData[row][col].color;
            document.getElementById('currentColorPreview').style.background = color;
            document.getElementById('currentColorCode').textContent = color;
        }
    }
    
    handleCanvasClick(e) {
        if (!this.isGuiding) return;
        
        const rect = this.patternCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const cellSize = 20 * this.zoomLevel;
        const col = Math.floor(x / cellSize);
        const row = Math.floor(y / cellSize);
        
        const clickedIndex = this.getCellIndex(row, col);
        
        if (clickedIndex === this.currentStep) {
            this.nextStep();
        }
    }
    
    handleCanvasHover(e) {
        const rect = this.patternCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const cellSize = 20 * this.zoomLevel;
        const col = Math.floor(x / cellSize);
        const row = Math.floor(y / cellSize);
        
        if (row >= 0 && row < this.patternData.length && 
            col >= 0 && col < this.patternData[0].length) {
            const cell = this.patternData[row][col];
            const tooltip = document.getElementById('tooltip');
            tooltip.textContent = `位置: (${row}, ${col}) | 颜色: ${cell.color} | ${cell.completed ? '已完成' : '未完成'}`;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
        }
    }
    
    handleKeyPress(e) {
        if (!this.isGuiding) return;
        
        switch(e.key) {
            case ' ':
            case 'Enter':
                e.preventDefault();
                this.nextStep();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                this.previousStep();
                break;
            case 'ArrowRight':
                e.preventDefault();
                this.nextStep();
                break;
        }
    }
    
    zoom(factor) {
        this.zoomLevel *= factor;
        this.zoomLevel = Math.max(0.5, Math.min(3, this.zoomLevel));
        this.drawPattern();
    }
    
    fitToScreen() {
        this.zoomLevel = 1;
        this.drawPattern();
    }
    
    saveProgress() {
        const progress = {
            patternData: this.patternData,
            colorPalette: this.colorPalette,
            currentStep: this.currentStep,
            completedSteps: Array.from(this.completedSteps),
            gridSize: this.gridSize,
            colorCount: this.colorCount
        };
        
        localStorage.setItem('knittingProgress', JSON.stringify(progress));
        alert('进度已保存！');
    }
    
    loadProgress() {
        const saved = localStorage.getItem('knittingProgress');
        if (saved) {
            const progress = JSON.parse(saved);
            if (confirm('检测到保存的进度，是否继续上次的编织？')) {
                this.patternData = progress.patternData;
                this.colorPalette = progress.colorPalette;
                this.currentStep = progress.currentStep;
                this.completedSteps = new Set(progress.completedSteps);
                this.gridSize = progress.gridSize;
                this.colorCount = progress.colorCount;
                
                if (this.patternData.length > 0) {
                    this.drawPattern();
                    this.displayColorPalette();
                    this.showControls();
                }
            }
        }
    }
    
    // 分析已有的针织图案
    analyzePattern() {
        if (!this.originalImage) {
            alert('请先上传针织图案图片！');
            return;
        }
        
        this.clearDebugMessages();
        
        console.log('========================================');
        this.addDebugMessage('========================================');
        this.addDebugMessage('🧶 开始分析针织图案', 'info');
        this.addDebugMessage(`📐 图片尺寸: ${this.originalImage.width} x ${this.originalImage.height}`, 'info');
        this.addDebugMessage('========================================');
        console.log('🧶 开始分析针织图案');
        console.log(`图片尺寸: ${this.originalImage.width} x ${this.originalImage.height}`);
        console.log('========================================');
        
        // 创建临时画布进行分析
        const canvas = document.createElement('canvas');
        canvas.width = this.originalImage.width;
        canvas.height = this.originalImage.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.originalImage, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // 检测网格线颜色
        const gridLineColor = document.getElementById('autoDetectGrid').checked ?
            this.detectGridLineColor(data, canvas.width, canvas.height) :
            this.hexToRgb(document.getElementById('gridLineColor').value);
        
        // 检测网格单元大小
        const cellSize = this.detectCellSize(data, canvas.width, canvas.height, gridLineColor);
        
        if (!cellSize) {
            console.log('========================================');
            console.log('❌ 分析失败');
            console.log('========================================');
            
            this.addDebugMessage('❌ 无法检测网格大小', 'error');
            this.addDebugMessage('💡 请尝试手动选择网格线颜色', 'warning');
            
            alert(`无法自动检测网格大小！
            
可能的原因：
1. 图片网格线不够清晰
2. 网格线颜色与背景对比度不够
3. 图片分辨率过低或过高

建议：
• 取消勾选"自动检测网格"
• 手动选择网格线颜色
• 或者尝试使用普通图片转换模式`);
            return;
        }
        
        // 提取网格数据
        this.extractGridPattern(canvas, cellSize, gridLineColor);
        
        console.log('========================================');
        console.log('✅ 分析完成！');
        console.log('========================================');
        
        this.addDebugMessage('✅ 分析完成！', 'success');
    }
    
    // 检测网格线颜色（通常是灰色或黑色）
    detectGridLineColor(data, width, height) {
        console.log('🔍 开始检测网格线颜色...');
        this.addDebugMessage('🔍 正在检测网格线颜色...', 'info');
        
        const colorCounts = {};
        const sampleStep = 5; // 更密集的采样
        
        // 统计所有像素颜色
        for (let y = 0; y < height; y += sampleStep) {
            for (let x = 0; x < width; x += sampleStep) {
                const idx = (y * width + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const a = data[idx + 3];
                
                // 跳过透明像素
                if (a < 128) continue;
                
                // 将颜色分组（容差10）
                const rGroup = Math.round(r / 10) * 10;
                const gGroup = Math.round(g / 10) * 10;
                const bGroup = Math.round(b / 10) * 10;
                const color = `${rGroup},${gGroup},${bGroup}`;
                colorCounts[color] = (colorCounts[color] || 0) + 1;
            }
        }
        
        // 按频率排序
        const sortedColors = Object.entries(colorCounts)
            .sort((a, b) => b[1] - a[1]);
        
        console.log('📊 前10个最常见的颜色:');
        sortedColors.slice(0, 10).forEach(([color, count]) => {
            console.log(`  ${color} - 出现 ${count} 次`);
        });
        
        // 寻找网格线颜色：通常是浅灰色（RGB接近且在180-220范围）或深灰色/黑色
        let gridColor = null;
        for (const [color, count] of sortedColors) {
            const rgb = color.split(',').map(Number);
            const r = rgb[0], g = rgb[1], b = rgb[2];
            
            // 检查是否是灰色系（RGB值接近）
            const isGrayish = Math.abs(r - g) < 30 && Math.abs(g - b) < 30 && Math.abs(r - b) < 30;
            
            // 检查是否在网格线的亮度范围
            const avg = (r + g + b) / 3;
            const isLightGray = avg >= 180 && avg <= 230; // 浅灰色网格线
            const isDarkGray = avg >= 80 && avg <= 150;   // 深灰色网格线
            
            if (isGrayish && (isLightGray || isDarkGray)) {
                gridColor = { r: rgb[0], g: rgb[1], b: rgb[2] };
                console.log(`✅ 检测到网格线颜色: rgb(${r}, ${g}, ${b}) - ${isLightGray ? '浅灰色' : '深灰色'}`);
                this.addDebugMessage(`✅ 网格线颜色: rgb(${r}, ${g}, ${b}) - ${isLightGray ? '浅灰色' : '深灰色'}`, 'success');
                break;
            }
        }
        
        // 如果没找到，使用最常见的灰色
        if (!gridColor) {
            for (const [color, count] of sortedColors) {
                const rgb = color.split(',').map(Number);
                const r = rgb[0], g = rgb[1], b = rgb[2];
                if (Math.abs(r - g) < 30 && Math.abs(g - b) < 30) {
                    gridColor = { r: rgb[0], g: rgb[1], b: rgb[2] };
                    console.log(`⚠️ 使用备选网格线颜色: rgb(${r}, ${g}, ${b})`);
                    break;
                }
            }
        }
        
        // 最后的后备方案
        if (!gridColor) {
            gridColor = { r: 200, g: 200, b: 200 };
            console.log('⚠️ 无法自动检测，使用默认灰色');
        }
        
        return gridColor;
    }
    
    // 检测单元格大小
    detectCellSize(data, width, height, gridColor) {
        console.log('📏 开始检测单元格大小...');
        console.log(`  使用网格线颜色: rgb(${gridColor.r}, ${gridColor.g}, ${gridColor.b})`);
        this.addDebugMessage('📏 正在检测单元格大小...', 'info');
        
        const tolerance = 40; // 增加颜色容差
        const distances = [];
        
        // 多行扫描获得更准确的结果
        const scanLines = [
            Math.floor(height * 0.3),
            Math.floor(height * 0.5),
            Math.floor(height * 0.7)
        ];
        
        for (const scanY of scanLines) {
            let lastGridX = -1;
            let inGridLine = false;
            
            for (let x = 0; x < width; x++) {
                const idx = (scanY * width + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                
                // 检查是否是网格线颜色
                const isGridLine = Math.abs(r - gridColor.r) < tolerance &&
                                  Math.abs(g - gridColor.g) < tolerance &&
                                  Math.abs(b - gridColor.b) < tolerance;
                
                // 检测网格线的边缘
                if (isGridLine && !inGridLine) {
                    // 进入网格线
                    if (lastGridX >= 0) {
                        const distance = x - lastGridX;
                        if (distance > 8 && distance < 200) { // 合理的单元格大小范围
                            distances.push(distance);
                        }
                    }
                    lastGridX = x;
                    inGridLine = true;
                } else if (!isGridLine && inGridLine) {
                    // 离开网格线
                    inGridLine = false;
                }
            }
        }
        
        console.log(`  检测到 ${distances.length} 个间距`);
        
        if (distances.length === 0) {
            console.log('❌ 未检测到网格线间距');
            return null;
        }
        
        // 使用中位数而不是平均值，更健壮
        distances.sort((a, b) => a - b);
        const median = distances[Math.floor(distances.length / 2)];
        
        // 过滤掉偏差过大的值
        const filtered = distances.filter(d => Math.abs(d - median) < median * 0.3);
        const avgDistance = Math.round(filtered.reduce((a, b) => a + b) / filtered.length);
        
        console.log(`  原始间距范围: ${Math.min(...distances)} - ${Math.max(...distances)}`);
        console.log(`  中位数: ${median}`);
        console.log(`  平均间距: ${avgDistance}`);
        console.log(`✅ 单元格大小: ${avgDistance}px`);
        
        this.addDebugMessage(`✅ 单元格大小: ${avgDistance}px`, 'success');
        this.addDebugMessage(`   检测到 ${distances.length} 个网格间距`, 'info');
        
        return avgDistance;
    }
    
    // 提取网格图案
    extractGridPattern(canvas, cellSize, gridColor) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // 计算网格数量
        const cols = Math.floor(canvas.width / cellSize);
        const rows = Math.floor(canvas.height / cellSize);
        
        console.log(`检测到 ${rows} 行 x ${cols} 列`);
        
        // 提取每个单元格的颜色
        this.patternData = [];
        const allColors = [];
        
        for (let row = 0; row < rows; row++) {
            this.patternData[row] = [];
            for (let col = 0; col < cols; col++) {
                // 采样单元格中心点的颜色
                const centerX = col * cellSize + Math.floor(cellSize / 2);
                const centerY = row * cellSize + Math.floor(cellSize / 2);
                
                if (centerX >= canvas.width || centerY >= canvas.height) continue;
                
                const idx = (centerY * canvas.width + centerX) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                
                const color = `rgb(${r}, ${g}, ${b})`;
                allColors.push(color);
                
                this.patternData[row][col] = {
                    color: color,
                    completed: false
                };
            }
        }
        
        // 提取调色板（去重）
        const uniqueColors = [...new Set(allColors)];
        
        // 过滤掉网格线颜色和白色背景
        this.colorPalette = uniqueColors.filter(color => {
            const rgb = color.match(/\d+/g).map(Number);
            const isGridLine = Math.abs(rgb[0] - gridColor.r) < 30 &&
                              Math.abs(rgb[1] - gridColor.g) < 30 &&
                              Math.abs(rgb[2] - gridColor.b) < 30;
            const isWhite = rgb[0] > 240 && rgb[1] > 240 && rgb[2] > 240;
            return !isGridLine && !isWhite;
        });
        
        console.log('提取到的颜色:', this.colorPalette);
        
        // 如果颜色太多，进行聚类
        if (this.colorPalette.length > 20) {
            this.colorPalette = this.quantizeColors(allColors, 15);
            // 重新映射颜色
            for (let row = 0; row < this.patternData.length; row++) {
                for (let col = 0; col < this.patternData[row].length; col++) {
                    const originalColor = this.patternData[row][col].color;
                    this.patternData[row][col].color = this.findClosestColor(originalColor);
                }
            }
        }
        
        console.log('========================================');
        console.log('📊 分析结果:');
        console.log(`  网格: ${rows} 行 x ${cols} 列`);
        console.log(`  单元格大小: ${cellSize}px`);
        console.log(`  颜色数量: ${this.colorPalette.length}`);
        console.log(`  总方块数: ${rows * cols}`);
        console.log('========================================');
        
        this.addDebugMessage('========================================');
        this.addDebugMessage('📊 分析结果:', 'success');
        this.addDebugMessage(`   网格: ${rows} 行 x ${cols} 列`, 'success');
        this.addDebugMessage(`   颜色数量: ${this.colorPalette.length} 种`, 'success');
        this.addDebugMessage(`   总方块数: ${rows * cols}`, 'success');
        this.addDebugMessage('========================================');
        
        // 显示结果
        this.drawPattern();
        this.displayColorPalette();
        this.showControls();
        
        alert(`✅ 成功分析针织图案！

📐 检测结果:
• 网格大小: ${rows} 行 x ${cols} 列
• 单元格: ${cellSize} x ${cellSize} 像素
• 颜色数量: ${this.colorPalette.length} 种
• 总方块数: ${rows * cols}

💡 提示:
可以查看浏览器控制台获取详细的分析日志`);
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 200, g: 200, b: 200 };
    }
    
    // 加载示例图案
    async loadExamplePattern() {
        try {
            this.clearDebugMessages();
            this.addDebugMessage('📥 正在加载示例图案...', 'info');
            
            const response = await fetch('example-pattern.json');
            if (!response.ok) {
                throw new Error('无法加载示例文件');
            }
            
            const exampleData = await response.json();
            
            this.addDebugMessage(`✅ 成功加载: ${exampleData.name}`, 'success');
            this.addDebugMessage(`📐 尺寸: ${exampleData.rows} 行 x ${exampleData.cols} 列`, 'info');
            this.addDebugMessage(`🎨 颜色: ${exampleData.colors.length} 种`, 'info');
            
            // 转换数据格式
            this.patternData = [];
            this.colorPalette = exampleData.colors.map(c => c.rgb);
            
            for (let row = 0; row < exampleData.rows; row++) {
                this.patternData[row] = [];
                for (let col = 0; col < exampleData.cols; col++) {
                    const colorIndex = exampleData.pattern[row][col];
                    this.patternData[row][col] = {
                        color: exampleData.colors[colorIndex].rgb,
                        completed: false
                    };
                }
            }
            
            this.gridSize = 20; // 示例使用固定网格大小
            
            // 显示结果
            this.drawPattern();
            this.displayColorPalette();
            this.showControls();
            
            this.addDebugMessage('✅ 示例加载完成！可以开始引导', 'success');
            
            alert(`✅ 成功加载示例图案！

📊 图案信息:
• 名称: ${exampleData.name}
• 尺寸: ${exampleData.rows} 行 x ${exampleData.cols} 列
• 颜色: ${exampleData.colors.length} 种
• 总方块: ${exampleData.rows * exampleData.cols}

💡 提示:
现在可以点击"开始引导"按钮开始编织了！`);
            
        } catch (error) {
            console.error('加载示例失败:', error);
            this.addDebugMessage(`❌ 加载失败: ${error.message}`, 'error');
            alert('加载示例图案失败！\n请确保 example-pattern.json 文件存在。');
        }
    }
    
    // 导出当前图案为JSON
    exportPattern() {
        if (!this.patternData || this.patternData.length === 0) {
            alert('没有可导出的图案！请先生成或加载图案。');
            return;
        }
        
        // 提取唯一颜色并创建映射
        const uniqueColors = [...new Set(this.colorPalette)];
        const colorMap = {};
        uniqueColors.forEach((color, index) => {
            colorMap[color] = index;
        });
        
        // 转换图案数据为索引数组
        const pattern = this.patternData.map(row => 
            row.map(cell => colorMap[cell.color])
        );
        
        // 创建颜色信息
        const colors = uniqueColors.map((rgb, index) => {
            const match = rgb.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
            if (match) {
                const r = parseInt(match[1]);
                const g = parseInt(match[2]);
                const b = parseInt(match[3]);
                const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`.toUpperCase();
                return {
                    name: `颜色${index + 1}`,
                    rgb: rgb,
                    hex: hex
                };
            }
            return { name: `颜色${index + 1}`, rgb: rgb, hex: '#000000' };
        });
        
        // 创建导出数据
        const exportData = {
            name: "自定义针织图案",
            description: "通过针织引导应用生成",
            rows: this.patternData.length,
            cols: this.patternData[0].length,
            colors: colors,
            pattern: pattern,
            metadata: {
                creator: "针织引导应用",
                created: new Date().toISOString().split('T')[0],
                version: "1.0",
                gridSize: this.gridSize
            }
        };
        
        // 创建下载链接
        const jsonStr = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `knitting-pattern-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        alert(`✅ 图案已导出！

📊 导出信息:
• 尺寸: ${exportData.rows} 行 x ${exportData.cols} 列
• 颜色: ${exportData.colors.length} 种
• 文件名: ${a.download}

💡 提示:
将导出的JSON文件重命名为 example-pattern.json 
并放在应用目录下，即可作为预设示例使用！`);
    }
}

// 初始化应用
window.addEventListener('DOMContentLoaded', () => {
    new KnittingGuide();
});

