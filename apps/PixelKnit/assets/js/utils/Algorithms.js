/**
 * 算法工具类
 * 包含路径计算相关的算法
 */
export class Algorithms {
    /**
     * 使用 Bresenham 算法计算两点之间的直线路径
     * @param {Object} p1 - 起点 {row, col}
     * @param {Object} p2 - 终点 {row, col}
     * @returns {Array} 路径坐标数组 [{row, col}, ...]
     */
    static calculatePath(p1, p2) {
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

    /**
     * 计算所有对角线路径（从右下角开始，按对角线从右向左推进）
     * @param {Array} gridData - 网格数据
     * @param {Object} colorMap - 颜色映射
     * @returns {Array} 对角线路径数组，每个路径包含 path, totalSquares, colorBreakdown, colorSequence
     */
    static calculateAllDiagonalPaths(gridData, colorMap) {
        const height = gridData.length;
        const width = gridData[0].length;
        const diagonalPaths = [];
        let maxColorCount = 0;

        // 从右下角开始 (r = height - 1, c = width - 1)
        // 对角线总数 = width + height - 1
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
                                startIndex: sequenceStartIndex,
                                endIndex: sequenceEndIndex
                            });
                        }
                        lastColorKey = currentColorKey;
                        count = 1;
                        lastPixelValue = pixelValue;
                        sequenceStartIndex = pathIndex;
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
                        startIndex: sequenceStartIndex,
                        endIndex: sequenceEndIndex
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

        return { diagonalPaths, maxColorCount };
    }
}

