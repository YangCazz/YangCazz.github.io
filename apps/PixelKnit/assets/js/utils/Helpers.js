/**
 * 辅助函数工具类
 * 包含各种通用的辅助函数
 */
export class Helpers {
    /**
     * 生成地图ID（5位数字格式：map_00001）
     * @param {Number} nextId - 下一个ID数字
     * @returns {String} 格式化的地图ID
     */
    static generateMapId(nextId) {
        const idStr = String(nextId).padStart(5, '0');
        return `map_${idStr}`;
    }

    /**
     * 将十六进制颜色转换为RGB对象
     * @param {String} hex - 十六进制颜色值（如 #ffffff）
     * @returns {Object|null} RGB对象 {r, g, b} 或 null
     */
    static hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    /**
     * RGB数组转换为RGB字符串
     * @param {Array} rgb - RGB数组 [r, g, b]
     * @returns {String} RGB字符串 "rgb(r, g, b)"
     */
    static rgbToString(rgb) {
        if (!rgb || !Array.isArray(rgb) || rgb.length !== 3) {
            return 'rgb(0, 0, 0)';
        }
        return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }

    /**
     * 坐标转换为字符串键
     * @param {Number} row - 行索引
     * @param {Number} col - 列索引
     * @returns {String} 坐标键 "row,col"
     */
    static coordToKey(row, col) {
        return `${row},${col}`;
    }

    /**
     * 坐标键转换为坐标对象
     * @param {String} key - 坐标键 "row,col"
     * @returns {Object|null} 坐标对象 {row, col} 或 null
     */
    static keyToCoord(key) {
        const parts = key.split(',');
        if (parts.length !== 2) return null;
        const row = parseInt(parts[0]);
        const col = parseInt(parts[1]);
        if (isNaN(row) || isNaN(col)) return null;
        return { row, col };
    }

    /**
     * 深拷贝对象
     * @param {*} obj - 要拷贝的对象
     * @returns {*} 深拷贝后的对象
     */
    static deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    }

    /**
     * 节流函数
     * @param {Function} func - 要节流的函数
     * @param {Number} delay - 延迟时间（毫秒）
     * @returns {Function} 节流后的函数
     */
    static throttle(func, delay) {
        let lastCall = 0;
        return function(...args) {
            const now = Date.now();
            if (now - lastCall >= delay) {
                lastCall = now;
                return func.apply(this, args);
            }
        };
    }

    /**
     * 防抖函数
     * @param {Function} func - 要防抖的函数
     * @param {Number} delay - 延迟时间（毫秒）
     * @returns {Function} 防抖后的函数
     */
    static debounce(func, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }
}

