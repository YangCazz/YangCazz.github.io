/**
 * 编辑历史管理类
 * 支持撤销/重做功能
 */
export class History {
    constructor(maxHistorySize = 50) {
        this.history = [];
        this.currentIndex = -1;
        this.maxHistorySize = maxHistorySize;
    }

    /**
     * 添加历史记录
     * @param {Array} gridData - 网格数据
     * @param {Object} colorMap - 颜色映射
     */
    add(gridData, colorMap) {
        // 如果当前不在历史记录的末尾，删除后面的记录
        if (this.currentIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.currentIndex + 1);
        }

        // 添加新记录
        this.history.push({
            gridData: JSON.parse(JSON.stringify(gridData)),
            colorMap: JSON.parse(JSON.stringify(colorMap))
        });

        // 限制历史记录大小
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        } else {
            this.currentIndex++;
        }
    }

    /**
     * 撤销
     * @returns {Object|null} 历史记录 {gridData, colorMap} 或 null
     */
    undo() {
        if (this.canUndo()) {
            this.currentIndex--;
            return this.history[this.currentIndex];
        }
        return null;
    }

    /**
     * 重做
     * @returns {Object|null} 历史记录 {gridData, colorMap} 或 null
     */
    redo() {
        if (this.canRedo()) {
            this.currentIndex++;
            return this.history[this.currentIndex];
        }
        return null;
    }

    /**
     * 是否可以撤销
     * @returns {Boolean}
     */
    canUndo() {
        return this.currentIndex > 0;
    }

    /**
     * 是否可以重做
     * @returns {Boolean}
     */
    canRedo() {
        return this.currentIndex < this.history.length - 1;
    }

    /**
     * 清空历史记录
     */
    clear() {
        this.history = [];
        this.currentIndex = -1;
    }

    /**
     * 获取当前历史记录
     * @returns {Object|null} 当前历史记录或 null
     */
    getCurrent() {
        if (this.currentIndex >= 0 && this.currentIndex < this.history.length) {
            return this.history[this.currentIndex];
        }
        return null;
    }
}

