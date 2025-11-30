/**
 * 数据持久化工具类
 * 使用 localStorage 进行数据存储和读取
 */
export class Storage {
    /**
     * 保存地图数据到 localStorage
     * @param {String} mapName - 地图名称
     * @param {Object} mapData - 地图数据 {grid_data, color_map}
     */
    static saveMapData(mapName, mapData) {
        const storageKey = `pixelMap_${mapName}`;
        localStorage.setItem(storageKey, JSON.stringify(mapData));
    }

    /**
     * 从 localStorage 读取地图数据
     * @param {String} mapName - 地图名称
     * @returns {Object|null} 地图数据或 null
     */
    static loadMapData(mapName) {
        const storageKey = `pixelMap_${mapName}`;
        const data = localStorage.getItem(storageKey);
        return data ? JSON.parse(data) : null;
    }

    /**
     * 删除地图数据
     * @param {String} mapName - 地图名称
     */
    static deleteMapData(mapName) {
        const storageKey = `pixelMap_${mapName}`;
        localStorage.removeItem(storageKey);
    }

    /**
     * 保存用户地图列表
     * @param {Array} mapsList - 地图列表
     */
    static saveUserMapsList(mapsList) {
        localStorage.setItem('userMapsList', JSON.stringify(mapsList));
    }

    /**
     * 加载用户地图列表
     * @returns {Array} 用户地图列表
     */
    static loadUserMapsList() {
        const data = localStorage.getItem('userMapsList');
        return data ? JSON.parse(data) : [];
    }

    /**
     * 保存编织进度
     * @param {String} mapName - 地图名称
     * @param {Object} progressState - 进度状态 {completedSquares, currentDiagonalIndex, isDiagonalMode, isConfirmedCompleted}
     */
    static saveProgress(mapName, progressState) {
        const storageKey = `weavingProgressState_${mapName}`;
        localStorage.setItem(storageKey, JSON.stringify(progressState));
    }

    /**
     * 加载编织进度
     * @param {String} mapName - 地图名称
     * @returns {Object|null} 进度状态或 null
     */
    static loadProgress(mapName) {
        const storageKey = `weavingProgressState_${mapName}`;
        const data = localStorage.getItem(storageKey);
        return data ? JSON.parse(data) : null;
    }

    /**
     * 删除编织进度
     * @param {String} mapName - 地图名称
     */
    static deleteProgress(mapName) {
        const storageKey = `weavingProgressState_${mapName}`;
        localStorage.removeItem(storageKey);
    }

    /**
     * 保存下一个地图ID
     * @param {Number} nextId - 下一个ID
     */
    static saveNextMapId(nextId) {
        localStorage.setItem('nextMapId', String(nextId));
    }

    /**
     * 加载下一个地图ID
     * @returns {Number} 下一个ID（默认为1）
     */
    static loadNextMapId() {
        const data = localStorage.getItem('nextMapId');
        return data ? parseInt(data, 10) : 1;
    }
}

