/**
 * PixelKnit 主入口文件
 * 整合所有模块并初始化应用
 */

// 导入工具类
import { Algorithms } from './utils/Algorithms.js';
import { Helpers } from './utils/Helpers.js';
import { Storage } from './utils/Storage.js';
import { History } from './utils/History.js';

// 导入特效
import { ParticleSystem } from './effects/ParticleSystem.js';

// 导出工具类供其他模块使用
window.PixelKnit = {
    Algorithms,
    Helpers,
    Storage,
    History
};

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    // 初始化粒子系统
    new ParticleSystem();
    
    // 注意：由于代码量较大，当前版本暂时使用原有的 script.js
    // 完整的模块化迁移将在后续版本中完成
    // 如需使用模块化版本，请参考 README.md 中的重构进度
    console.log('PixelKnit 模块化架构已初始化');
    console.log('工具类已加载:', Object.keys(window.PixelKnit));
});

