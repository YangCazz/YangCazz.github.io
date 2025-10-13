# 语法高亮快速参考

## 🌈 支持的编程语言

### ✅ 已优化的语言

| 语言 | Markdown标识 | 特色高亮 |
|------|-------------|---------|
| **Python** | \`\`\`python | 装饰器、类型提示、docstring |
| **JavaScript** | \`\`\`javascript 或 \`\`\`js | ES6+语法、async/await |
| **TypeScript** | \`\`\`typescript 或 \`\`\`ts | 类型注解 |
| **HTML** | \`\`\`html | 标签、属性 |
| **CSS** | \`\`\`css | 选择器、属性 |
| **SCSS/Sass** | \`\`\`scss | 变量、嵌套 |
| **Bash/Shell** | \`\`\`bash 或 \`\`\`shell | 命令、变量 |
| **YAML** | \`\`\`yaml 或 \`\`\`yml | 键值对 |
| **JSON** | \`\`\`json | 结构化数据 |
| **Markdown** | \`\`\`markdown 或 \`\`\`md | 标记语法 |
| **Java** | \`\`\`java | 类、方法 |
| **C/C++** | \`\`\`c 或 \`\`\`cpp | 指针、类型 |
| **Ruby** | \`\`\`ruby | 符号、块 |
| **Go** | \`\`\`go | 接口、goroutine |
| **Rust** | \`\`\`rust | 所有权、生命周期 |
| **SQL** | \`\`\`sql | 查询语句 |
| **XML** | \`\`\`xml | 标签、属性 |

## 🎨 语法高亮主题：VS Code Dark+

### 颜色说明

```python
# 这是注释 - 绿色斜体 #6a9955

def function_name():  # def是关键字 - 蓝色 #569cd6
    """这是文档字符串 - 绿色斜体 #6a9955"""
    
    # 函数名 - 黄色 #dcdcaa
    # 字符串 - 橙色 #ce9178
    text = "Hello, World!"
    
    # 数字 - 浅绿色 #b5cea8
    number = 123
    
    # 内置函数 - 青色 #4ec9b0
    print(text)
    
    # 布尔值 - 蓝色 #569cd6
    is_valid = True
    
    return number

# 装饰器 - 黄色 #dcdcaa
@decorator
class ClassName:  # 类名 - 青色 #4ec9b0
    # 变量 - 浅蓝色 #9cdcfe
    variable_name = "value"
```

## 📝 使用示例

### Python 完整示例

\`\`\`python
import numpy as np
from typing import List, Optional

class DataProcessor:
    """数据处理类"""
    
    def __init__(self, data: List[float]):
        self.data = data
        self._processed = False
    
    @property
    def is_processed(self) -> bool:
        """检查是否已处理"""
        return self._processed
    
    def process(self, threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        处理数据
        
        Args:
            threshold: 阈值
            
        Returns:
            处理后的数组
        """
        if not self.data:
            return None
        
        result = np.array([x for x in self.data if x > threshold])
        self._processed = True
        
        return result

# 使用示例
processor = DataProcessor([0.3, 0.7, 0.9, 0.2])
output = processor.process(threshold=0.5)
print(f"处理结果: {output}")
\`\`\`

### JavaScript ES6+ 示例

\`\`\`javascript
// 导入模块
import React, { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * 数据获取组件
 */
const DataFetcher = ({ url }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const response = await axios.get(url);
                setData(response.data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
    }, [url]);
    
    if (loading) return <div>加载中...</div>;
    if (error) return <div>错误: {error}</div>;
    
    return (
        <div className="data-container">
            <h2>数据展示</h2>
            <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
    );
};

export default DataFetcher;
\`\`\`

### Bash 脚本示例

\`\`\`bash
#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 函数：打印成功消息
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 函数：打印错误消息
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 检查依赖
check_dependencies() {
    local deps=("npm" "node" "git")
    
    for dep in "${deps[@]}"; do
        if command -v "$dep" &> /dev/null; then
            print_success "$dep 已安装"
        else
            print_error "$dep 未找到"
            exit 1
        fi
    done
}

# 主流程
main() {
    echo "开始部署..."
    
    check_dependencies
    
    # 安装依赖
    npm install || {
        print_error "依赖安装失败"
        exit 1
    }
    
    # 构建项目
    npm run build || {
        print_error "构建失败"
        exit 1
    }
    
    print_success "部署完成！"
}

# 执行主函数
main "$@"
\`\`\`

### YAML 配置示例

\`\`\`yaml
# GitHub Actions工作流
name: CI/CD Pipeline

on:
  push:
    branches:
      - main
      - develop
  pull_request:
    types: [opened, synchronize]

env:
  NODE_VERSION: '18'
  CACHE_KEY: npm-cache-v1

jobs:
  build:
    name: Build and Test
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: |
          npm ci
          npm run build
      
      - name: Run tests
        run: npm test
        env:
          CI: true
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        if: success()
        with:
          name: build-output
          path: dist/
          retention-days: 7
\`\`\`

## 🔧 高级技巧

### 1. 指定行高亮（需要额外配置）

某些Markdown处理器支持行号高亮：

\`\`\`python{1,3-5}
def example():
    # 第1行会高亮
    print("Hello")  # 第3-5行会高亮
    return True
\`\`\`

### 2. 添加文件名标注

\`\`\`python:app.py
# 这会显示文件名
def main():
    pass
\`\`\`

### 3. 禁用语法高亮

如果不想高亮，使用 \`\`\`text 或 \`\`\`plaintext：

\`\`\`text
这段文本不会被高亮
保持原样显示
\`\`\`

## 📊 性能优化

### 最佳实践

1. **指定语言**：总是明确指定代码块的语言
   - ✅ \`\`\`python
   - ❌ \`\`\` (无语言标识)

2. **合理长度**：避免过长的代码块
   - 建议：每个代码块 < 100行
   - 超长代码：考虑拆分或链接到GitHub

3. **避免嵌套**：不要在代码块中嵌套Markdown

## 🎯 故障排除

### 问题：代码没有高亮

**可能原因**：
1. 语言标识拼写错误
2. Jekyll配置问题
3. CSS文件未加载

**解决方案**：
1. 检查语言标识：\`\`\`python（不是 \`\`\`Python）
2. 确认 `_config.yml` 中配置了 `highlighter: rouge`
3. 确认 `_sass/_highlight-syntax.scss` 已导入

### 问题：某些元素颜色不对

**解决方案**：
- 检查Rouge版本：`bundle show rouge`
- 更新Rouge：`bundle update rouge`
- 清除Jekyll缓存：`bundle exec jekyll clean`

## 📚 参考资料

- [Rouge支持的语言列表](https://github.com/rouge-ruby/rouge/wiki/List-of-supported-languages-and-lexers)
- [VS Code主题参考](https://github.com/microsoft/vscode/tree/main/extensions/theme-defaults)
- [Markdown语法指南](https://www.markdownguide.org/extended-syntax/#fenced-code-blocks)

## 🚀 下一步

想要更多自定义？编辑 `_sass/_highlight-syntax.scss` 文件！

---

*最后更新：2025年10月*

