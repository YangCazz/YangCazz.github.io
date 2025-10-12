#!/usr/bin/env node

/**
 * 博客创建脚本
 * 使用方法: node _scripts/create-blog.js "博客标题" "分类1,分类2" "标签1,标签2,标签3"
 */

const fs = require('fs');
const path = require('path');

// 获取命令行参数
const args = process.argv.slice(2);
if (args.length < 1) {
    console.log('使用方法: node _scripts/create-blog.js "博客标题" "分类1,分类2" "标签1,标签2,标签3"');
    console.log('示例: node _scripts/create-blog.js "深度学习优化技巧" "技术,AI" "深度学习,优化,技巧"');
    process.exit(1);
}

const title = args[0];
const categories = args[1] ? args[1].split(',').map(cat => cat.trim()) : ['技术'];
const tags = args[2] ? args[2].split(',').map(tag => tag.trim()) : ['博客'];

// 生成文件名（基于标题和当前日期）
const now = new Date();
const dateStr = now.toISOString().split('T')[0];
const slug = title
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .substring(0, 50);

const filename = `${dateStr}-${slug}.md`;
const filepath = path.join('_posts', filename);

// 生成博客内容模板
const template = `---
title: "${title}"
date: ${dateStr}
categories: [${categories.map(cat => cat).join(', ')}]
tags: [${tags.map(tag => tag).join(', ')}]
excerpt: "请在这里添加博客摘要..."
---

# ${title}

## 引言

在这里写您的博客引言...

## 主要内容

### 章节一

在这里写主要内容...

### 章节二

在这里写更多内容...

## 总结

在这里写总结...

## 参考文献

1. 参考文献1
2. 参考文献2
`;

// 检查文件是否已存在
if (fs.existsSync(filepath)) {
    console.log(`❌ 文件已存在: ${filepath}`);
    process.exit(1);
}

// 创建目录（如果不存在）
const postsDir = path.dirname(filepath);
if (!fs.existsSync(postsDir)) {
    fs.mkdirSync(postsDir, { recursive: true });
}

// 写入文件
try {
    fs.writeFileSync(filepath, template, 'utf8');
    console.log(`✅ 博客创建成功: ${filepath}`);
    console.log(`📝 标题: ${title}`);
    console.log(`📂 分类: ${categories.join(', ')}`);
    console.log(`🏷️  标签: ${tags.join(', ')}`);
    console.log(`\n💡 提示: 编辑文件后运行 \`bundle exec jekyll serve --livereload\` 查看效果`);
} catch (error) {
    console.error(`❌ 创建失败: ${error.message}`);
    process.exit(1);
}
