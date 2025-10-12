#!/bin/bash

# 博客管理助手 - Linux/Mac 版本

echo ""
echo "========================================"
echo "          博客管理助手"
echo "========================================"
echo ""

show_menu() {
    echo "请选择操作:"
    echo "1. 创建新博客"
    echo "2. 查看博客列表"
    echo "3. 启动开发服务器"
    echo "4. 退出"
    echo ""
}

create_blog() {
    echo ""
    echo "--- 创建新博客 ---"
    read -p "请输入博客标题: " title
    read -p "请输入分类 (用逗号分隔，如: 技术,AI): " categories
    read -p "请输入标签 (用逗号分隔，如: 深度学习,优化): " tags
    
    echo ""
    echo "正在创建博客..."
    node _scripts/create-blog.js "$title" "$categories" "$tags"
    echo ""
    read -p "按回车键继续..."
}

list_blogs() {
    echo ""
    echo "--- 博客列表 ---"
    ls -la _posts/*.md 2>/dev/null || echo "暂无博客文章"
    echo ""
    read -p "按回车键继续..."
}

start_server() {
    echo ""
    echo "--- 启动开发服务器 ---"
    echo "正在启动 Jekyll 服务器..."
    bundle exec jekyll serve --livereload
}

while true; do
    show_menu
    read -p "请输入选项 (1-4): " choice
    
    case $choice in
        1) create_blog ;;
        2) list_blogs ;;
        3) start_server ;;
        4) echo "再见！"; exit 0 ;;
        *) echo "无效选项，请重新选择" ;;
    esac
done
