@echo off
chcp 65001 >nul
echo.
echo ========================================
echo          博客管理助手
echo ========================================
echo.

:menu
echo 请选择操作:
echo 1. 创建新博客
echo 2. 查看博客列表
echo 3. 启动开发服务器
echo 4. 退出
echo.
set /p choice=请输入选项 (1-4): 

if "%choice%"=="1" goto create_blog
if "%choice%"=="2" goto list_blogs
if "%choice%"=="3" goto start_server
if "%choice%"=="4" goto exit
echo 无效选项，请重新选择
goto menu

:create_blog
echo.
echo --- 创建新博客 ---
set /p title=请输入博客标题: 
set /p categories=请输入分类 (用逗号分隔，如: 技术,AI): 
set /p tags=请输入标签 (用逗号分隔，如: 深度学习,优化): 

echo.
echo 正在创建博客...
node _scripts/create-blog.js "%title%" "%categories%" "%tags%"
echo.
pause
goto menu

:list_blogs
echo.
echo --- 博客列表 ---
dir _posts\*.md /b
echo.
pause
goto menu

:start_server
echo.
echo --- 启动开发服务器 ---
echo 正在启动 Jekyll 服务器...
bundle exec jekyll serve --livereload
goto menu

:exit
echo 再见！
exit
