@echo off
chcp 65001 >nul
echo.
echo ========================================
echo          博客创建助手
echo ========================================
echo.

set /p title=请输入博客标题: 
set /p categories=请输入分类 (用逗号分隔，如: 技术,AI): 
set /p tags=请输入标签 (用逗号分隔，如: 深度学习,优化): 

echo.
echo 正在创建博客...

:: 获取当前日期
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "date=%YYYY%-%MM%-%DD%"

:: 生成文件名（简化版）
set "filename=%date%-test-blog.md"

:: 创建博客内容
(
echo ---
echo title: "%title%"
echo date: %date%
echo categories: [%categories%]
echo tags: [%tags%]
echo excerpt: "请在这里添加博客摘要..."
echo ---
echo.
echo # %title%
echo.
echo ## 引言
echo.
echo 在这里写您的博客引言...
echo.
echo ## 主要内容
echo.
echo ### 章节一
echo.
echo 在这里写主要内容...
echo.
echo ### 章节二
echo.
echo 在这里写更多内容...
echo.
echo ## 总结
echo.
echo 在这里写总结...
echo.
echo ## 参考文献
echo.
echo 1. 参考文献1
echo 2. 参考文献2
) > "_posts\%filename%"

echo ✅ 博客创建成功: _posts\%filename%
echo 📝 标题: %title%
echo 📂 分类: %categories%
echo 🏷️  标签: %tags%
echo.
echo 💡 提示: 编辑文件后刷新浏览器查看效果
echo.
pause
