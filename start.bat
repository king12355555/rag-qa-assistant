@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
title 企业级 RAG 智能问答助手 - 一键启动

REM ============================================================
REM  一键启动脚本（双击即可运行）
REM  1. 自动切换到项目目录
REM  2. 自动选择 Python（优先使用内置离线虚拟环境 data\venv_offline）
REM  3. 检查依赖缺失时自动提供「离线 / 联网」两种安装方式
REM  4. 自动生成 .env、自动检测并构建向量索引
REM  5. 菜单选择：Web 界面 / 命令行问答 / 重建索引
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   企业级 RAG 智能问答助手（一键启动）
echo ============================================================
echo.

REM ==================== 1. 选择 Python 解释器 ====================
set "PY="
if exist "data\venv_offline\Scripts\python.exe" (
    set "PY=%CD%\data\venv_offline\Scripts\python.exe"
    echo [1/4] 使用内置虚拟环境：data\venv_offline
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%i in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%i"
    )
    if not defined PY set "PY=python"
    echo [1/4] 使用系统 Python：%PY%
)

"%PY%" --version >nul 2>nul
if errorlevel 1 (
    echo [错误] 无法运行 Python（%PY%）。
    echo        请先安装 Python 3.10~3.12 并勾选 "Add to PATH"。
    pause
    exit /b 1
)
"%PY%" --version

REM ==================== 2. 检查依赖 ====================
echo [2/4] 检查依赖……
"%PY%" -c "import streamlit, langchain, langchain_community, chromadb, sentence_transformers" >nul 2>nul
if errorlevel 1 (
    call :install_deps
    if errorlevel 1 (
        echo [错误] 依赖未就绪，无法继续。
        pause
        exit /b 1
    )
) else (
    echo       依赖检查通过。
)

REM ==================== 3. 确保 .env ====================
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo [3/4] 已生成 .env（复制自 .env.example），如需对接 LLM 请修改其内容。
    ) else (
        echo [3/4] 未找到 .env，使用默认配置。
    )
) else (
    echo [3/4] .env 已存在。
)

REM ==================== 4. 确保向量索引 ====================
set "NEED_INDEX="
if not exist "data\chroma_db" set "NEED_INDEX=1"
if not defined NEED_INDEX (
    dir /b "data\chroma_db" 2>nul | findstr /r "." >nul
    if errorlevel 1 set "NEED_INDEX=1"
)
if defined NEED_INDEX (
    echo [4/4] 未检测到向量索引，正在构建（文档目录：test_samples）……
    call :build_index
) else (
    echo [4/4] 向量索引已存在。
)

:menu
echo.
echo ============================================================
echo   请选择操作：
echo   [1] 启动 Web 界面（Streamlit，自动打开浏览器）
echo   [2] 命令行问答（交互模式）
echo   [3] 重建向量索引
echo   [4] 退出
echo ============================================================
choice /c 1234 /n /m "请输入选项 [1-4]："
if errorlevel 4 exit /b 0
if errorlevel 3 (
    call :build_index
    goto menu
)
if errorlevel 2 (
    "%PY%" -m src.cli
    echo.
    pause
    goto menu
)
if errorlevel 1 (
    echo 正在启动 Web 界面：http://localhost:8501 （首次加载模型稍慢）……
    start "RAG Web 界面" cmd /k ""%PY%" -m streamlit run src\app.py"
    ping -n 9 127.0.0.1 >nul
    start "" "http://localhost:8501"
    echo Web 界面已在独立窗口启动，关闭该窗口即停止服务。
    pause
    goto menu
)

REM ==================== 子程序：安装依赖 ====================
:install_deps
echo.
echo [提示] 依赖未安装或已损坏，请选择安装方式：
echo   [1] 离线安装（使用 offline 目录，无需联网，推荐）
echo   [2] 联网安装（pip 清华镜像，需要网络）
echo   [3] 跳过（返回主菜单）
choice /c 123 /n /m "请输入选项 [1-3]："
if errorlevel 3 (
    echo [提示] 已跳过依赖安装。
    exit /b 1
)
if errorlevel 2 (
    echo 正在联网安装依赖（清华镜像）……
    "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 联网安装失败，请检查网络后重试。
        exit /b 1
    )
    echo [OK] 依赖安装完成。
    exit /b 0
)
if errorlevel 1 (
    if exist "offline\pypi" (
        echo 正在离线安装依赖（不访问网络）……
        "%PY%" -m pip install --no-index --find-links "offline\pypi" --find-links "offline\torch" -r requirements.txt
        if errorlevel 1 (
            echo [错误] 离线安装失败，请确认 offline 目录完整。
            exit /b 1
        )
        echo [OK] 依赖安装完成。
        exit /b 0
    ) else (
        echo [错误] 未找到 offline 离线包目录。
        exit /b 1
    )
)
exit /b 0

REM ==================== 子程序：构建索引 ====================
:build_index
echo 正在构建向量索引（首次加载 BGE 模型稍慢）……
"%PY%" -m src.build_index --data test_samples --rebuild
if errorlevel 1 (
    echo [错误] 索引构建失败。
) else (
    echo [OK] 索引构建完成。
)
exit /b 0
