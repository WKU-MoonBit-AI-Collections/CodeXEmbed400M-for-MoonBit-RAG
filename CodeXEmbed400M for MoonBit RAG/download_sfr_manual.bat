@echo off
chcp 65001 >nul
echo 开始下载 SFR-Embedding-Code-400M_R 模型...
echo.

:: 创建目录
if not exist "models" mkdir models
if not exist "models\SFR-Embedding-Code-400M_R" mkdir models\SFR-Embedding-Code-400M_R
cd models\SFR-Embedding-Code-400M_R

:: 基础URL
set BASE_URL=https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R/resolve/main

echo 📁 创建目录完成，开始下载文件...
echo.

:: 检查是否有下载工具
where curl >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 未找到 curl 命令。请安装 curl 或使用 PowerShell 脚本
    echo 或者手动下载以下文件：
    echo.
    echo 1. %BASE_URL%/model.safetensors
    echo 2. %BASE_URL%/config.json
    echo 3. %BASE_URL%/tokenizer.json
    echo 4. %BASE_URL%/tokenizer_config.json
    echo 5. %BASE_URL%/vocab.txt
    echo 6. %BASE_URL%/special_tokens_map.json
    echo 7. %BASE_URL%/configuration.py
    echo.
    pause
    exit /b 1
)

:: 下载文件
echo 1/7 下载 model.safetensors (868MB)...
curl -L "%BASE_URL%/model.safetensors" -o model.safetensors
if %errorlevel% neq 0 echo ❌ model.safetensors 下载失败

echo 2/7 下载 config.json...
curl -L "%BASE_URL%/config.json" -o config.json
if %errorlevel% neq 0 echo ❌ config.json 下载失败

echo 3/7 下载 tokenizer.json (712KB)...
curl -L "%BASE_URL%/tokenizer.json" -o tokenizer.json
if %errorlevel% neq 0 echo ❌ tokenizer.json 下载失败

echo 4/7 下载 tokenizer_config.json...
curl -L "%BASE_URL%/tokenizer_config.json" -o tokenizer_config.json
if %errorlevel% neq 0 echo ❌ tokenizer_config.json 下载失败

echo 5/7 下载 vocab.txt (232KB)...
curl -L "%BASE_URL%/vocab.txt" -o vocab.txt
if %errorlevel% neq 0 echo ❌ vocab.txt 下载失败

echo 6/7 下载 special_tokens_map.json...
curl -L "%BASE_URL%/special_tokens_map.json" -o special_tokens_map.json
if %errorlevel% neq 0 echo ❌ special_tokens_map.json 下载失败

echo 7/7 下载 configuration.py...
curl -L "%BASE_URL%/configuration.py" -o configuration.py
if %errorlevel% neq 0 echo ❌ configuration.py 下载失败

echo.
echo ✅ 下载任务完成！
echo 📁 模型路径: %CD%
echo.
echo 文件列表：
dir /b
echo.
echo 🚀 现在可以运行：
echo python scripts/create-vecdb-lite.py
echo.
pause
