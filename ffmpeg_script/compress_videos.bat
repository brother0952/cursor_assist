@echo off
setlocal enabledelayedexpansion

:: 设置输入和输出文件夹
set "input_dir=input"
set "output_dir=output"

:: 创建输出目录
if not exist "%output_dir%" mkdir "%output_dir%"

:: 定义压缩配置
:: 格式: 名称;crf值;预设;输出后缀
set "profiles[1]=超高画质;18;veryslow;_uhq"
set "profiles[2]=高画质;23;slow;_hq"
set "profiles[3]=中等画质;28;medium;_mq"
set "profiles[4]=低画质高压缩;33;veryfast;_lq"
set "profiles[5]=极限压缩;40;ultrafast;_min"

:menu
cls
echo 视频压缩工具
echo =====================================
echo 请选择压缩配置:
echo.
echo 1. 超高画质 (文件较大，最佳质量)
echo    - CRF: 18
echo    - 预设: veryslow
echo    - 压缩率: ~60-70%%
echo.
echo 2. 高画质 (推荐)
echo    - CRF: 23
echo    - 预设: slow
echo    - 压缩率: ~40-50%%
echo.
echo 3. 中等画质
echo    - CRF: 28
echo    - 预设: medium
echo    - 压缩率: ~30-40%%
echo.
echo 4. 低画质高压缩
echo    - CRF: 33
echo    - 预设: veryfast
echo    - 压缩率: ~20-30%%
echo.
echo 5. 极限压缩 (质量较差)
echo    - CRF: 40
echo    - 预设: ultrafast
echo    - 压缩率: ~10-20%%
echo.
echo 0. 退出
echo =====================================

set /p choice="请输入选择 (0-5): "

if "%choice%"=="0" goto :eof
if not defined profiles[%choice%] (
    echo 无效选择，请重试
    pause
    goto menu
)

:: 解析选择的配置
for /f "tokens=1-4 delims=;" %%a in ("!profiles[%choice%]!") do (
    set "name=%%a"
    set "crf=%%b"
    set "preset=%%c"
    set "suffix=%%d"
)

echo.
echo 已选择: %name%
echo 开始处理视频...
echo.

:: 处理所有视频文件
for %%i in ("%input_dir%\*.mp4" "%input_dir%\*.avi" "%input_dir%\*.mov") do (
    if exist "%%i" (
        echo 正在处理: %%~nxi
        
        :: 构建输出文件名
        set "output_file=%output_dir%\%%~ni%suffix%%%~xi"
        
        :: 压缩视频
        ffmpeg -i "%%i" ^
            -c:v libx264 ^
            -crf %crf% ^
            -preset %preset% ^
            -c:a aac ^
            -b:a 128k ^
            -movflags +faststart ^
            -y ^
            "!output_file!"
            
        echo 完成: %%~nxi
        echo.
    )
)

echo 所有视频处理完成！
echo.
pause
goto menu 