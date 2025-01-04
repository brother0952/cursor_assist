@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: 设置 FFmpeg 路径
set "FFMPEG=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"

:: 设置输入和输出文件夹
set "input_dir=input"
set "output_dir=output"

:: 创建输出目录
if not exist "%output_dir%" mkdir "%output_dir%"

:: 检查是否支持NVIDIA编码
set "has_nvidia=0"
"%FFMPEG%" -hide_banner -encoders > encoders.txt
findstr /C:"h264_nvenc" encoders.txt > nul
if !errorlevel! equ 0 (
    set "has_nvidia=1"
)
del encoders.txt

:: 设置默认CPU配置
set "profile_count=5"

:: 设置CPU配置
set "p1_name=超高画质"
set "p1_quality=18"
set "p1_preset=veryslow"
set "p1_suffix=_uhq"
set "p1_encoder=libx264"

set "p2_name=高画质"
set "p2_quality=23"
set "p2_preset=slow"
set "p2_suffix=_hq"
set "p2_encoder=libx264"

set "p3_name=中等画质"
set "p3_quality=28"
set "p3_preset=medium"
set "p3_suffix=_mq"
set "p3_encoder=libx264"

set "p4_name=低画质高压缩"
set "p4_quality=33"
set "p4_preset=veryfast"
set "p4_suffix=_lq"
set "p4_encoder=libx264"

set "p5_name=极限压缩"
set "p5_quality=40"
set "p5_preset=ultrafast"
set "p5_suffix=_min"
set "p5_encoder=libx264"

if "!has_nvidia!"=="1" (
    set "profile_count=7"
    
    REM 设置GPU配置
    set "p1_name=超高画质(GPU)"
    set "p1_quality=18"
    set "p1_preset=medium"
    set "p1_suffix=_uhq_gpu"
    set "p1_encoder=h264_nvenc"
    
    set "p2_name=高画质(GPU)"
    set "p2_quality=23"
    set "p2_preset=medium"
    set "p2_suffix=_hq_gpu"
    set "p2_encoder=h264_nvenc"
    
    set "p3_name=中等画质(GPU)"
    set "p3_quality=28"
    set "p3_preset=medium"
    set "p3_suffix=_mq_gpu"
    set "p3_encoder=h264_nvenc"
    
    set "p4_name=低画质高压缩(GPU)"
    set "p4_quality=33"
    set "p4_preset=fast"
    set "p4_suffix=_lq_gpu"
    set "p4_encoder=h264_nvenc"
    
    set "p5_name=极限压缩(GPU)"
    set "p5_quality=40"
    set "p5_preset=fast"
    set "p5_suffix=_min_gpu"
    set "p5_encoder=h264_nvenc"
    
    set "p6_name=超高画质(CPU)"
    set "p6_quality=18"
    set "p6_preset=veryslow"
    set "p6_suffix=_uhq_cpu"
    set "p6_encoder=libx264"
    
    set "p7_name=高画质(CPU)"
    set "p7_quality=23"
    set "p7_preset=slow"
    set "p7_suffix=_hq_cpu"
    set "p7_encoder=libx264"
)

:menu
echo.
echo 视频压缩工具
echo =====================================
echo 请选择压缩配置:
echo.

if "!has_nvidia!"=="1" (
    echo --- GPU加速模式 (NVIDIA) ---
    echo 1. 超高画质 (GPU加速)
    echo    - 质量: 最佳
    echo    - 预设: medium
    echo    - 压缩率: 60-70%%
    echo.
    echo 2. 高画质 (GPU加速, 推荐)
    echo    - 质量: 很好
    echo    - 预设: medium
    echo    - 压缩率: 40-50%%
    echo.
    echo 3. 中等画质 (GPU加速)
    echo    - 质量: 良好
    echo    - 预设: medium
    echo    - 压缩率: 30-40%%
    echo.
    echo 4. 低画质高压缩 (GPU加速)
    echo    - 质量: 一般
    echo    - 预设: fast
    echo    - 压缩率: 20-30%%
    echo.
    echo 5. 极限压缩 (GPU加速)
    echo    - 质量: 较差
    echo    - 预设: fast
    echo    - 压缩率: 10-20%%
    echo.
    echo --- CPU模式 ---
    echo 6. 超高画质 (CPU)
    echo 7. 高画质 (CPU)
) else (
    echo --- CPU模式 ---
    echo 1. 超高画质
    echo    - CRF: 18
    echo    - 预设: veryslow
    echo    - 压缩率: 60-70%%
    echo.
    echo 2. 高画质 (推荐)
    echo    - CRF: 23
    echo    - 预设: slow
    echo    - 压缩率: 40-50%%
    echo.
    echo 3. 中等画质
    echo    - CRF: 28
    echo    - 预设: medium
    echo    - 压缩率: 30-40%%
    echo.
    echo 4. 低画质高压缩
    echo    - CRF: 33
    echo    - 预设: veryfast
    echo    - 压缩率: 20-30%%
    echo.
    echo 5. 极限压缩
    echo    - CRF: 40
    echo    - 预设: ultrafast
    echo    - 压缩率: 10-20%%
)

echo.
echo 0. 退出
echo =====================================

set /p "choice=请输入选择 (0-!profile_count!): "

if "!choice!"=="0" goto :eof
if !choice! gtr !profile_count! (
    echo 无效选择，请重试
    pause
    goto menu
)

:: 获取选择的配置
set "name=!p%choice%_name!"
set "quality=!p%choice%_quality!"
set "preset=!p%choice%_preset!"
set "suffix=!p%choice%_suffix!"
set "encoder=!p%choice%_encoder!"

echo.
echo 已选择: !name!
echo 开始处理视频...
echo.

:: 处理所有视频文件
for %%i in ("%input_dir%\*.mp4" "%input_dir%\*.avi" "%input_dir%\*.mov") do (
    if exist "%%i" (
        echo 正在处理: %%~nxi
        
        :: 构建输出文件名
        set "output_file=%output_dir%\%%~ni!suffix!%%~xi"
        
        :: 根据编码器选择压缩命令
        if "!encoder!"=="h264_nvenc" (
            :: GPU编码
            "%FFMPEG%" -hwaccel cuda -i "%%i" -c:v h264_nvenc -preset !preset! -cq !quality! -rc-lookahead 32 -c:a aac -b:a 128k -movflags +faststart -y "!output_file!"
        ) else (
            :: CPU编码
            "%FFMPEG%" -i "%%i" -c:v libx264 -crf !quality! -preset !preset! -c:a aac -b:a 128k -movflags +faststart -y "!output_file!"
        )
            
        echo 完成: %%~nxi
        echo.
    )
)

echo 所有视频处理完成！
echo.
pause
goto menu 