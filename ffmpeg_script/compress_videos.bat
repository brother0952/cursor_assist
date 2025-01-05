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

:: 设置FPS控制选项（默认关闭）
set "control_fps=0"

:fps_menu
cls
echo FPS控制设置
echo =====================================
echo 当前状态: !control_fps!
echo [0] 关闭 - 保持原始FPS
echo [1] 开启 - 60FPS自动降至30FPS
echo.
set /p fps_choice="请选择FPS控制模式 (0/1): "

if "!fps_choice!"=="0" (
    set "control_fps=0"
) else if "!fps_choice!"=="1" (
    set "control_fps=1"
) else (
    echo 无效选择，使用默认值（关闭）
    set "control_fps=0"
)

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
        
        :: 获取视频FPS
        for /f "tokens=*" %%f in ('"%FFMPEG%" -i "%%i" 2^>^&1 ^| findstr "fps"') do (
            set "fps_info=%%f"
        )
        :: 提取FPS值
        for /f "tokens=2 delims=," %%f in ("!fps_info!") do (
            set "fps_value=%%f"
        )
        set "fps_value=!fps_value: fps=!"
        set /a "fps_num=!fps_value!"
        
        echo 原始FPS: !fps_num!
        
        :: 获取原始视频的码率（以kbps为单位）
        for /f "tokens=*" %%b in ('"%FFMPEG%" -i "%%i" 2^>^&1 ^| findstr "bitrate"') do (
            set "bitrate_info=%%b"
        )
        
        :: 提取码率数值（kbps）
        for /f "tokens=6 delims=:, " %%b in ("!bitrate_info!") do (
            set /a "orig_bitrate=%%b"
        )
        
        :: 获取文件大小（以字节为单位）
        for %%s in ("%%i") do set "orig_size=%%~zs"
        
        :: 将码率从kbps转换为Mbps
        set /a "orig_bitrate_m=orig_bitrate/1000"
        echo 原始码率: !orig_bitrate_m! Mbps
        echo 原始大小: !orig_size! 字节
        
        :: 构建输出文件名
        set "output_file=%output_dir%\%%~ni!suffix!%%~xi"
        
        :: 根据编码器和原始码率选择压缩策略
        if "!encoder!"=="h264_nvenc" (
            :: GPU编码
            if !orig_bitrate_m! leq 2 (
                echo 原始码率已经很低，跳过压缩
                echo.
                continue
            ) else (
                :: 设置目标码率为原始码率的一半，但不超过15Mbps
                set /a "target_bitrate=orig_bitrate_m/2"
                if !target_bitrate! gtr 15 set "target_bitrate=15"
                echo 目标码率: !target_bitrate! Mbps
                
                :: 根据FPS控制选项和当前FPS决定是否需要降帧
                if "!control_fps!"=="1" if !fps_num! geq 50 (
                    echo 检测到高帧率，降至30FPS
                    "%FFMPEG%" -hwaccel cuda -i "%%i" -c:v h264_nvenc ^
                        -preset !preset! ^
                        -rc:v vbr_hq ^
                        -cq:v !quality! ^
                        -b:v !target_bitrate!M ^
                        -maxrate:v !target_bitrate!M ^
                        -rc-lookahead 32 ^
                        -spatial-aq 1 ^
                        -aq-strength 8 ^
                        -r 30 ^
                        -c:a copy ^
                        -movflags +faststart ^
                        -y "!output_file!"
                ) else (
                    "%FFMPEG%" -hwaccel cuda -i "%%i" -c:v h264_nvenc ^
                        -preset !preset! ^
                        -rc:v vbr_hq ^
                        -cq:v !quality! ^
                        -b:v !target_bitrate!M ^
                        -maxrate:v !target_bitrate!M ^
                        -rc-lookahead 32 ^
                        -spatial-aq 1 ^
                        -aq-strength 8 ^
                        -c:a copy ^
                        -movflags +faststart ^
                        -y "!output_file!"
                )
            )
        ) else (
            :: CPU编码
            if !orig_bitrate_m! leq 2 (
                echo 原始码率已经很低，跳过压缩
                echo.
                continue
            ) else (
                :: 使用CRF模式，但设置最大码率
                set /a "target_bitrate=orig_bitrate_m/2"
                if !target_bitrate! gtr 15 set "target_bitrate=15"
                echo 目标码率: !target_bitrate! Mbps
                
                :: 根据FPS控制选项和当前FPS决定是否需要降帧
                if "!control_fps!"=="1" if !fps_num! geq 50 (
                    echo 检测到高帧率，降至30FPS
                    "%FFMPEG%" -i "%%i" -c:v libx264 ^
                        -crf !quality! ^
                        -maxrate:v !target_bitrate!M ^
                        -bufsize !target_bitrate!M ^
                        -preset !preset! ^
                        -r 30 ^
                        -c:a copy ^
                        -movflags +faststart ^
                        -y "!output_file!"
                ) else (
                    "%FFMPEG%" -i "%%i" -c:v libx264 ^
                        -crf !quality! ^
                        -maxrate:v !target_bitrate!M ^
                        -bufsize !target_bitrate!M ^
                        -preset !preset! ^
                        -c:a copy ^
                        -movflags +faststart ^
                        -y "!output_file!"
                )
            )
        )
        
        :: 检查压缩结果
        if exist "!output_file!" (
            for %%s in ("!output_file!") do set "new_size=%%~zs"
            echo 新文件大小: !new_size! 字节
            
            :: 如果新文件更大，则删除它并保留原始文件
            if !new_size! gtr !orig_size! (
                echo 警告：压缩后文件更大，保留原始文件
                del "!output_file!"
            ) else (
                set /a "saved_space=orig_size-new_size"
                set /a "saved_percent=saved_space*100/orig_size"
                echo 节省空间: !saved_percent!%%
            )
        )
            
        echo 完成: %%~nxi
        echo.
    )
)

echo 所有视频处理完成！
echo.
pause
goto menu 