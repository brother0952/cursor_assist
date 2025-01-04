@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: 检查参数
if "%~1"=="" (
    echo 用法: compress.bat [视频目录路径]
    echo 示例: compress.bat D:\Videos
    exit /b 1
)

:: 设置路径
set "FFMPEG=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"
set "target_dir=%~1"

:: 检查目录是否存在
if not exist "%target_dir%" (
    echo 错误: 目录 "%target_dir%" 不存在
    exit /b 1
)

:: 设置基本压缩参数
set "preset=medium"
set "quality=26"
set "max_bitrate=15M"
set "suffix=_compressed"

echo.
echo 开始处理视频文件...
echo 目标目录: %target_dir%
echo.

:: 计数器
set "processed=0"
set "skipped=0"

:: 创建临时文件列表
set "temp_file=%temp%\video_list.txt"
if exist "%temp_file%" del "%temp_file%"

:: 递归查找所有视频文件
for /r "%target_dir%" %%i in (*.mp4 *.avi *.mov *.mkv) do (
    echo %%i >> "%temp_file%"
)

:: 处理找到的所有视频文件
for /f "usebackq delims=" %%i in ("%temp_file%") do (
    :: 检查是否是压缩后的文件
    echo "%%~ni" | findstr /C:"_compressed" >nul
    if !errorlevel! equ 0 (
        echo 跳过压缩后的文件: %%~nxi
        set /a "skipped+=1"
    ) else (
        :: 检查是否已存在压缩后的文件
        if exist "%%~dpi%%~ni%suffix%%%~xi" (
            echo 跳过已有压缩版本的文件: %%~nxi
            set /a "skipped+=1"
        ) else (
            echo.
            echo 处理文件: %%~nxi
            echo 所在目录: %%~dpi
            
            :: 获取原始视频的码率（以bps为单位）
            for /f "tokens=*" %%b in ('"%FFMPEG%" -i "%%i" 2^>^&1 ^| findstr "bitrate" ^| find /v "Stream"') do (
                set "bitrate_info=%%b"
            )
            
            :: 提取码率数值（kbps）
            for /f "tokens=6 delims=:, " %%b in ("!bitrate_info!") do (
                set /a "orig_bitrate=%%b"
            )
            
            :: 将kbps转换为Mbps并设置目标码率
            set /a "orig_bitrate_m=orig_bitrate/1000"
            if !orig_bitrate_m! geq 15 (
                set "target_bitrate=15M"
            ) else (
                set "target_bitrate=!orig_bitrate_m!M"
            )
            
            echo 原始码率: !orig_bitrate_m!Mbps
            echo 目标码率: !target_bitrate!
            
            :: 构建输出文件名（保持在原目录）
            set "outfile=%%~dpi%%~ni%suffix%%%~xi"
            
            :: 使用 NVENC 进行压缩
            "%FFMPEG%" -hwaccel cuda -i "%%i" -c:v h264_nvenc ^
                -preset %preset% ^
                -rc:v vbr_hq ^
                -cq:v %quality% ^
                -b:v !target_bitrate! ^
                -maxrate:v !target_bitrate! ^
                -profile:v high ^
                -rc-lookahead 32 ^
                -spatial-aq 1 ^
                -aq-strength 8 ^
                -c:a copy ^
                -movflags +faststart ^
                -y "!outfile!"
            
            if !errorlevel! equ 0 (
                echo 完成: %%~nxi
                set /a "processed+=1"
            ) else (
                echo 处理失败: %%~nxi
                if exist "!outfile!" del "!outfile!"
            )
        )
    )
)

:: 删除临时文件
if exist "%temp_file%" del "%temp_file%"

echo.
echo 处理完成！
echo 成功压缩: !processed! 个文件
echo 跳过已压缩: !skipped! 个文件
echo.
pause 