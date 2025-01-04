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

:: 设置压缩参数（使用中等压缩比例）
set "preset=medium"
set "quality=26"
set "bitrate=15M"
set "suffix=_compressed"

echo.
echo 开始处理视频文件...
echo 目标目录: %target_dir%
echo.

:: 计数器
set "processed=0"
set "skipped=0"

:: 处理所有视频文件
for %%i in ("%target_dir%\*.mp4" "%target_dir%\*.avi" "%target_dir%\*.mov" "%target_dir%\*.mkv") do (
    if exist "%%i" (
        :: 检查文件名是否已包含压缩标记
        echo "%%~ni" | findstr /C:"_compressed" >nul
        if !errorlevel! equ 0 (
            echo 跳过已压缩文件: %%~nxi
            set /a "skipped+=1"
        ) else (
            echo 正在处理: %%~nxi
            
            :: 构建输出文件名（在原目录下创建）
            set "outfile=%%~dpi%%~ni%suffix%%%~xi"
            
            :: 使用 NVENC 进行压缩
            "%FFMPEG%" -hwaccel cuda -i "%%i" -c:v h264_nvenc ^
                -preset %preset% ^
                -rc:v vbr_hq ^
                -cq:v %quality% ^
                -b:v %bitrate% ^
                -maxrate:v %bitrate% ^
                -profile:v high ^
                -rc-lookahead 32 ^
                -spatial-aq 1 ^
                -aq-strength 8 ^
                -c:a aac ^
                -b:a 128k ^
                -movflags +faststart ^
                -y "!outfile!"
            
            if !errorlevel! equ 0 (
                echo 完成: %%~nxi
                set /a "processed+=1"
            ) else (
                echo 处理失败: %%~nxi
                if exist "!outfile!" del "!outfile!"
            )
            echo.
        )
    )
)

echo 处理完成！
echo 成功压缩: !processed! 个文件
echo 跳过已压缩: !skipped! 个文件
echo.
pause 