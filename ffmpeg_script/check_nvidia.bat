@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 开始检测NVIDIA编码器...

:: 设置完整的FFmpeg路径
set "FFMPEG=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"

echo 1. 检查 FFmpeg 是否可用...
"%FFMPEG%" -version >nul 2>&1
if !errorlevel! neq 0 (
    echo [错误] FFmpeg 未找到或无法运行: %FFMPEG%
    goto :error
)
echo [成功] FFmpeg 可用

echo.
echo 2. 检查支持的编码器...
"%FFMPEG%" -hide_banner -encoders > encoders.txt
findstr /C:"h264_nvenc" encoders.txt >nul
if !errorlevel! equ 0 (
    echo [成功] 找到 NVIDIA 编码器 (h264_nvenc^)
    echo 编码器详细信息:
    findstr /C:"h264_nvenc" encoders.txt
) else (
    echo [错误] 未找到 NVIDIA 编码器
    echo 请确保：
    echo 1. 已安装 NVIDIA 显卡驱动
    echo 2. 显卡支持 NVENC 编码
    echo 3. FFmpeg 版本支持 NVIDIA 编码
    del encoders.txt
    goto :error
)
del encoders.txt

echo.
echo 3. 测试 NVIDIA 编码功能...
"%FFMPEG%" -hide_banner -y -f lavfi -i color=c=black:s=1280x720:d=1 -c:v h264_nvenc -preset p7 -f null - 2>nul
if !errorlevel! equ 0 (
    echo [成功] NVIDIA 编码器工作正常
) else (
    echo [错误] NVIDIA 编码器测试失败
    echo 尝试使用不同的预设...
    "%FFMPEG%" -hide_banner -y -f lavfi -i color=c=black:s=1280x720:d=1 -c:v h264_nvenc -preset medium -f null - 2>nul
    if !errorlevel! equ 0 (
        echo [成功] NVIDIA 编码器使用 medium 预设工作正常
    ) else (
        echo [错误] NVIDIA 编码器测试失败
        goto :error
    )
)

echo.
echo [成功] 所有检查完成，NVIDIA 编码器可以正常使用！
goto :end

:error
echo.
echo [错误] 检测过程中发现错误，请解决上述问题后再运行视频压缩脚本。

:end
pause 