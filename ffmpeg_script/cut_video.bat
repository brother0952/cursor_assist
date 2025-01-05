@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置 FFmpeg 路径
set "FFMPEG_PATH=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin"
set "PATH=%FFMPEG_PATH%;%PATH%"

:: 设置控制台字体
reg add "HKEY_CURRENT_USER\Console" /v "FaceName" /t REG_SZ /d "Consolas" /f >nul
reg add "HKEY_CURRENT_USER\Console" /v "FontFamily" /t REG_DWORD /d 54 /f >nul

:: 设置输入和输出文件夹
set "input_dir=input"
set "output_dir=output"

:: 创建输出目录
if not exist "%output_dir%" mkdir "%output_dir%"

:menu
cls
echo 视频切割工具
echo =====================================
echo 请选择切割方式:
echo.
echo 1. 按时间段切割 (指定开始和结束时间)
echo 2. 按时长切割 (从开始时间切指定时长)
echo 3. 按时间间隔切割 (将视频分割为等长片段)
echo 4. 精确帧切割 (指定开始和结束帧)
echo 5. 剪掉头部 (删除指定时间之前的部分)
echo 6. 剪掉尾部 (删除指定时间之后的部分)
echo.
echo 0. 退出
echo =====================================

set /p choice="请输入选择 (0-6): "

if "%choice%"=="0" goto :eof
if "%choice%"=="1" goto time_range
if "%choice%"=="2" goto duration
if "%choice%"=="3" goto segments
if "%choice%"=="4" goto frame_range
if "%choice%"=="5" goto trim_head
if "%choice%"=="6" goto trim_tail

echo 无效选择，请重试
pause
goto menu

:time_range
cls
echo 按时间段切割
echo =====================================
echo 时间格式: HH:MM:SS 或 HH:MM:SS.xxx
echo 示例: 00:01:30 或 00:01:30.500
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p start_time="输入开始时间: "
set /p end_time="输入结束时间: "

ffmpeg -i "%input_dir%\%input_file%" ^
    -ss %start_time% ^
    -to %end_time% ^
    -c:v copy ^
    -c:a copy ^
    -avoid_negative_ts 1 ^
    "%output_dir%\cut_%~n1_%start_time:.=_%-%end_time:.=_%%~x1"

echo.
echo 切割完成！
pause
goto menu

:duration
cls
echo 按时长切割
echo =====================================
echo 时间格式: HH:MM:SS 或 HH:MM:SS.xxx
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p start_time="输入开始时间: "
set /p duration="输入持续时长: "

ffmpeg -i "%input_dir%\%input_file%" ^
    -ss %start_time% ^
    -t %duration% ^
    -c:v copy ^
    -c:a copy ^
    -avoid_negative_ts 1 ^
    "%output_dir%\cut_%~n1_%start_time:.=_%-%duration:.=_%%~x1"

echo.
echo 切割完成！
pause
goto menu

:segments
cls
echo 按时间间隔切割
echo =====================================
echo 将视频分割为等长片段
echo 时间格式: HH:MM:SS
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p segment_time="输入每段时长: "

ffmpeg -i "%input_dir%\%input_file%" ^
    -f segment ^
    -segment_time %segment_time% ^
    -reset_timestamps 1 ^
    -c copy ^
    "%output_dir%\%~n1_segment_%%03d%%~x1"

echo.
echo 切割完成！
pause
goto menu

:frame_range
cls
echo 精确帧切割
echo =====================================
echo 注意：此模式会重新编码视频
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p start_frame="输入开始帧号: "
set /p end_frame="输入结束帧号: "

:: 首先获取视频的帧率
for /f "tokens=*" %%a in ('ffprobe -v error -select_streams v -of default^=noprint_wrappers^=1:nokey^=1 -show_entries stream^=r_frame_rate "%input_dir%\%input_file%"') do set "fps=%%a"

:: 计算时间戳
set /a start_seconds=start_frame/fps
set /a end_seconds=end_frame/fps

ffmpeg -i "%input_dir%\%input_file%" ^
    -vf "select='between(n\,%start_frame%\,%end_frame%)'" ^
    -c:v libx264 ^
    -crf 18 ^
    -preset fast ^
    -c:a copy ^
    "%output_dir%\cut_%~n1_frame_%start_frame%-%end_frame%%%~x1"

echo.
echo 切割完成！
pause
goto menu

:trim_head
cls
echo 剪掉头部
echo =====================================
echo 删除指定时间之前的部分
echo 时间格式: HH:MM:SS 或 HH:MM:SS.xxx
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p cut_time="输入要删除到的时间点: "

:: 获取视频总时长
for /f "tokens=*" %%a in ('ffprobe -v error -show_entries format^=duration -of default^=noprint_wrappers^=1:nokey^=1 "%input_dir%\%input_file%"') do set "duration=%%a"

ffmpeg -i "%input_dir%\%input_file%" ^
    -ss %cut_time% ^
    -c:v copy ^
    -c:a copy ^
    -avoid_negative_ts 1 ^
    "%output_dir%\cut_%~n1_from_%cut_time:.=_%%~x1"

echo.
echo 切割完成！
pause
goto menu

:trim_tail
cls
echo 剪掉尾部
echo =====================================
echo 删除指定时间之后的部分
echo 时间格式: HH:MM:SS 或 HH:MM:SS.xxx
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p cut_time="输入要保留到的时间点: "

ffmpeg -i "%input_dir%\%input_file%" ^
    -t %cut_time% ^
    -c:v copy ^
    -c:a copy ^
    -avoid_negative_ts 1 ^
    "%output_dir%\cut_%~n1_to_%cut_time:.=_%%~x1"

echo.
echo 切割完成！
pause
goto menu 