@echo off
setlocal enabledelayedexpansion

:: 设置输入和输出文件夹
set "input_dir=input"
set "output_dir=output"

:: 创建输出目录
if not exist "%output_dir%" mkdir "%output_dir%"

:menu
cls
echo 视频帧率调整工具
echo =====================================
echo 请选择操作模式:
echo.
echo 1. 简单帧率调整 (保持时长不变)
echo    - 只改变帧率，视频时长保持不变
echo    - 可能会丢帧或重复帧
echo.
echo 2. 帧率倍速调整 (改变视频时长)
echo    - 改变帧率同时改变播放速度
echo    - 不会丢帧，时长会相应变化
echo.
echo 3. 光流法帧率提升
echo    - 使用光流插值生成中间帧
echo    - 适合提高帧率，画面更流畅
echo    - 处理时间较长
echo.
echo 0. 退出
echo =====================================

set /p choice="请输入选择 (0-3): "

if "%choice%"=="0" goto :eof
if "%choice%"=="1" goto simple_fps
if "%choice%"=="2" goto speed_fps
if "%choice%"=="3" goto flow_fps

echo 无效选择，请重试
pause
goto menu

:simple_fps
cls
echo 简单帧率调整
echo =====================================
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p target_fps="输入目标帧率: "

:: 获取原始帧率
for /f "tokens=*" %%a in ('ffprobe -v error -select_streams v -of default^=noprint_wrappers^=1:nokey^=1 -show_entries stream^=r_frame_rate "%input_dir%\%input_file%"') do set "orig_fps=%%a"
echo 原始帧率: %orig_fps%

ffmpeg -i "%input_dir%\%input_file%" ^
    -filter:v fps=%target_fps% ^
    -c:v libx264 ^
    -preset medium ^
    -crf 18 ^
    -c:a copy ^
    "%output_dir%\fps_%target_fps%_%~n1%%~x1"

echo.
echo 处理完成！
pause
goto menu

:speed_fps
cls
echo 帧率倍速调整
echo =====================================
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p speed="输入速度倍率 (例如 2 表示两倍速): "

:: 获取原始帧率
for /f "tokens=*" %%a in ('ffprobe -v error -select_streams v -of default^=noprint_wrappers^=1:nokey^=1 -show_entries stream^=r_frame_rate "%input_dir%\%input_file%"') do set "orig_fps=%%a"
echo 原始帧率: %orig_fps%

:: 计算新帧率
set /a "target_fps=%orig_fps% * %speed%"

ffmpeg -i "%input_dir%\%input_file%" ^
    -filter:v "setpts=PTS/%speed%" ^
    -r %target_fps% ^
    -c:v libx264 ^
    -preset medium ^
    -crf 18 ^
    -af "atempo=%speed%" ^
    "%output_dir%\speed_%speed%x_%~n1%%~x1"

echo.
echo 处理完成！
pause
goto menu

:flow_fps
cls
echo 光流法帧率提升
echo =====================================
echo 注意：此方法处理时间较长，但效果最好
echo.
set /p input_file="输入视频文件名 (需放在input文件夹): "
set /p target_fps="输入目标帧率: "

ffmpeg -i "%input_dir%\%input_file%" ^
    -filter:v "minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:vsbmc=1:fps=%target_fps%'" ^
    -c:v libx264 ^
    -preset slow ^
    -crf 18 ^
    -c:a copy ^
    "%output_dir%\flow_fps_%target_fps%_%~n1%%~x1"

echo.
echo 处理完成！
pause
goto menu 