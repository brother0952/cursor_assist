@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: 设置 FFmpeg 路径
set "FFMPEG=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"

:: 配置参数
set "buffer_seconds=5"
set "post_event_seconds=5"
set "fps=30"
set "temp_dir=temp_segments"
set "output_dir=event_videos"

:: 创建必要的目录
if not exist "%temp_dir%" mkdir "%temp_dir%"
if not exist "%output_dir%" mkdir "%output_dir%"

:: 计算segment时长（秒）
set "segment_length=1"
set /a total_segments=buffer_seconds/segment_length

echo 事件视频录制工具
echo =====================================
echo 缓冲时长: %buffer_seconds%秒
echo 事件后录制: %post_event_seconds%秒
echo 帧率: %fps%
echo.
echo 按任意键开始录制，发生事件时按空格键...
pause >nul

:record_loop
:: 清理旧的临时文件
del /q "%temp_dir%\segment_*.mp4" 2>nul

:: 开始循环录制
set "segment_index=0"
:segment_loop
:: 录制一个片段
"%FFMPEG%" -y -f dshow -framerate %fps% -video_size 1280x720 -i video="USB Camera" ^
    -t %segment_length% -c:v h264_nvenc -preset p7 -b:v 5M ^
    "%temp_dir%\segment_%segment_index%.mp4"

:: 检查是否按下空格键
choice /c " q" /n /t 1 /d " " >nul
if errorlevel 2 goto :end
if errorlevel 1 goto :event_triggered

:: 更新segment索引
set /a "segment_index=(segment_index + 1) %% total_segments"
goto :segment_loop

:event_triggered
:: 事件触发后，继续录制指定时长
echo 事件已触发，继续录制%post_event_seconds%秒...
"%FFMPEG%" -y -f dshow -framerate %fps% -video_size 1280x720 -i video="USB Camera" ^
    -t %post_event_seconds% -c:v h264_nvenc -preset p7 -b:v 5M ^
    "%temp_dir%\post_event.mp4"

:: 生成片段列表
echo file '%temp_dir%\segment_0.mp4' > "%temp_dir%\list.txt"
for /l %%i in (1,1,%total_segments%) do (
    echo file '%temp_dir%\segment_%%i.mp4' >> "%temp_dir%\list.txt"
)
echo file '%temp_dir%\post_event.mp4' >> "%temp_dir%\list.txt"

:: 合并所有片段
set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=!timestamp: =0!"
"%FFMPEG%" -f concat -safe 0 -i "%temp_dir%\list.txt" -c copy ^
    "%output_dir%\event_!timestamp!.mp4"

echo 视频已保存：%output_dir%\event_!timestamp!.mp4

:end
:: 清理临时文件
del /q "%temp_dir%\*.mp4" 2>nul
del /q "%temp_dir%\list.txt" 2>nul

echo 录制已结束
pause 