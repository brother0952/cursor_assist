@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set "FFMPEG=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"
set "input_dir=input"
set "output_dir=output"

if not exist "%output_dir%" mkdir "%output_dir%"

:menu
echo.
echo =====================================
echo           Video Compressor
echo =====================================
echo.
echo  [1] GPU - High Quality
echo  [2] GPU - Medium Quality
echo  [3] GPU - Low Quality
echo  [4] CPU - High Quality
echo  [5] CPU - Medium Quality
echo  [0] Exit
echo.
echo =====================================
echo.

set /p choice="Select option (0-5): "

if "%choice%"=="0" exit /b
if "%choice%"=="1" goto gpu_high
if "%choice%"=="2" goto gpu_medium
if "%choice%"=="3" goto gpu_low
if "%choice%"=="4" goto cpu_high
if "%choice%"=="5" goto cpu_medium
goto menu

:gpu_high
set "preset=medium"
set "quality=18"
set "suffix=_gpu_hq"
set "gpu=1"
goto process

:gpu_medium
set "preset=medium"
set "quality=23"
set "suffix=_gpu_mq"
set "gpu=1"
goto process

:gpu_low
set "preset=fast"
set "quality=28"
set "suffix=_gpu_lq"
set "gpu=1"
goto process

:cpu_high
set "preset=slow"
set "quality=23"
set "suffix=_cpu_hq"
set "gpu=0"
goto process

:cpu_medium
set "preset=medium"
set "quality=28"
set "suffix=_cpu_mq"
set "gpu=0"
goto process

:process
echo.
echo Processing videos...
echo.

for %%i in ("%input_dir%\*.mp4" "%input_dir%\*.avi" "%input_dir%\*.mov") do (
    if exist "%%i" (
        echo Processing: %%~nxi
        set "outfile=%output_dir%\%%~ni%suffix%%%~xi"
        
        if "%gpu%"=="1" (
            "%FFMPEG%" -hwaccel cuda -i "%%i" -c:v h264_nvenc -preset %preset% -cq %quality% -rc-lookahead 32 -c:a aac -b:a 128k -movflags +faststart -y "!outfile!"
        ) else (
            "%FFMPEG%" -i "%%i" -c:v libx264 -crf %quality% -preset %preset% -c:a aac -b:a 128k -movflags +faststart -y "!outfile!"
        )
        
        echo Completed: %%~nxi
        echo.
    )
)

echo All videos processed!
echo.
pause
goto menu 