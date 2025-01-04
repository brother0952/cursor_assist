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
echo  [1] Light Compression (90% quality)
echo  [2] Medium Compression (80% quality)
echo  [3] High Compression (70% quality)
echo  [4] Ultra Compression (60% quality)
echo  [0] Exit
echo.
echo =====================================
echo.

set /p choice="Select option (0-4): "

if "%choice%"=="0" exit /b
if "%choice%"=="1" goto quality_90
if "%choice%"=="2" goto quality_80
if "%choice%"=="3" goto quality_70
if "%choice%"=="4" goto quality_60
goto menu

:quality_90
set "preset=medium"
set "quality=23"
set "bitrate=20M"
set "suffix=_90"
goto process

:quality_80
set "preset=medium"
set "quality=26"
set "bitrate=15M"
set "suffix=_80"
goto process

:quality_70
set "preset=medium"
set "quality=28"
set "bitrate=10M"
set "suffix=_70"
goto process

:quality_60
set "preset=medium"
set "quality=30"
set "bitrate=8M"
set "suffix=_60"
goto process

:process
echo.
echo Processing videos...
echo.

for %%i in ("%input_dir%\*.mp4" "%input_dir%\*.avi" "%input_dir%\*.mov") do (
    if exist "%%i" (
        echo Processing: %%~nxi
        set "outfile=%output_dir%\%%~ni%suffix%%%~xi"
        
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
        
        echo Completed: %%~nxi
        echo.
    )
)

echo All videos processed!
echo.
pause
goto menu 