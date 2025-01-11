@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: Set FFmpeg path
set "FFMPEG=F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"

:: Set input and output folders
set "input_dir=input"
set "output_dir=output"

:: Create output directory
if not exist "%output_dir%" mkdir "%output_dir%"

:: Set FPS control option (default: off)
set "control_fps=0"

:fps_menu
cls
echo FPS Control Settings
echo =====================================
echo Current Status: !control_fps!
echo [0] Off - Keep Original FPS
echo [1] On - Auto reduce 60FPS to 30FPS
echo.
set /p fps_choice="Select FPS control mode (0/1): "

if "!fps_choice!"=="0" (
    set "control_fps=0"
) else if "!fps_choice!"=="1" (
    set "control_fps=1"
) else (
    echo Invalid choice, using default (Off)
    set "control_fps=0"
)

:: Check NVIDIA encoder support
set "has_nvidia=0"
"%FFMPEG%" -hide_banner -encoders > encoders.txt
findstr /C:"h264_nvenc" encoders.txt > nul
if !errorlevel! equ 0 (
    set "has_nvidia=1"
)
del encoders.txt

:: Set default CPU profiles count
set "profile_count=5"

:: Set CPU profiles
set "p1_name=Ultra High Quality"
set "p1_quality=18"
set "p1_preset=veryslow"
set "p1_suffix=_uhq"
set "p1_encoder=libx264"

set "p2_name=High Quality"
set "p2_quality=23"
set "p2_preset=slow"
set "p2_suffix=_hq"
set "p2_encoder=libx264"

set "p3_name=Medium Quality"
set "p3_quality=28"
set "p3_preset=medium"
set "p3_suffix=_mq"
set "p3_encoder=libx264"

set "p4_name=Low Quality High Compression"
set "p4_quality=33"
set "p4_preset=veryfast"
set "p4_suffix=_lq"
set "p4_encoder=libx264"

set "p5_name=Maximum Compression"
set "p5_quality=40"
set "p5_preset=ultrafast"
set "p5_suffix=_min"
set "p5_encoder=libx264"

if "!has_nvidia!"=="1" (
    set "profile_count=7"
    
    REM Set GPU profiles
    set "p1_name=Ultra High Quality (GPU)"
    set "p1_quality=18"
    set "p1_preset=medium"
    set "p1_suffix=_uhq_gpu"
    set "p1_encoder=h264_nvenc"
    
    set "p2_name=High Quality (GPU)"
    set "p2_quality=23"
    set "p2_preset=medium"
    set "p2_suffix=_hq_gpu"
    set "p2_encoder=h264_nvenc"
    
    set "p3_name=Medium Quality (GPU)"
    set "p3_quality=28"
    set "p3_preset=medium"
    set "p3_suffix=_mq_gpu"
    set "p3_encoder=h264_nvenc"
    
    set "p4_name=Low Quality High Compression (GPU)"
    set "p4_quality=33"
    set "p4_preset=fast"
    set "p4_suffix=_lq_gpu"
    set "p4_encoder=h264_nvenc"
    
    set "p5_name=Maximum Compression (GPU)"
    set "p5_quality=40"
    set "p5_preset=fast"
    set "p5_suffix=_min_gpu"
    set "p5_encoder=h264_nvenc"
    
    set "p6_name=Ultra High Quality (CPU)"
    set "p6_quality=18"
    set "p6_preset=veryslow"
    set "p6_suffix=_uhq_cpu"
    set "p6_encoder=libx264"
    
    set "p7_name=High Quality (CPU)"
    set "p7_quality=23"
    set "p7_preset=slow"
    set "p7_suffix=_hq_cpu"
    set "p7_encoder=libx264"
)

:menu
echo.
echo Video Compression Tool
echo =====================================
echo Select compression profile:
echo.

if not "!has_nvidia!"=="1" goto cpu_menu

:gpu_menu
echo === GPU Acceleration Mode (NVIDIA) ===
echo 1. Ultra High Quality (GPU)
echo    - Quality: Best
echo    - Preset: medium
echo    - Compression ratio: 0.6-0.7x
echo.
echo 2. High Quality (GPU, Recommended)
echo    - Quality: Very Good
echo    - Preset: medium
echo    - Compression ratio: 0.4-0.5x
echo.
echo 3. Medium Quality (GPU)
echo    - Quality: Good
echo    - Preset: medium
echo    - Compression ratio: 0.3-0.4x
echo.
echo 4. Low Quality High Compression (GPU)
echo    - Quality: Fair
echo    - Preset: fast
echo    - Compression ratio: 0.2-0.3x
echo.
echo 5. Maximum Compression (GPU)
echo    - Quality: Poor
echo    - Preset: fast
echo    - Compression ratio: 0.1-0.2x
echo.
echo === CPU Mode ===
echo 6. Ultra High Quality (CPU)
echo 7. High Quality (CPU)
goto menu_end

:cpu_menu
echo === CPU Mode ===
echo 1. Ultra High Quality
echo    - CRF: 18
echo    - Preset: veryslow
echo    - Compression ratio: 0.6-0.7x
echo.
echo 2. High Quality (Recommended)
echo    - CRF: 23
echo    - Preset: slow
echo    - Compression ratio: 0.4-0.5x
echo.
echo 3. Medium Quality
echo    - CRF: 28
echo    - Preset: medium
echo    - Compression ratio: 0.3-0.4x
echo.
echo 4. Low Quality High Compression
echo    - CRF: 33
echo    - Preset: veryfast
echo    - Compression ratio: 0.2-0.3x
echo.
echo 5. Maximum Compression
echo    - CRF: 40
echo    - Preset: ultrafast
echo    - Compression ratio: 0.1-0.2x

:menu_end
echo.
echo 0. Exit
echo =====================================

set /p "choice=Enter your choice (0-!profile_count!): "

if "!choice!"=="0" goto :eof
if !choice! gtr !profile_count! (
    echo Invalid choice, please try again
    pause
    goto menu
)

:: Get selected profile
set "name=!p%choice%_name!"
set "quality=!p%choice%_quality!"
set "preset=!p%choice%_preset!"
set "suffix=!p%choice%_suffix!"
set "encoder=!p%choice%_encoder!"

echo.
echo Selected: !name!
echo Starting video processing...
echo.

:: Check if input directory exists
if not exist "%input_dir%" (
    echo Error: Input directory "%input_dir%" does not exist.
    echo Please create the input directory and place your videos inside.
    echo.
    pause
    goto menu
)

:: Check if input directory has video files
set "has_videos=0"
for %%i in ("%input_dir%\*.mp4" "%input_dir%\*.avi" "%input_dir%\*.mov") do (
    if exist "%%i" set "has_videos=1"
)

if "!has_videos!"=="0" (
    echo Error: No video files found in input directory.
    echo Please place your videos in the "%input_dir%" directory.
    echo Supported formats: .mp4, .avi, .mov
    echo.
    pause
    goto menu
)

:: Create output directory if it doesn't exist
if not exist "%output_dir%" (
    mkdir "%output_dir%"
    echo Created output directory: %output_dir%
    echo.
)

:: Process all video files
for %%i in ("%input_dir%\*.mp4" "%input_dir%\*.avi" "%input_dir%\*.mov") do (
    if exist "%%i" (
        echo Processing: %%~nxi
        
        :: Get video FPS
        for /f "tokens=*" %%f in ('"%FFMPEG%" -i "%%i" 2^>^&1 ^| findstr "fps"') do (
            set "fps_info=%%f"
        )
        :: Extract FPS value
        for /f "tokens=2 delims=," %%f in ("!fps_info!") do (
            set "fps_value=%%f"
        )
        set "fps_value=!fps_value: fps=!"
        set /a "fps_num=!fps_value!"
        
        echo Original FPS: !fps_num!
        
        :: Get original video bitrate (in kbps)
        for /f "tokens=*" %%b in ('"%FFMPEG%" -i "%%i" 2^>^&1 ^| findstr "bitrate"') do (
            set "bitrate_info=%%b"
        )
        
        :: Extract bitrate value (kbps)
        for /f "tokens=6 delims=:, " %%b in ("!bitrate_info!") do (
            set /a "orig_bitrate=%%b"
        )
        
        :: Get file size (in bytes)
        for %%s in ("%%i") do set "orig_size=%%~zs"
        
        :: Convert bitrate from kbps to Mbps
        set /a "orig_bitrate_m=orig_bitrate/1000"
        echo Original bitrate: !orig_bitrate_m! Mbps
        echo Original size: !orig_size! bytes
        
        :: Build output filename
        set "output_file=%output_dir%\%%~ni!suffix!%%~xi"
        
        :: Choose compression strategy based on encoder and original bitrate
        if "!encoder!"=="h264_nvenc" (
            :: GPU encoding
            if !orig_bitrate_m! leq 2 (
                echo Original bitrate too low, skipping compression
                echo.
                continue
            ) else (
                :: Set target bitrate to half of original, but not exceeding 15Mbps
                set /a "target_bitrate=orig_bitrate_m/2"
                if !target_bitrate! gtr 15 set "target_bitrate=15"
                echo Target bitrate: !target_bitrate! Mbps
                
                :: Check if FPS reduction is needed based on control option and current FPS
                if "!control_fps!"=="1" if !fps_num! geq 50 (
                    echo High FPS detected, reducing to 30FPS
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
            :: CPU encoding
            if !orig_bitrate_m! leq 2 (
                echo Original bitrate too low, skipping compression
                echo.
                continue
            ) else (
                :: Use CRF mode with maxrate
                set /a "target_bitrate=orig_bitrate_m/2"
                if !target_bitrate! gtr 15 set "target_bitrate=15"
                echo Target bitrate: !target_bitrate! Mbps
                
                :: Check if FPS reduction is needed based on control option and current FPS
                if "!control_fps!"=="1" if !fps_num! geq 50 (
                    echo High FPS detected, reducing to 30FPS
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
        
        :: Check compression results
        if exist "!output_file!" (
            for %%s in ("!output_file!") do set "new_size=%%~zs"
            echo New file size: !new_size! bytes
            
            :: If new file is larger, delete it and keep original
            if !new_size! gtr !orig_size! (
                echo Warning: Compressed file is larger, keeping original
                del "!output_file!"
            ) else (
                set /a "saved_space=orig_size-new_size"
                set /a "saved_ratio=(saved_space*100)/orig_size"
                echo Space saved ratio: 0.!saved_ratio!x
            )
        )
            
        echo Completed: %%~nxi
        echo.
    )
)

echo All videos processed!
echo.
pause
goto menu 