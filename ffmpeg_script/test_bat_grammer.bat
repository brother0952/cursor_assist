@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: check if NVIDIA GPU is available
set "has_nvidia=0"

if "!has_nvidia!"=="1" (
    echo "GPU mode (NVIDIA)" ---
) else (
    echo "CPU mode" ---
)

set "has_nvidia=1"

if "!has_nvidia!"=="1" (
    echo "GPU mode (NVIDIA)" ---
) else (
    echo "CPU mode" ---
)
