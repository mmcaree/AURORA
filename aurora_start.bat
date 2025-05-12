@echo off
title Starting Aurora AI Assistant
cd /d E:\AURORA

:: Activate the environment and run controller in new window
start "Aurora Controller" powershell -NoExit -Command ".\aurora-env\Scripts\Activate.ps1; python aurora_controller.py"

:: Run the viewer in another window
::start "Aurora Memory Viewer" powershell -NoExit -Command ".\aurora-env\Scripts\Activate.ps1; python aurora_memory_viewer.py"

:: Run the wesocket start in another window
start "Aurora Websockets" powershell -NoExit -Command ".\aurora-env\Scripts\Activate.ps1; python websocket_server.py"

:: Start KoboldCpp ROCm with preset
::start "KoboldCpp" powershell -Command "Start-Process -FilePath '.\\koboldcpp\\koboldcpp_cu12.exe' -ArgumentList 'E:\\AURORA\\koboldcpp\\aurora_load_12B.kcpps'"

exit
