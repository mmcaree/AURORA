@echo off
REM === AURORA ROCm/AMD 7800XT Build Script ===
echo [1/5] Setting up environment...

REM Adjust path if your ROCm is installed elsewhere
set "PATH=C:\Program Files\AMD\ROCm\6.2\bin;%PATH%"

echo [2/5] Creating build directory...
rmdir /S /Q build
mkdir build
cd build

echo [3/5] Running CMake configuration for ROCm HIPBLAS...

cmake .. -G Ninja -DLLAMA_CUBLAS=OFF -DLLAMA_HIPBLAS=ON -DLLAMA_METAL=OFF -DLLAMA_VULKAN=OFF -DLLAMA_BUILD=chat -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF

if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    pause
    exit /b 1
)

echo [4/5] Building llama.dll...
cmake --build .

if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)


echo [✅ DONE] llama.cpp built and installed with ROCm HIPBLAS!
pause
