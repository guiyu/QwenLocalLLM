@echo off
setlocal enabledelayedexpansion

echo Starting project cleanup...

REM Create backup with timestamp
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%
echo Creating backup...
xcopy /E /I /Y ".\" "..\qwen_backup_%timestamp%"

REM Clean model and cache files
echo Cleaning model and cache files...
if exist "models" rd /S /Q "models"
if exist "model_cache" rd /S /Q "model_cache"
if exist ".cache\huggingface" rd /S /Q ".cache\huggingface"

REM Clean compiled and temporary files
echo Cleaning compiled and temporary files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /S /Q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /S /Q "%%d"
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /S /Q "%%d"
del /S /Q *.pyc
del /S /Q *.pyo
del /S /Q *.pyd

REM Clean log files
echo Cleaning log files...
if exist "logs" del /S /Q "logs\*.*"

REM Clean Android build files
echo Cleaning Android build files...
if exist "android\app\build" rd /S /Q "android\app\build"
if exist "android\.gradle" rd /S /Q "android\.gradle"
if exist "android\local.properties" del "android\local.properties"

REM Create necessary directories
echo Creating necessary directories...
mkdir models
mkdir model_cache
mkdir logs
mkdir models\original
mkdir models\pruned
mkdir models\quantized
mkdir models\android

REM Clean Python virtual environment (optional)
set /p cleanenv="Do you want to clean the Python virtual environment? (Y/N) "
if /i "%cleanenv%"=="Y" (
    echo Cleaning Python virtual environment...
    if exist "venv" rd /S /Q "venv"
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
)

echo Cleanup completed!
pause