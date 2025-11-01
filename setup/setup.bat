@echo off
REM === Move to script directory ===
cd /d "%~dp0.."

REM === Step 1: Create or activate virtual environment ===
if not exist ".venv" (
    echo Creating virtual environment with uv...
    uv venv
)

call .venv\Scripts\activate

REM === Step 2: Sync project dependencies ===
echo Syncing environment...
uv sync

REM === Step 2: Ensure SAM2 exists ===
if not exist "sam2" (
    echo SAM2 not found. Cloning from GitHub...
    git clone https://github.com/facebookresearch/sam2.git
    if errorlevel 1 (
        echo [Error] Git clone failed. Please check your internet connection or git installation.
        pause
        exit /b 1
    )
)

REM === Step 3: Install SAM2 in editable mode ===
if exist "sam2" (
    echo Installing SAM2 in editable mode...
    cd sam2
    uv pip install -e .
    cd ..
) else (
    echo [Warning] sam2 directory not found. Skipping SAM2 installation.
)

REM === Step 5: Ensure SAM2 checkpoints exist ===
if exist "sam2\checkpoints" (
    echo Checking SAM2 checkpoints...
    call :check_download "sam2\checkpoints" || echo [Warning] Failed to download checkpoints.
) else (
    echo [Info] sam2\checkpoints directory not found. Skipping checkpoint download.
)


REM === Step 4: Run the Leaf Shape Tool ===
echo Launching Leaf Shape Tool...
leaf-shape-tool

pause
exit /b 0


:check_download
set "CKPT_DIR=%~1"
cd /d "%CKPT_DIR%"

REM URLs
set "BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824"
set "BPLUS=%BASE_URL%/sam2.1_hiera_base_plus.pt"
set "TINY=%BASE_URL%/sam2.1_hiera_tiny.pt"
set "SMALL=%BASE_URL%/sam2.1_hiera_small.pt"
set "LARGE=%BASE_URL%/sam2.1_hiera_large.pt"

REM Detect downloader
where curl >nul 2>&1
if %errorlevel%==0 (
    set "DOWNLOADER=curl"
) else (
    where powershell >nul 2>&1
    if %errorlevel%==0 (
        set "DOWNLOADER=powershell"
    ) else (
        echo [Error] Neither curl nor PowerShell found.
        exit /b 1
    )
)

REM Download each checkpoint if missing
call :download_if_missing "sam2.1_hiera_tiny.pt" "%TINY%"
call :download_if_missing "sam2.1_hiera_small.pt" "%SMALL%"
call :download_if_missing "sam2.1_hiera_base_plus.pt" "%BPLUS%"
call :download_if_missing "sam2.1_hiera_large.pt" "%LARGE%"

cd /d "%~dp0.."
exit /b 0


:download_if_missing
set "FILE=%~1"
set "URL=%~2"

if exist "%FILE%" (
    echo [Skip] %FILE% already exists.
) else (
    echo [Download] %FILE%
    if "%DOWNLOADER%"=="curl" (
        curl -L -O "%URL%"
    ) else (
        powershell -Command "Invoke-WebRequest '%URL%' -OutFile '%FILE%'"
    )
    if %errorlevel% neq 0 (
        echo [Error] Failed to download %FILE%.
        exit /b 1
    )
)
exit /b 0
