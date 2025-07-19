@echo off
REM =============================================================================
REM CoCoSpec Project - Windows Main Script with Auto-Troubleshooting
REM =============================================================================
@echo off
REM =============================================================================
REM GBM Detection Project - Windows Main Script with Auto-Troubleshooting
REM =============================================================================


setlocal enabledelayedexpansion

REM Project configuration
set PROJECT_NAME=GBM_Detection
set ENV_NAME=cocospec_env
set SCRIPTS_DIR=scripts

REM === Python version check: ONLY Python 3.12.3 is supported ===
set PYTHON_EXE=
where python3.12 >nul 2>&1
if !errorlevel! == 0 set PYTHON_EXE=python3.12
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if !errorlevel! == 0 (
        for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
        echo !PY_VER! | findstr /C:"3.12.3" >nul
        if !errorlevel! == 0 set PYTHON_EXE=python
    )
)
if not defined PYTHON_EXE (
    echo Error: Solo se admite Python 3.12.3. Instala esa versión y agrégala al PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('!PYTHON_EXE! -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo Usando Python: !PYTHON_PATH! (solo 3.12.3 soportado)

echo ============================================
echo   GBM Detection Project Setup ^& Run
echo ============================================

REM Main execution starts here
goto :main

REM ============================================================================
REM FUNCTIONS
REM ============================================================================

REM Function to run complete troubleshooting
:troubleshoot
echo Running automatic troubleshooting...
if exist "%SCRIPTS_DIR%\troubleshoot.bat" (
    call "%SCRIPTS_DIR%\troubleshoot.bat" auto
) else (
    call "%SCRIPTS_DIR%\cleanup.bat" >nul 2>&1
)
ping 127.0.0.1 -n 2 >nul
goto :eof

REM Function to fix pip
:fix_pip
echo Attempting to fix pip installation...
if exist "%SCRIPTS_DIR%\fix_pip.bat" (
    call "%SCRIPTS_DIR%\fix_pip.bat"
) else (
    python -m ensurepip --upgrade --default-pip --user >nul 2>&1
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Downloading and installing pip...
        if exist "get-pip.py" del "get-pip.py"
        curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
        if exist "get-pip.py" (
            python get-pip.py --user >nul 2>&1
            del "get-pip.py" >nul 2>&1
        )
    )
)
goto :eof

REM ============================================================================
REM MAIN EXECUTION
REM ============================================================================

:main

REM Optional: Run health check first (uncomment the next 2 lines to enable)
REM echo Running system health check...
REM if exist "%SCRIPTS_DIR%\health_check.bat" call "%SCRIPTS_DIR%\health_check.bat"

REM Check if Python is installed
echo ============================================
echo   Python Environment Check
echo ============================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Error: Python is not installed or not in PATH.
    echo.
    echo SOLUTIONS:
    echo 1. Install Python from: https://python.org/downloads/
    echo 2. Make sure to check "Add Python to PATH" during installation
    echo 3. Restart Command Prompt after installation
    echo.
    pause
    exit /b 1
)

echo Found Python version:
python --version

REM Get detailed Python info for troubleshooting
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo Python executable: %PYTHON_PATH%

REM Check if this is a problematic Python installation (like some conda installations)
echo %PYTHON_PATH% | findstr /i "conda\|anaconda" >nul
if %errorlevel% equ 0 (
    echo.
:extraction_complete
REM Extraction complete, continue to main script
goto :after_extraction
    echo  WARNING: Detected Conda/Anaconda Python
    echo This may cause virtual environment issues.
    echo.
    echo RECOMMENDATION:
    echo 1. Use 'conda create' instead of venv, OR
    echo 2. Install standalone Python from python.org
    echo.
    echo Continuing anyway...
    timeout /t 3 /nobreak >nul
)

REM Check Python architecture
for /f "tokens=*" %%i in ('python -c "import platform; print(platform.architecture()[0])"') do set PYTHON_ARCH=%%i
echo Python architecture: %PYTHON_ARCH%

REM Check and fix pip if needed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: pip not available. Auto-fixing...
    call :fix_pip
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Could not fix pip. Please reinstall Python with pip included.
        pause
        exit /b 1
    )
)

REM Check venv module
python -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: venv module not available. Please reinstall Python.
    pause
    exit /b 1
)

REM ============================================
REM   Virtual Environment Management
REM ============================================

set NEED_NEW_ENV=0
set NEED_INSTALL_PACKAGES=0

echo Checking virtual environment: %ENV_NAME%

REM Step 1: Check if environment exists and is functional
if exist "%ENV_NAME%\Scripts\python.exe" (
    echo Environment directory found
    
    REM Test if environment can be activated
    call "%ENV_NAME%\Scripts\activate.bat" >nul 2>&1
    if %errorlevel% neq 0 (
        echo  Environment cannot be activated - will recreate
        set NEED_NEW_ENV=1
    ) else (
        echo Environment can be activated
        
        REM Step 2: Check if all required packages are installed (simple check)
        echo Checking required packages...
        "%ENV_NAME%\Scripts\python.exe" -c "import torch, torchvision, numpy, sklearn, matplotlib, pandas, pydicom, cv2, tqdm; print('All packages available')" >nul 2>&1
        if %errorlevel% neq 0 (
            echo  Some required packages are missing
            set NEED_INSTALL_PACKAGES=1
        ) else (
            echo All required packages are installed
            set NEED_INSTALL_PACKAGES=0
        )
        
        call deactivate >nul 2>&1
    )
) else (
    echo  Environment not found - will create new one
    set NEED_NEW_ENV=1
    set NEED_INSTALL_PACKAGES=1
)

REM Step 3: Create environment if needed
if %NEED_NEW_ENV%==1 (
    echo.
    echo Creating new virtual environment...
    
    REM Remove existing broken environment
    if exist "%ENV_NAME%" (
        echo Removing broken environment...
        rmdir /s /q "%ENV_NAME%" >nul 2>&1
    )
    
    REM Create new environment
    python -m venv "%ENV_NAME%"
    if %errorlevel% neq 0 (
        echo  Failed to create environment with venv, trying alternative...
        python -m venv "%ENV_NAME%" --without-pip
        if %errorlevel% neq 0 (
            echo  Environment creation failed completely
            echo.
            echo SOLUTION: Use scripts\diagnose.bat or scripts\troubleshoot.bat for automatic problem solving
            pause
            exit /b 1
        )
    )

    if exist "%ENV_NAME%\Scripts\python.exe" (
        echo Virtual environment created, checking pip...
        "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo  pip not found in new environment, attempting to install with get-pip.py...
            if exist "get-pip.py" del "get-pip.py"
            curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
            if exist "get-pip.py" (
                "%ENV_NAME%\Scripts\python.exe" get-pip.py >nul 2>&1
                del "get-pip.py" >nul 2>&1
            )
        )
        "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo  Error: Could not install pip in the new environment. Please check your Python installation.
            echo  Try reinstalling Python from python.org and ensure pip is included.
            pause
            exit /b 1
        )
        echo Virtual environment created and pip is available
        set NEED_INSTALL_PACKAGES=1
    ) else (
        echo  Environment creation verification failed
        echo.
        echo SOLUTION: Use scripts\diagnose.bat or scripts\troubleshoot.bat for automatic problem solving
        pause
        exit /b 1
    )
)

REM Step 4: Activate environment
echo.
echo Activating virtual environment...
call "%ENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo  Failed to activate environment
    echo.
    echo SOLUTION: Use scripts\diagnose.bat or scripts\troubleshoot.bat for problem solving
    pause
    exit /b 1
)
echo Environment activated successfully

REM Step 5: Install packages if needed
if %NEED_INSTALL_PACKAGES%==1 (
    echo.
    echo Installing required packages...
    
    REM Upgrade pip first
    python -m pip install --upgrade pip --quiet

    REM Install requirements
    if exist "requirements.txt" (
        echo Installing packages from requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo  requirements.txt not found, installing essential packages...
        pip install numpy pandas scikit-learn matplotlib tqdm pydicom opencv-python
    )
) else (
    echo Packages already installed, skipping installation
)


REM Create necessary directories
echo Creating necessary directories...
if not exist "models" mkdir models
if not exist "figures" mkdir figures
if not exist "data" mkdir data

REM Download dataset
echo ============================================
echo   Downloading CoCoaSpec Dataset
echo ============================================

if exist "data\data.rar" (
    echo Dataset archive already exists. Skipping download...
) else (
    echo Starting download from HuggingFace...
    curl -L -o "data\data.rar" "https://huggingface.co/datasets/ecos-nord-ginp-uis/CoCoaSpec/resolve/main/data.rar"
    if %errorlevel% neq 0 (
        echo Error: Failed to download dataset
        pause
        exit /b 1
    )
    echo Dataset download completed successfully!
)

REM Extract dataset


REM === CoCoSpec Extraction: simple and robust ===
set KEY_FILE=data\resources\wavelengths\specim_iq.csv
if exist "%KEY_FILE%" (
    echo Dataset already extracted and verified. Skipping extraction...
    goto :after_extraction
)

echo Extracting dataset...
set EXTRACTION_SUCCESS=0

REM Method 1: Try 7-Zip first (most common on Windows)
echo Trying 7-Zip extraction...
set SEVENZIP_FOUND=0
if exist "C:\Program Files\7-Zip\7z.exe" (
    set SEVENZIP_PATH="C:\Program Files\7-Zip\7z.exe"
    set SEVENZIP_FOUND=1
) else if exist "C:\Program Files (x86)\7-Zip\7z.exe" (
    set SEVENZIP_PATH="C:\Program Files (x86)\7-Zip\7z.exe"
    set SEVENZIP_FOUND=1
) else (
    where 7z >nul 2>&1
    if %errorlevel% equ 0 (
        set SEVENZIP_PATH=7z
        set SEVENZIP_FOUND=1
    )
)

if %SEVENZIP_FOUND%==1 (
    echo Found 7-Zip, attempting extraction...
    %SEVENZIP_PATH% x "data\data.rar" -o"data" -y >nul 2>&1
    if exist "%KEY_FILE%" (
        echo Dataset extracted successfully with 7-Zip!
        set EXTRACTION_SUCCESS=1
        goto :extraction_complete
    ) else (
        echo 7-Zip extraction failed, trying next method...
    )
) else (
    echo 7-Zip not found, trying WinRAR...
)

REM Method 2: Try WinRAR
echo Trying WinRAR extraction...
set WINRAR_FOUND=0
if exist "C:\Program Files\WinRAR\WinRAR.exe" (
    set WINRAR_PATH="C:\Program Files\WinRAR\WinRAR.exe"
    set WINRAR_FOUND=1
) else if exist "C:\Program Files (x86)\WinRAR\WinRAR.exe" (
    set WINRAR_PATH="C:\Program Files (x86)\WinRAR\WinRAR.exe"
    set WINRAR_FOUND=1
) else if exist "C:\Program Files\WinRAR\unrar.exe" (
    set WINRAR_PATH="C:\Program Files\WinRAR\unrar.exe"
    set WINRAR_FOUND=1
) else if exist "C:\Program Files (x86)\WinRAR\unrar.exe" (
    set WINRAR_PATH="C:\Program Files (x86)\WinRAR\unrar.exe"
    set WINRAR_FOUND=1
) else (
    where winrar >nul 2>&1
    if %errorlevel% equ 0 (
        set WINRAR_PATH=winrar
        set WINRAR_FOUND=1
    ) else (
        where unrar >nul 2>&1
        if %errorlevel% equ 0 (
            set WINRAR_PATH=unrar
            set WINRAR_FOUND=1
        )
    )
)

if %WINRAR_FOUND%==1 (
    echo Found WinRAR/unrar, attempting extraction...
    %WINRAR_PATH% x "data\data.rar" "data" >nul 2>&1
    if exist "%KEY_FILE%" (
        echo Dataset extracted successfully with WinRAR!
        set EXTRACTION_SUCCESS=1
        goto :extraction_complete
    ) else (
        echo WinRAR extraction failed, trying Python packages...
    )
) else (
    echo WinRAR not found, trying Python packages...
)

REM Method 3: Try Python-based extraction
echo Trying Python-based extraction packages...
python -c "import rarfile" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing rarfile package...
    pip install rarfile --quiet >nul 2>&1
)

python -c "import rarfile; rf = rarfile.RarFile('data\\data.rar'); rf.extractall('data'); rf.close(); print('Python rarfile extraction completed')" 2>nul
if %errorlevel% equ 0 (
    if exist "%KEY_FILE%" (
        echo Dataset extracted successfully with Python rarfile!
        set EXTRACTION_SUCCESS=1
        goto :extraction_complete
    ) else (
        echo rarfile extraction created directory but no files found. Trying patoolib...
    )
) else (
    echo Python rarfile extraction failed (missing unrar tool), trying patoolib...
)

python -c "import patoolib" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing patoolib package...
    pip install patoolib --quiet >nul 2>&1
)

python -c "import patoolib; patoolib.extract_archive('data\\data.rar', outdir='data'); print('Python patoolib extraction completed')" 2>nul
if %errorlevel% equ 0 (
    if exist "%KEY_FILE%" (
        echo Dataset extracted successfully with Python patoolib!
        set EXTRACTION_SUCCESS=1
        goto :extraction_complete
    ) else (
        echo patoolib extraction created directory but no files found. Trying PowerShell...
    )
) else (
    echo Python patoolib extraction failed, trying PowerShell...
)

REM Method 4: Try PowerShell as last resort
echo Trying PowerShell method (rename to .zip)...
copy "data\data.rar" "data\dataset_temp.zip" >nul 2>&1
if exist "data\dataset_temp.zip" (
    powershell -Command "try { Expand-Archive -Path 'data\dataset_temp.zip' -DestinationPath 'data' -Force; exit 0 } catch { exit 1 }" >nul 2>&1
    if exist "%KEY_FILE%" (
        echo Dataset extracted successfully with PowerShell!
        set EXTRACTION_SUCCESS=1
        del "data\dataset_temp.zip" >nul 2>&1
        goto :extraction_complete
    ) else (
        echo PowerShell extraction failed...
    )
    del "data\dataset_temp.zip" >nul 2>&1
)

:after_extraction
echo.
echo Running main script...
cd /d %~dp0
"%ENV_NAME%\Scripts\python.exe" -m src.Main

endlocal
