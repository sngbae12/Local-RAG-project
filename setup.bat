@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Local PDF RAG Chatbot - first-time setup
echo Internet access is required. Model weights total about 4.2 GB.
echo Read models\README.md for model sources and terms before downloading.
echo.

if not exist "requirements.txt" goto missing_project
if exist ".venv\Scripts\python.exe" goto check_python

py -3.11 -m venv .venv 2>nul
if "%ERRORLEVEL%"=="0" goto check_python
python -c "import sys, struct; sys.exit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)" 2>nul
if not "%ERRORLEVEL%"=="0" goto missing_python
python -m venv .venv
if not "%ERRORLEVEL%"=="0" goto failed

:check_python
".venv\Scripts\python.exe" -c "import sys, struct; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)"
if not "%ERRORLEVEL%"=="0" goto invalid_environment

echo.
echo [1/3] Installing packages and a pre-built CPU runtime...
".venv\Scripts\python.exe" -m pip install --only-binary=llama-cpp-python -r requirements.txt
if not "%ERRORLEVEL%"=="0" goto failed

echo.
echo [2/3] Checking package dependencies...
".venv\Scripts\python.exe" -m pip check
if not "%ERRORLEVEL%"=="0" goto failed

echo.
echo [3/3] Downloading and checking model files...
".venv\Scripts\python.exe" download_models.py
if not "%ERRORLEVEL%"=="0" goto failed

echo.
echo Setup complete. Run start.bat and open http://127.0.0.1:5050/
echo Model loading and CPU compatibility are checked when the app runs.
pause
exit /b 0

:missing_project
echo [ERROR] Project files not found. Extract the entire ZIP first.
goto failed

:missing_python
echo [ERROR] Install Python 3.11, 64-bit, then run setup.bat again.
echo Download: https://www.python.org/downloads/release/python-3119/
goto failed

:invalid_environment
echo [ERROR] Existing .venv must use Python 3.11, 64-bit. See README.md.
goto failed

:failed
echo.
echo Setup did not complete. Keep the error above for troubleshooting.
echo See docs\troubleshooting.md. You can retry setup.bat after fixing it.
pause
exit /b 1
