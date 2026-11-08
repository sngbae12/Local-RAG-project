@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Local PDF RAG Chatbot
echo.

if not exist ".venv\Scripts\python.exe" goto missing_environment
if not exist "app.py" goto missing_app

".venv\Scripts\python.exe" -c "import sys, struct; sys.exit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 else 1)"
if not "%ERRORLEVEL%"=="0" goto invalid_python

".venv\Scripts\python.exe" download_models.py --check
if not "%ERRORLEVEL%"=="0" goto missing_models

".venv\Scripts\python.exe" -c "import flask, langchain, chromadb, llama_cpp"
if not "%ERRORLEVEL%"=="0" goto missing_packages

echo.
echo Open http://127.0.0.1:5050/ after the server starts.
echo Wait for the model status to become ready before uploading a PDF.
echo Keep this window open. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" app.py
set "APP_EXIT=%ERRORLEVEL%"
echo.
echo Server stopped. Exit code: %APP_EXIT%
pause
exit /b %APP_EXIT%

:missing_environment
echo [ERROR] Project Python environment not found. Run setup.bat first.
goto failed

:missing_app
echo [ERROR] app.py not found. Extract the entire ZIP before running.
goto failed

:invalid_python
echo [ERROR] This environment requires Python 3.11, 64-bit. See README.md.
goto failed

:missing_models
echo [ERROR] Model files are missing or invalid. See models\README.md.
echo Run: .venv\Scripts\python.exe download_models.py
goto failed

:missing_packages
echo [ERROR] A Python package could not load. See docs\troubleshooting.md.
goto failed

:failed
pause
exit /b 1
