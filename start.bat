@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Local PDF RAG Chatbot
echo ============================================
echo 작업 폴더: %CD%
echo.

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
  echo 가상환경: .venv
) else (
  set "PY=python"
  echo 가상환경: 없음 (시스템 Python 사용)
)

"%PY%" -c "import sys; print('Python:', sys.version)" 2>nul
if errorlevel 1 (
  echo.
  echo [오류] Python을 실행할 수 없습니다.
  echo Python 3.11 이상을 설치하거나, 프로젝트 폴더에 .venv를 만든 뒤 다시 실행하세요.
  echo.
  pause
  exit /b 1
)

if not exist "app.py" (
  echo.
  echo [오류] app.py가 없습니다. start.bat을 프로젝트 루트에서 실행하세요.
  echo.
  pause
  exit /b 1
)

if not exist "models\exaone_2.4b\EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf" (
  echo.
  echo [오류] LLM 파일이 없습니다.
  echo 필요 경로: models\exaone_2.4b\EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf
  echo 모델 준비 방법은 models\README.md 와 README.md 를 보세요.
  echo.
  pause
  exit /b 1
)

if not exist "models\bge-m3\config.json" (
  echo.
  echo [오류] 임베딩 모델 폴더가 없습니다.
  echo 필요 경로: models\bge-m3\  (config.json 포함)
  echo 모델 준비 방법은 models\README.md 와 README.md 를 보세요.
  echo.
  pause
  exit /b 1
)

"%PY%" -c "import flask, langchain, chromadb, llama_cpp" 2>nul
if errorlevel 1 (
  echo.
  echo [오류] Python 패키지가 부족합니다.
  echo 아래를 실행한 뒤 다시 시도하세요:
  echo   "%PY%" -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo.
echo 서버를 시작합니다. 브라우저에서 http://127.0.0.1:5050/ 으로 접속하세요.
echo 종료하려면 이 창에서 Ctrl+C 를 누르세요.
echo.

"%PY%" app.py
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
  echo 서버가 오류 코드 %ERR% 로 종료되었습니다.
) else (
  echo 서버가 종료되었습니다.
)
echo.
pause
exit /b %ERR%
