# 설치·실행 오류 해결

모든 명령은 `app.py`가 있는 프로젝트 폴더의 **명령 프롬프트(CMD)**에서 실행합니다. 모델 파일을 바꾸거나 라이브러리를 교체할 때는 서버 창에서 `Ctrl + C`로 먼저 종료합니다.

## 웹 화면은 열리지만 질문할 수 없음

`Running on http://127.0.0.1:5050`은 웹서버가 시작됐다는 뜻입니다. 임베딩 모델과 답변 모델까지 준비됐다는 뜻은 아닙니다.

서버 실행 중 [모델 상태](http://127.0.0.1:5050/api/status)를 열어 아래 항목을 봅니다.

| 항목 | 의미 |
| --- | --- |
| `ready` | 전체 준비 상태 |
| `embeddings_ready` | BGE-M3 임베딩 준비 상태 |
| `llm_ready` | EXAONE 답변 모델 준비 상태 |
| `embedding_error` | 임베딩 모델 오류 |
| `llm_error` | 답변 모델 오류 |

`ready: true`가 된 뒤 PDF를 업로드합니다. `GET /api/status ... 200` 로그는 상태 조회 성공이며, 모델 로딩 성공을 뜻하지 않습니다.

## Python 또는 패키지가 없음

Python 3.11 64비트를 설치한 뒤 `setup.bat`을 실행합니다. 기존 `.venv`가 다른 Python 버전이면 서버를 종료하고 해당 폴더 이름을 바꿔 보관한 후 새 환경을 만들 수 있습니다. `uploads/`, `chroma_db/`, `models/`를 지울 필요는 없습니다.

```bat
.venv\Scripts\python.exe -m pip check
```

설치 중 `No matching distribution found`가 나오면 누락된 패키지 이름과 전체 메시지를 남겨 주세요. `setup.bat`은 `llama-cpp-python`을 CPU용 사전 빌드 파일로 설치하도록 요청하며, 해당 파일을 찾지 못하면 소스 빌드로 넘어가지 않고 중단합니다.

## 배치 파일에서 잘린 명령이 나타남

예: `'thon"'은(는) 내부 또는 외부 명령 ...`

이 메시지만으로 원인을 확정할 수는 없습니다. 현재 배치 파일은 ASCII 명령과 Windows용 CRLF 줄바꿈을 사용합니다. 최신 ZIP을 완전히 압축 해제해 사용하세요. `.gitattributes`는 저장된 배치 파일의 줄바꿈을 그대로 보존합니다.

배치 파일을 거치지 않고 기존 가상환경으로 실행할 수도 있습니다.

```bat
.venv\Scripts\python.exe app.py
```

## BGE-M3 설정 또는 model_type 오류

예: `No modules.json found`, `Unrecognized model`, `Should have a model_type key in its config.json`.

`models/bge-m3/`에 공식 모델 구성 파일이 없거나 잘못된 경우를 먼저 살펴봅니다. 공식 `config.json`의 `model_type`은 `xlm-roberta`입니다. 임의로 키만 추가하지 말고 공식 파일로 복구합니다.

```bat
.venv\Scripts\python.exe download_models.py --repair-config
.venv\Scripts\python.exe download_models.py --check
.venv\Scripts\python.exe app.py
```

`--repair-config`는 JSON 파일을 다시 받으며 가중치에는 강제 재다운로드를 요청하지 않습니다. 누락된 나머지 필수 파일도 다운로드를 시도합니다. 기본 검사를 통과해도 실제 모델 로딩을 별도로 확인해야 합니다.

가중치 손상이 의심되어 전체 파일을 다시 받으려면 아래 명령을 사용합니다. 모델 가중치 약 4.2GB를 포함해 다시 받으므로 시간과 다운로드 공간이 필요합니다.

```bat
.venv\Scripts\python.exe download_models.py --force
```

## EXAONE 로딩 중 0xc000001d 오류

예: `WinError -1073741795`, `Windows Error 0xc000001d`, `Could not load Llama model`.

이 Windows 코드는 잘못된 명령 실행(`STATUS_ILLEGAL_INSTRUCTION`)을 뜻합니다. 설치된 네이티브 실행 파일과 CPU 명령어 지원의 불일치 등이 원인일 수 있으나, 이 코드만으로 원인을 확정할 수는 없습니다.

다음 명령으로 CPU와 설치 버전을 확인합니다.

```bat
powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
.venv\Scripts\python.exe -c "import sys, platform; from importlib.metadata import version; print(sys.version); print(platform.machine()); print(version('llama-cpp-python'))"
```

웹앱과 분리한 모델 로딩 시험입니다. 성공하면 마지막에 `MODEL_LOAD_OK`가 표시됩니다. 이 시험은 로딩 단계만 검사하며, 실제 질의응답 성공까지 의미하지 않습니다.

```bat
.venv\Scripts\python.exe -c "from llama_cpp import Llama; m=Llama(model_path='models/exaone_2.4b/EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf', n_ctx=512, n_batch=64, n_threads=4, n_gpu_layers=0, verbose=True); print('MODEL_LOAD_OK')"
```

다른 공식 CPU 배포본과 비교할 필요가 있을 때는 아래처럼 `0.3.19`로 바꾸어 시험할 수 있습니다. **호환성 비교용이며 해결이 확인된 버전은 아닙니다.** 기본 `requirements.txt`의 `0.3.33`은 그대로 두었습니다. 기존 `setup.bat`을 다시 실행하면 지정된 기본 버전으로 설치됩니다.

```bat
.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.19 --index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary=:all: --force-reinstall --no-cache-dir --no-deps
.venv\Scripts\python.exe -m pip check
```

설치 후 위의 모델 로딩 시험과 실제 PDF 질의응답을 다시 수행합니다. 기본 버전으로 되돌리는 명령:

```bat
.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.33 --index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary=:all: --force-reinstall --no-cache-dir --no-deps
```

같은 오류가 계속되면 CPU 이름, 버전, 모델 로딩 시험의 마지막 로그를 남겨 주세요. 모델 파일이 존재하고 `GGUF` 헤더가 맞는 것만으로 전체 파일 무결성이나 CPU 호환성을 증명할 수는 없습니다.

## CUDA 또는 llama.dll 오류

현재 앱은 CPU 추론을 사용합니다. CUDA 라이브러리를 찾지 못한다는 메시지가 나오면, 설치된 배포본부터 확인하고 위의 기본 CPU 배포본 재설치 명령을 사용합니다. 같은 오류 코드처럼 보이더라도 `0xc000001d`와 DLL 누락은 구분해야 합니다.

## PDF를 처리하지 못함

- PDF에서 글자를 마우스로 선택할 수 있는지 확인합니다. 이미지 스캔 PDF의 OCR은 미구현입니다.
- 먼저 짧은 PDF 하나로 시험합니다. 서버의 업로드 요청 제한은 50MiB이며, 여러 파일을 한 번에 올리면 요청 전체 크기에 적용됩니다.
- 업로드 완료 후 질문합니다. `임베딩 중`이면 아직 검색 데이터를 만드는 단계입니다.
- 출처가 없다는 답변은 검색 임계값 때문에 나올 수도 있습니다. 문서에 답이 있어도 검색이 놓칠 수 있습니다.

## 명령창 경고

`LangChainDeprecationWarning`은 향후 라이브러리 API 교체 안내입니다. 경고와 실제 오류를 구분하고 `/api/status`의 준비 상태를 봅니다. Flask 개발 서버 경고는 로컬 개인 실행 중에도 표시됩니다.

## 참고 자료

- [Microsoft: NTSTATUS 값](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55)
- [llama-cpp-python 공식 설치 안내](https://github.com/abetlen/llama-cpp-python#installation)
- [공식 CPU 배포 파일 목록](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/)
- [BGE-M3 공식 설정](https://huggingface.co/BAAI/bge-m3/blob/main/config.json)
- [Hugging Face 다운로드 안내](https://huggingface.co/docs/huggingface_hub/guides/download)
