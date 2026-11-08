# Local PDF RAG Chatbot

### 내 컴퓨터에서 PDF를 읽고, 근거 페이지와 함께 답하는 챗봇

PDF를 업로드하면 관련 내용을 검색하고 로컬 AI 모델이 답변합니다. 문서 검색과 답변 생성은 내 컴퓨터에서 처리하며, 외부 채팅 API 키는 필요하지 않습니다.

**[프로그램 ZIP 다운로드](https://github.com/sngbae12/Local-RAG-project/archive/refs/heads/main.zip)** · **[설치하기](#windows에서-시작하기)** · **[오류 해결](docs/troubleshooting.md)** · **[모델 안내](models/README.md)**

> **처음 사용하는 분:** Python 3.11 64비트 설치 → ZIP 압축 해제 → `setup.bat` 실행 → `start.bat` 실행 → 브라우저 접속 순서로 사용합니다. 처음 설치할 때는 인터넷이 필요합니다.

## 할 수 있는 일

| 기능 | 사용 방법 |
| --- | --- |
| PDF 기반 질의응답 | 보고서·학습자료 등 텍스트가 있는 PDF를 올리고 내용에 대해 질문 |
| 근거 확인 | 답변 아래에 표시되는 파일명·페이지·문서 발췌를 원문과 비교 |
| 여러 문서 검색 | 왼쪽 사이드바에서 PDF 여러 개를 업로드 |
| 문서 밖 질문 처리 | 관련 내용을 찾지 못하면 “정보가 없어서 답변할 수 없습니다” 표시 |
| 이어서 질문 | 실행 중인 서버의 최근 대화 맥락을 함께 사용 |

답변의 정확성을 보장하지는 않습니다. 출처는 검색된 문서 조각이므로 중요한 내용은 해당 PDF 페이지와 대조하세요. 글자를 선택할 수 없는 이미지 스캔 PDF는 현재 지원하지 않습니다.

## 실행 환경

| 항목 | 안내 |
| --- | --- |
| 운영체제 | Windows 설치·실행 안내 제공. 다른 운영체제의 실행은 미검증 |
| Python | **3.11, 64비트**. 기존 실행 환경은 3.11.9 |
| 모델 실행 | CPU 사용. 현재 코드의 GPU 가속은 미구현 |
| 다운로드 | EXAONE 약 1.91GB + BGE-M3 가중치 약 2.27GB + 설정·토크나이저 파일 |
| 저장공간 | 위 모델 외에 Python 패키지·다운로드 캐시·업로드 문서 공간도 필요 |
| 메모리·속도 | 최소 RAM과 PC별 응답 시간은 측정하지 않음. CPU·메모리·문서 길이에 따라 차이 발생 |

모든 PC에서 실행되는 설치 패키지로 검증된 상태는 아닙니다. 특히 모델 실행 라이브러리와 CPU의 호환성은 실제 모델 로딩으로 확인해야 합니다.

## Windows에서 시작하기

### 1. Python 설치

[Python 3.11.9 공식 다운로드 페이지](https://www.python.org/downloads/release/python-3119/)의 **Windows installer (64-bit)**를 설치합니다. 설치 화면에서 **Add python.exe to PATH**를 선택합니다. 이미 Python 3.11 64비트가 있으면 다음 단계로 넘어갑니다.

### 2. 프로그램 다운로드와 압축 해제

상단의 **프로그램 ZIP 다운로드**를 누르거나, GitHub의 **Code → Download ZIP**을 선택합니다. ZIP을 완전히 압축 해제한 뒤 **`app.py`, `setup.bat`, `start.bat`이 함께 있는 폴더**를 엽니다.

Git 설치나 GitHub 계정 없이 ZIP으로 내려받을 수 있습니다. Git을 사용하는 경우:

```bat
git clone https://github.com/sngbae12/Local-RAG-project.git
cd Local-RAG-project
```

### 3. 최초 설치: setup.bat

`setup.bat`을 더블클릭합니다. 아래 순서로 준비합니다.

1. 프로젝트 전용 Python 환경 `.venv` 생성. 기존 환경이 있으면 재사용
2. `requirements.txt`에 지정된 패키지와 CPU용 모델 실행 라이브러리 설치
3. 패키지 의존성 검사
4. 공식 저장소에서 필요한 EXAONE·BGE-M3 파일만 다운로드
5. 모델 파일의 존재 여부와 기본 구조 검사

**실행 전 [모델 이용 조건](models/README.md#출처와-이용-조건)을 읽어 주세요.** 모델 가중치는 이 GitHub 저장소나 ZIP에 포함되지 않으며, 설치 중 Hugging Face에서 직접 받습니다.

다운로드에는 시간이 걸립니다. 창에 **`Setup complete`**가 나오는지 확인하세요. 오류가 나오면 [오류 해결](docs/troubleshooting.md)을 확인합니다. 같은 설치를 다시 실행할 수 있으며, 모델 다운로드는 Hugging Face의 로컬 메타데이터를 사용합니다.

### 4. 실행: start.bat

`start.bat`을 더블클릭하고 다음 메시지가 나오면 브라우저에서 주소를 엽니다.

```text
Running on http://127.0.0.1:5050
```

**접속 주소: [http://127.0.0.1:5050/](http://127.0.0.1:5050/)**

검은 명령창은 사용 중 계속 켜 둡니다. 화면의 모델 상태가 **준비됨**으로 바뀐 뒤 PDF를 올립니다. 웹서버 시작 메시지만으로 모델 로딩 성공을 판단하지 않습니다.

### 5. PDF 업로드와 첫 질문

1. 왼쪽에서 텍스트가 있는 짧은 PDF를 선택합니다.
2. 텍스트 추출·청킹·임베딩·저장이 끝날 때까지 기다립니다.
3. 문서에 답이 명확하게 적힌 질문을 입력합니다. 예: “이 보고서에서 제시한 실험 조건을 정리해 줘.”
4. 답변 아래의 참고 페이지를 원문과 비교합니다.

`Enter`는 전송, `Shift + Enter`는 줄바꿈입니다. 종료하려면 명령창에서 `Ctrl + C`를 누릅니다. 이후에는 `start.bat`으로 다시 실행합니다.

## 명령어로 직접 설치하기

프로젝트 폴더의 **명령 프롬프트(CMD)**에서 아래를 순서대로 실행합니다. 모델 조건을 읽은 뒤 다운로드 명령을 실행하세요.

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --only-binary=llama-cpp-python -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe app.py
```

모델이 이미 있다면 기본 파일 검사를 따로 실행할 수 있습니다. 이 검사는 실제 추론 시험이나 전체 파일의 체크섬 검증은 아닙니다.

```bat
.venv\Scripts\python.exe download_models.py --check
```

## 문제가 생겼을 때

| 증상 | 먼저 할 일 |
| --- | --- |
| Python이 없다는 메시지 | Python 3.11 64비트 설치 후 `setup.bat` 실행 |
| 배치 파일에서 잘린 명령 | 최신 ZIP 사용 또는 위의 CMD 직접 실행 절차 사용 |
| `No modules.json found` / `model_type` 오류 | [임베딩 설정 복구](docs/troubleshooting.md#bge-m3-설정-또는-model_type-오류) |
| `0xc000001d` / `WinError -1073741795` | [CPU·실행 라이브러리 진단](docs/troubleshooting.md#exaone-로딩-중-0xc000001d-오류) |
| 웹 화면은 열리지만 질문 불가 | [모델 상태](http://127.0.0.1:5050/api/status)의 `error`, `embedding_error`, `llm_error` 확인 |
| 텍스트를 추출할 수 없다는 메시지 | 글자를 선택할 수 있는 PDF로 시험. 이미지 PDF OCR은 미구현 |

해결되지 않으면 [GitHub Issues](https://github.com/sngbae12/Local-RAG-project/issues)에 Python 버전, CPU 이름, 오류 로그를 남겨 주세요. 개인 문서와 모델 가중치를 첨부할 필요는 없습니다.

## 처리 방식과 데이터 저장

PDF에서 텍스트를 추출해 문서 조각으로 나눈 뒤, BGE-M3로 검색용 벡터를 만들고 ChromaDB에 저장합니다. 질문과 관련된 조각을 찾아 EXAONE에 전달하고 답변을 생성합니다.

| 구성 | 사용 기술·설정 |
| --- | --- |
| 웹 화면·서버 | HTML / CSS / JavaScript / Flask |
| PDF 텍스트 추출 | PyPDFLoader / pypdf |
| 문서 분할 | 500자 단위, 인접 조각 100자 중첩 |
| 임베딩·검색 | BAAI/bge-m3 / ChromaDB / cosine distance 기준 |
| 답변 생성 | EXAONE-3.5-2.4B-Instruct Q5_K_M / llama-cpp-python / CPU |
| 검색 결과 | 조건을 통과한 관련 조각 최대 4개 |

| 저장 위치 | 내용 |
| --- | --- |
| `models/` | 내려받은 모델 파일 |
| `uploads/` | 업로드한 PDF |
| `chroma_db/` | 문서 조각과 검색 데이터 |
| 프로세스 메모리 | 대화 기록. 서버 종료 시 사라짐 |

**새 대화**는 대화 맥락을 지우며 PDF와 검색 데이터는 유지합니다. 같은 이름의 PDF를 올리면 교체를 시도합니다. 교체·복구 동작의 상세 범위는 [개발 기록](docs/development-notes.md)에 있습니다.

외부 채팅 API로 문서를 보내는 코드는 사용하지 않습니다. 패키지 설치·모델 다운로드는 인터넷을 사용하며, 의존 라이브러리의 텔레메트리까지 포함한 완전한 통신 차단은 검증하지 않았습니다. 현재 앱은 `127.0.0.1`에서 개인 사용을 전제로 실행됩니다.

## 검증 범위

- 기존 기능과 당시 테스트 기록: [개발 기록과 기존 검증 범위](docs/development-notes.md)
- 새 다운로드 도구: 모의 다운로드와 임시 파일로 누락·잘못된 설정·복구 동작 시험
- Windows 배치 파일: ASCII 명령과 CRLF 줄바꿈으로 작성. 이번 변경의 Windows 전체 설치·실제 모델 추론은 미검증
- `0xc000001d`에 대해 안내한 다른 CPU 배포본은 비교 시험용이며, 특정 PC에서 해결된 버전으로 확정하지 않음

이미지 PDF OCR, GPU 가속, 대화 기록 파일 저장은 현재 구현되어 있지 않습니다. 정확도·처리량·최소 하드웨어 요구사항의 정식 벤치마크도 수행하지 않았습니다.
