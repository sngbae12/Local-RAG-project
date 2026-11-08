# 모델 준비

앱에는 **EXAONE 답변 모델**과 **BGE-M3 임베딩 모델** 두 가지가 필요합니다. 폴더만 있거나 EXAONE 파일만 있으면 정상 실행되지 않습니다. 모델 가중치는 GitHub ZIP에 포함되지 않습니다.

## 자동 다운로드

프로젝트 루트에서 `setup.bat`을 실행하면 패키지 설치와 모델 다운로드를 진행합니다. 패키지를 이미 설치했다면 CMD에서:

```bat
.venv\Scripts\python.exe download_models.py
```

필요한 EXAONE GGUF 하나와 BGE-M3의 SentenceTransformer 파일만 받습니다. ONNX 모델이나 다른 GGUF 양자화 버전을 함께 받지 않습니다. 경로는 스크립트가 있는 프로젝트 폴더를 기준으로 결정합니다.

## 검사와 복구

| 명령 옵션 | 동작 |
| --- | --- |
| 옵션 없음 | 공식 파일 다운로드 후 기본 구조 검사 |
| `--check` | 인터넷 연결 없이 누락·빈 파일·JSON·GGUF 헤더 등 기본 검사 |
| `--repair-config` | JSON 파일 강제 재다운로드. 가중치에는 강제 재다운로드를 요청하지 않음 |
| `--force` | 필요한 파일 전체를 강제 재다운로드. 가중치 약 4.2GB 포함 |

```bat
.venv\Scripts\python.exe download_models.py --check
.venv\Scripts\python.exe download_models.py --repair-config
```

기본 검사는 전체 체크섬 검증이나 실제 모델 추론 시험이 아닙니다. 다운로드 중 오류가 나면 성공으로 표시하지 않습니다. 다운로드 도구는 `uploads/`와 `chroma_db/`를 삭제하지 않습니다.

## 필요한 파일

| 위치 | 형식·규모 |
| --- | --- |
| `models/exaone_2.4b/EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf` | EXAONE GGUF, 약 1.91GB |
| `models/bge-m3/pytorch_model.bin` | BGE-M3 가중치, 약 2.27GB |
| `models/bge-m3/config.json` | `model_type`이 `xlm-roberta`인 공식 모델 설정 |
| `models/bge-m3/modules.json` | SentenceTransformer 모듈 구성 |
| `models/bge-m3/config_sentence_transformers.json` | 라이브러리 설정 |
| `models/bge-m3/sentence_bert_config.json` | 문장 임베딩 설정 |
| `models/bge-m3/tokenizer.json` | 토크나이저 |
| `models/bge-m3/tokenizer_config.json` | 토크나이저 설정 |
| `models/bge-m3/special_tokens_map.json` | 특수 토큰 설정 |
| `models/bge-m3/sentencepiece.bpe.model` | 토큰 분할 모델 |
| `models/bge-m3/1_Pooling/config.json` | 풀링 설정 |

BGE-M3는 GGUF 파일이 아니라 Hugging Face/SentenceTransformer 형식의 폴더입니다. 앱은 CPU에서 `local_files_only=True`로 읽습니다. `modules.json`의 Normalize 모듈은 라이브러리에서 제공되며 별도 가중치 폴더가 필요하지 않습니다.

## 출처와 이용 조건

모델 다운로드 전에 각 원본 저장소의 조건을 읽어 주세요. 프로젝트의 코드와 별개로 모델 자체의 이용 조건이 적용됩니다.

- EXAONE: [LG AI Research 공식 GGUF 저장소](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF). 모델 카드에 **EXAONE AI Model License Agreement 1.1 - NC** 표기
- BGE-M3: [BAAI 공식 저장소](https://huggingface.co/BAAI/bge-m3). 모델 카드에 **MIT** 표기

모델 로딩 오류는 [설치·실행 오류 해결](../docs/troubleshooting.md)을 참고하세요.
