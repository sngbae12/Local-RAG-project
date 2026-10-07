# 모델 배치

이 폴더는 실행에 필요한 로컬 모델을 둡니다. 용량이 커서 Git에는 올리지 않습니다.

## 디렉터리

```text
models/
  exaone_2.4b/
    EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf
  bge-m3/
    config.json
    modules.json
    config_sentence_transformers.json
    sentence_bert_config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    sentencepiece.bpe.model
    pytorch_model.bin
    1_Pooling/config.json
```

경로는 `config.py`에서 프로젝트 루트 기준 상대경로로 읽습니다.

## LLM (GGUF)

- 파일명: `EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf`
- 배치: `models/exaone_2.4b/EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf`
- 로더: `llama-cpp-python` / LangChain `LlamaCpp`
- 이 저장소의 로컬 캐시 메타데이터에는 다운로드 URL이 없고 파일 해시만 있습니다. 배포처는 해당 파일명으로 직접 확인하세요.

## 임베딩 (BGE-M3)

- 형식: HuggingFace / SentenceTransformer 폴더 (GGUF가 아님)
- 배치: `models/bge-m3/`
- 로더: LangChain `HuggingFaceEmbeddings`, `device="cpu"`, `local_files_only=True`
- 로컬에 포함된 모델 카드 기준 공개 저장소: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- 코드와 프로젝트 내 README: [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)

허깅페이스 CLI 예시:

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

모델 이용 조건은 각 모델 카드의 라이선스를 확인하세요. BGE-M3 카드에는 MIT가 적혀 있습니다. EXAONE GGUF의 이용 조건은 이 저장소에서 확인하지 못했습니다.
