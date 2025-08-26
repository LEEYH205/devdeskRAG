# DevDesk-RAG 🚀

**나만의 ChatGPT - RAG 기반 문서 Q&A 시스템**

LangChain + RAG + Ollama + EXAONE 조합으로 만든 로컬/클라우드 호환 문서 질의응답 시스템입니다.

## ✨ 주요 기능

- 📚 **다양한 문서 지원**: PDF, Markdown, 웹페이지 크롤링
- 🔍 **하이브리드 검색**: 벡터 검색 + BM25 (선택적 재랭킹)
- 🤖 **로컬/클라우드 LLM**: Ollama (EXAONE) + Together API
- 🌐 **웹 API**: FastAPI 기반 RESTful API
- 🇰🇷 **한국어 최적화**: EXAONE 모델로 한국어 답변 품질 향상

## 🏗️ 아키텍처

```
문서 수집 → 청킹/임베딩 → 벡터DB → 검색 → RAG 체인 → 답변 생성
    ↓           ↓         ↓       ↓       ↓         ↓
  PDF/MD/     텍스트    Chroma   Top-K   LangChain  EXAONE
  웹페이지    분할      벡터DB   검색    LCEL      LLM
```

## 🚀 빠른 시작

### 1. 사전 준비

#### A) 로컬 LLM: Ollama 설치
```bash
# macOS
brew install ollama

# 또는
curl -fsSL https://ollama.com/install.sh | sh

# EXAONE 모델 다운로드
ollama pull exaone3.5:7.8b
# 또는
ollama pull exaone-deep:7.8b
```

#### B) 클라우드 LLM: Together API (선택사항)
```bash
# Together API 키 발급 후
export TOGETHER_API_KEY="your_api_key_here"
```

### 2. 프로젝트 설정

```bash
# 저장소 클론
git clone https://github.com/LEEYH205/devdeskRAG.git
cd devdeskRAG

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 설정

`.env` 파일을 편집하여 모드를 선택하세요:

```bash
# 로컬 모드 (기본값)
MODE=ollama
OLLAMA_MODEL=exaone3.5:7.8b

# 클라우드 모드
# MODE=together
# TOGETHER_API_KEY=your_api_key_here
# TOGETHER_MODEL=lgai/exaone-deep-32b

# 데이터베이스 및 임베딩 설정
CHROMA_DIR=chroma_db
EMBED_MODEL=BAAI/bge-m3

# 서버 설정
HOST=127.0.0.1
PORT=8000
```

### 4. 문서 수집 및 인덱싱

```bash
# data/ 폴더에 문서 추가 또는 urls.txt에 URL 추가
echo "https://example.com" > urls.txt

# 문서 수집 및 벡터DB 생성
python ingest.py
```

### 5. 서버 실행

```bash
# 로컬 모드
MODE=ollama python app.py

# 또는 uvicorn으로 실행
MODE=ollama uvicorn app:app --reload --host 127.0.0.1 --port 8000

# 클라우드 모드
MODE=together TOGETHER_API_KEY=your_key python app.py
```

## 📁 프로젝트 구조

```
devdeskRAG/
├── data/              # PDF/MD/HTML 등 원천 문서
├── urls.txt           # 수집할 URL 목록
├── ingest.py          # 문서 수집·청킹·임베딩
├── app.py             # FastAPI + LangChain RAG
├── test_request.py    # API 테스트 스크립트
├── requirements.txt   # Python 의존성
├── .env              # 환경 변수
└── README.md         # 프로젝트 문서
```

## 🔧 API 사용법

### 헬스체크
```bash
curl http://127.0.0.1:8000/health
```

### 질문하기
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "이 자료의 핵심 요약은?"}'
```

### Python으로 테스트
```bash
python test_request.py
```

## 📊 성능 최적화

### 청킹 파라미터 튜닝
```python
# ingest.py에서 조정
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # 청크 크기
    chunk_overlap=120,   # 오버랩
    separators=["\n\n", "\n", " ", ""]
)
```

### 검색 파라미터 튜닝
```python
# app.py에서 조정
retriever = vs.as_retriever(
    search_kwargs={
        "k": 4,           # 검색 결과 수
        "score_threshold": 0.7  # 점수 임계값
    }
)
```

## 🔍 문제 해결

### 일반적인 문제들

1. **Ollama 연결 실패**
   ```bash
   # Ollama 서비스 상태 확인
   ollama list
   # 서비스 재시작
   ollama serve
   ```

2. **벡터DB 오류**
   ```bash
   # 데이터베이스 재생성
   rm -rf chroma_db/
   python ingest.py
   ```

3. **메모리 부족**
   ```python
   # 임베딩 모델을 CPU로 강제 설정
   embed = HuggingFaceEmbeddings(
       model_name="BAAI/bge-m3",
       model_kwargs={'device': 'cpu'}
   )
   ```

4. **AttributeError: 'dict' object has no attribute 'replace'**
   ```bash
   # 이 오류는 임베딩 모델의 입력 처리 문제입니다
   # 최신 버전의 sentence-transformers를 사용하세요
   pip install --upgrade sentence-transformers
   ```

## 📈 모니터링 및 평가

### 성능 지표
- **정답률**: Top-k@{3,4,8}에서의 정답률
- **근거 포함율**: 출처 표기된 답변 비율 (≥95% 목표)
- **안전응답율**: "모르겠다" 응답 비율 (≥5% 목표)

### 로그 확인
```bash
# API 로그
tail -f app.log

# 벡터DB 상태
ls -la chroma_db/
```

## 🚀 확장 가능성

- **자동 동기화**: Notion, GitHub 자동 문서 수집
- **합성데이터**: FAQ 자동 생성으로 품질 향상
- **재랭킹**: Together API의 Rerank 기능으로 정확도 개선
- **Docker**: 컨테이너화로 배포 간소화

## 📚 참고 자료

- [HMG Developers LLM Tutorial With RAG 시리즈](https://hmg-developers.tistory.com/)
- [LangChain LCEL 가이드](https://python.langchain.com/docs/expression_language/)
- [EXAONE in Ollama](https://ollama.ai/library/exaone)
- [Together AI EXAONE Deep 32B](https://together.ai/models/lgai/exaone-deep-32b)

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**Made with ❤️ by DevDesk Team**

## 🎯 현재 상태

- ✅ **완료**: 기본 RAG 시스템 구축
- ✅ **완료**: FastAPI 서버 실행
- ✅ **완료**: 문서 수집 및 벡터DB 생성
- ✅ **완료**: 기본 API 엔드포인트 작동
- 🔄 **진행중**: Ollama 연결 및 RAG 기능 테스트
- 📋 **예정**: 성능 최적화 및 확장 기능
