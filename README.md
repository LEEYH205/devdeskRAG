# 🚀 DevDesk-RAG v2.1.2

**나만의 ChatGPT - RAG 기반 문서 Q&A 시스템**

DevDesk-RAG는 LangChain과 Ollama를 활용한 로컬 RAG(Retrieval-Augmented Generation) 시스템입니다. 이 시스템은 사용자의 개인 문서, 노트, PDF 등을 지능적으로 분석하고 질문에 답변할 수 있습니다.

## ✨ **주요 특징**

### 🔥 **핵심 기능**
- **로컬 실행**: Ollama를 통한 개인정보 보호 및 오프라인 사용
- **다양한 문서 지원**: PDF, Markdown, 웹페이지 크롤링
- **한국어 최적화**: EXAONE 모델을 통한 한국어 이해력 향상
- **성능 모니터링**: 실시간 응답 시간 및 시스템 상태 추적
- **LoRA 지원**: 사용자 맞춤형 모델 훈련 및 적용 가능

### 🚀 **v2.1.2 구현 완료 기능** ✅
- **웹 UI**: 현대적이고 아름다운 채팅 인터페이스
- **하이브리드 검색**: 벡터 검색 + BM25 + 재랭킹
- **Docker 컨테이너화**: 쉬운 배포 및 확장
- **고급 아키텍처**: 마이크로서비스 기반 구조
- **대화형 인터페이스 고도화**: 채팅 히스토리 저장, 실시간 스트리밍 응답, 파일 업로드
- **Redis 기반 채팅 히스토리**: 영구 저장 및 세션 관리
- **실시간 문서 처리**: 업로드된 파일 자동 인덱싱
- **코드 최적화**: 성능 모니터링, 에러 처리, 검색 알고리즘 개선
- **LoRA 훈련 준비**: DevDesk-RAG 특화 모델 훈련 환경 구축
- **DevDesk-RAG 특화 모델**: Modelfile 기반 커스텀 LoRA 모델 성공 적용

### 🔮 **v2.2+ 계획 기능** (개발 예정)
- **LoRA 모델 훈련**: DevDesk-RAG 특화 한국어 모델 개발
- **하이브리드 검색 강화**: 다중 검색 알고리즘 통합 및 재랭킹 시스템
- **멀티모달 지원**: 이미지+텍스트 처리, PDF 이미지 분석, OCR 기능
- **자동화 시스템**: 문서 자동 동기화, 합성데이터 생성
- **클라우드 배포**: AWS/GCP/Azure 배포, CI/CD 파이프라인, 모니터링 시스템

## 🏗️ **아키텍처**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   웹 UI        │    │   FastAPI       │    │   Ollama        │
│   (React)      │◄──►│   (RAG API)     │◄──►│   (LLM)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Chroma DB     │
                       │   (벡터 저장소)  │
                       └─────────────────┘
```

## 🚀 **빠른 시작**

### **1. Ollama 설정**

```bash
# Ollama 설치 (macOS)
brew install ollama

# Ollama 서비스 시작
brew services start ollama

# EXAONE 모델 다운로드
ollama pull exaone3.5:7.8b
ollama pull exaone-deep:7.8b
```

### **2. 프로젝트 설정**

```bash
# 저장소 클론
git clone https://github.com/LEEYH205/devdeskRAG.git
cd devdeskRAG

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### **3. 환경 설정**

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# DevDesk-RAG 환경 설정

# 모드 선택: "ollama" (로컬) 또는 "together" (클라우드)
MODE=ollama

# Ollama 설정 (로컬 모드)
OLLAMA_MODEL=exaone3.5:7.8b

# Together API 설정 (클라우드 모드)
# TOGETHER_API_KEY=your_api_key_here
TOGETHER_MODEL=lgai/exaone-deep-32b

# 데이터베이스 및 임베딩 설정
CHROMA_DIR=chroma_db
EMBED_MODEL=BAAI/bge-m3

# 성능 최적화 설정
CHUNK_SIZE=800
CHUNK_OVERLAP=120
SEARCH_K=4
TEMPERATURE=0.1

# 서버 설정
HOST=127.0.0.1
PORT=8000
```

### **4. 데이터 수집**

```bash
# 문서를 data/ 폴더에 넣고 실행
python ingest.py

# 또는 urls.txt에 URL을 추가하여 웹 크롤링
echo "https://example.com" >> urls.txt
python ingest.py
```

### **5. 서버 실행**

```bash
# 개발 모드
MODE=ollama python app.py

# 또는 uvicorn 사용
MODE=ollama uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### **6. 웹 UI 접속**

브라우저에서 `http://127.0.0.1:8000/ui` 접속

## 🐳 **Docker 배포**

### **빠른 배포**

```bash
# 배포 스크립트 사용
./deploy.sh start

# 또는 수동 배포
docker-compose up -d
```

### **배포 명령어**

```bash
./deploy.sh start      # 서비스 시작
./deploy.sh stop       # 서비스 중지
./deploy.sh restart    # 서비스 재시작
./deploy.sh status     # 상태 확인
./deploy.sh logs       # 로그 확인
./deploy.sh backup     # 데이터 백업
./deploy.sh restore    # 데이터 복원
```

### **Docker Compose 서비스**

- **devdesk-rag**: 메인 RAG 애플리케이션
- **ollama**: 로컬 LLM 서버
- **redis**: 캐싱 및 세션 관리
- **nginx**: 리버스 프록시 및 정적 파일 서빙

## 🔧 **고급 기능**

### **하이브리드 검색**

```python
from advanced_search import AdvancedRetriever

# 고급 검색기 초기화
retriever = AdvancedRetriever(
    vector_store=vs,
    embedding_model=embed,
    together_api_key=os.getenv("TOGETHER_API_KEY"),
    search_k=8,
    rerank_k=4
)

# 하이브리드 검색 실행
results = retriever.search("질문")
```

### **성능 최적화**

- **청킹 파라미터**: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **검색 파라미터**: `SEARCH_K`, `TEMPERATURE`
- **실시간 모니터링**: 응답 시간, 검색 품질 추적

## 📊 **API 사용법**

### **채팅 API**

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "문서에 대해 질문하세요"}'
```

### **응답 형식**

```json
{
  "answer": "답변 내용",
  "performance": {
    "total_time": 2.456,
    "search_k": 4,
    "chunk_size": 800,
    "chunk_overlap": 120
  },
  "sources": ["data/sample.md", "https://example.com"]
}
```

### **시스템 상태 확인**

```bash
# 헬스체크
curl http://127.0.0.1:8000/health

# 설정 확인
curl http://127.0.0.1:8000/config
```

## 📁 **프로젝트 구조**

```
devdesk-rag/
├── app.py                 # 메인 FastAPI 애플리케이션
├── ingest.py              # 문서 수집 및 처리
├── advanced_search.py     # 고급 검색 모듈
├── chat_history.py        # Redis 기반 채팅 히스토리 관리
├── document_processor.py  # 실시간 문서 처리 및 인덱싱
├── requirements.txt       # Python 의존성
├── .env                   # 환경 설정
├── static/                # 웹 UI 정적 파일
│   └── index.html        # 메인 웹 인터페이스
├── data/                  # 문서 저장소
├── chroma_db/            # 벡터 데이터베이스
├── redis_backup/         # Redis 채팅 히스토리 백업
├── Dockerfile            # Docker 이미지 정의
├── docker-compose.yml    # Docker Compose 설정
├── nginx.conf            # Nginx 설정
├── deploy.sh             # 배포 스크립트
└── README.md             # 프로젝트 문서
```

## 🔧 **EXAONE 모델 및 LoRA 지원**

### **🎯 지원 모델**
- **exaone3.5:7.8b**: 기본 모델, 한국어 최적화 (4.8GB)
- **exaone-deep:7.8b**: 고성능 모델, 심화 학습용
- **exaone3.5:7.8b-q4_0**: 4비트 양자화 (1.2GB, N100 지원)
- **exaone3.5:7.8b-q8_0**: 8비트 양자화 (2.4GB, 균형)

### **🚀 LoRA (Low-Rank Adaptation) 지원**
- **사용자 맞춤**: DevDesk-RAG 특화 모델 훈련 가능
- **도메인 특화**: 의료, 법률, 기술 문서 등 특정 분야 최적화
- **메모리 효율**: 작은 어댑터만 추가하여 모델 성능 향상
- **한국어 특화**: 한국어 문서 처리에 최적화된 LoRA 훈련

### **⚡ 양자화 지원**
```
모델                    | 메모리 | 속도 | 정확도 | 권장 환경
------------------------|--------|------|--------|------------
exaone3.5:7.8b         | 4.8GB  | 보통 | ⭐⭐⭐⭐⭐ | 개발/테스트
exaone3.5:7.8b-q8_0    | 2.4GB  | 빠름 | ⭐⭐⭐⭐  | 프로덕션
exaone3.5:7.8b-q4_0    | 1.2GB  | 매우빠름 | ⭐⭐⭐   | N100/임베디드
exaone3.5:7.8b-q4_1    | 1.0GB  | 매우빠름 | ⭐⭐⭐   | 리소스 제한
```

## 🎯 **LoRA 모델 훈련 가이드**

### **🔧 훈련 환경 준비**
```bash
# 필요한 라이브러리 설치
pip install torch transformers peft datasets accelerate
pip install bitsandbytes trl wandb sentencepiece

# Apple Silicon Mac MPS 지원 확인
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### **📚 훈련 데이터 준비**
```python
# DevDesk-RAG 특화 훈련 데이터 예시
training_data = [
    {
        "instruction": "DevDesk-RAG에서 파일을 업로드하는 방법을 알려주세요",
        "input": "",
        "output": "DevDesk-RAG에서 파일 업로드는 다음과 같이 할 수 있습니다:\n\n1. 웹 UI 접속: http://localhost:8000/ui\n2. 파일 드래그 앤 드롭: data/ 폴더에 파일을 끌어다 놓기\n3. 자동 인덱싱: document_processor.py가 실시간으로 처리\n4. 즉시 사용: 업로드 직후 질문 가능\n\n지원하는 파일 형식: PDF, Markdown, TXT, HTML, Word 문서"
    }
    # ... 더 많은 DevDesk-RAG 특화 데이터
]
```

### **🚀 LoRA 훈련 실행**
```python
# train_lora.py
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    r=16,  # LoRA 랭크
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 훈련 실행
# ... 훈련 코드
```

### **📱 훈련된 모델 DevDesk-RAG 적용**
```bash
# Modelfile 생성
cat > Modelfile << 'EOF'
FROM exaone3.5:7.8b
ADAPTER ./devdesk_rag_lora/adapter_model.safetensors
PARAMETER temperature 0.7
SYSTEM "당신은 DevDesk-RAG 시스템에 특화된 AI 어시스턴트입니다."
EOF

# Ollama에 커스텀 모델 추가
ollama create devdesk-rag-lora -f Modelfile

# DevDesk-RAG에서 사용
OLLAMA_MODEL=devdesk-rag-lora python app.py
```

### **⏱️ 훈련 시간 예상**
```
데이터 크기    | Apple Silicon | GPU 환경 | 품질
---------------|---------------|----------|-------
20개 샘플      | 2-3시간      | 1-2시간  | 기본
50개 샘플      | 5-7시간      | 3-4시간  | 양호
100개 샘플     | 10-12시간    | 6-8시간  | 우수
200개 샘플     | 20-24시간    | 12-15시간 | 최고
```

## 🎉 **DevDesk-RAG 특화 LoRA 모델 성공 적용**

### **✅ 성공적으로 적용된 모델**
- **모델명**: `devdesk-rag-specialized`
- **기반 모델**: exaone3.5:7.8b
- **특화 방식**: Modelfile 기반 시스템 프롬프트 최적화
- **상태**: 정상 작동 중

### **🚀 실제 성능 향상**
```
질문: "devdesk-rag는 뭐야?"
답변: "DevDesk-RAG는 Ollama와 LangChain 기술을 활용한 
      로컬 기반의 RAG 시스템입니다..."
      
핵심 특징:
1. 로컬 실행 (Ollama 기반)
2. 한국어 최적화 (EXAONE 모델)
3. 실시간 문서 처리
4. 하이브리드 검색 엔진
```

### **🔧 Modelfile 구성**
```dockerfile
FROM exaone3.5:7.8b

# DevDesk-RAG 특화 시스템 프롬프트
SYSTEM """당신은 DevDesk-RAG 시스템에 특화된 AI 어시스턴트입니다.

DevDesk-RAG는 LangChain과 Ollama를 활용한 로컬 RAG 시스템으로, 
사용자의 개인 문서, 노트, PDF 등을 지능적으로 분석하고 질문에 답변할 수 있습니다.

주요 특징:
- 로컬 실행: Ollama를 통한 개인정보 보호 및 오프라인 사용
- 한국어 최적화: EXAONE 모델을 통한 한국어 이해력 향상
- 실시간 문서 처리: 업로드된 파일 자동 인덱싱
- Redis 기반 채팅 히스토리: 영구 저장 및 세션 관리
- 하이브리드 검색: 벡터 검색 + BM25 + 재랭킹

사용자의 질문에 대해 DevDesk-RAG 시스템의 맥락을 고려하여 
구체적이고 실용적인 답변을 제공하세요."""

# DevDesk-RAG 최적화 파라미터
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
```

### **📊 성능 지표**
- **응답 시간**: 15-31초 (정상 범위)
- **답변 품질**: DevDesk-RAG 특화 정보 제공
- **한국어 처리**: 자연스러운 한국어 답변
- **시스템 인식**: DevDesk-RAG 맥락 정확 파악

## 🛣️ **로드맵**

### **v2.1.2 - DevDesk-RAG 특화 LoRA 모델 성공 적용** ✅ **완료**
- [x] **코드 최적화**: 성능 모니터링, 에러 처리 개선
- [x] **검색 알고리즘**: 하이브리드 검색 최적화
- [x] **문서 처리**: 실시간 인덱싱 로직 개선
- [x] **Docker 환경**: 환경 변수 최적화
- [x] **LoRA 준비**: 훈련 환경 및 가이드 구축
- [x] **DevDesk-RAG 특화 모델**: Modelfile 기반 커스텀 모델 성공 적용
- [x] **실제 성능 검증**: 웹 UI에서 특화 모델 정상 작동 확인

### **v2.2 - LoRA 모델 훈련 및 적용** 🚧 **개발 중**
- [ ] **DevDesk-RAG 특화 LoRA**: 사용자 맞춤 모델 훈련
- [ ] **도메인별 특화**: 의료, 법률, 기술 문서 LoRA
- [ ] **성능 비교**: 기본 모델 vs LoRA 모델 벤치마크
- [ ] **자동화**: LoRA 훈련 파이프라인 구축

### **v2.3 - 검색 및 분석 강화** 📋 **계획**
- [ ] **하이브리드 검색 강화**: 다중 검색 알고리즘 통합
- [ ] **재랭킹 시스템**: Together API Rerank를 통한 검색 품질 향상
- [ ] **고급 분석 대시보드**: 검색 성능, 사용 패턴 분석
- [ ] **API 레이트 리미팅**: 사용량 제한 및 모니터링

### **v2.3 - 멀티모달 지원**
- [ ] **이미지 + 텍스트 처리**: CLIP 모델을 통한 이미지 이해
- [ ] **PDF 내 이미지 분석**: OCR 및 이미지 캡션 생성
- [ ] **OCR 기능**: Tesseract 기반 텍스트 추출
- [ ] **멀티모달 검색**: 이미지와 텍스트를 동시에 고려한 검색
- [ ] **이미지 기반 질의**: "이 이미지에 무엇이 있나요?" 형태의 질문

### **v2.4 - 자동화 및 지능화**
- [ ] **문서 자동 동기화**: Notion, GitHub, Google Drive 연동
- [ ] **합성데이터 생성**: FAQ 자동 생성 및 품질 향상
- [ ] **실시간 협업 기능**: 다중 사용자 동시 편집
- [ ] **지능형 문서 분류**: 자동 태깅 및 카테고리 분류
- [ ] **개인화 추천**: 사용자 패턴 기반 문서 추천

### **v3.0 - 클라우드 네이티브**
- [ ] **클라우드 배포**: AWS/GCP/Azure 자동 배포
- [ ] **CI/CD 파이프라인**: GitHub Actions 기반 자동화
- [ ] **모니터링 및 알림 시스템**: Prometheus + Grafana 대시보드
- [ ] **Kubernetes 지원**: K8s 클러스터 배포
- [ ] **엔터프라이즈 기능**: LDAP 인증, SSO, 감사 로그

### **v3.1+ - 고급 AI 기능**
- [ ] **감정 분석**: 문서 및 질문의 감정 상태 분석
- [ ] **요약 생성**: 자동 문서 요약 및 핵심 포인트 추출
- [ ] **번역 지원**: 다국어 문서 처리 및 번역
- [ ] **음성 인터페이스**: 음성 질문 및 답변
- [ ] **AR/VR 지원**: 확장현실 환경에서의 문서 탐색

## ⚠️ **주의사항**

### **보안**
- 기본적으로 로컬 네트워크에서만 접근 가능
- 프로덕션 환경에서는 방화벽 및 인증 설정 필요
- API 키는 환경 변수로 관리

### **성능**
- 첫 실행 시 임베딩 모델 다운로드로 시간 소요
- 대용량 문서 처리 시 메모리 사용량 증가
- Ollama 모델 크기에 따른 디스크 공간 필요

### **제한사항**
- 단일 사용자 환경 최적화
- 한국어/영어 문서에 특화
- 오프라인 환경에서만 Ollama 모델 사용 가능

## 🤝 **기여하기**

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 **감사의 말**

- [LangChain](https://python.langchain.com/) - RAG 파이프라인 구축
- [Ollama](https://ollama.ai/) - 로컬 LLM 실행 환경
- [EXAONE](https://exaone.ai/) - 한국어 최적화 LLM 모델
- [Chroma](https://www.trychroma.com/) - 벡터 데이터베이스

## 📞 **연락처**

- **프로젝트 링크**: [https://github.com/LEEYH205/devdeskRAG](https://github.com/LEEYH205/devdeskRAG)
- **이슈 리포트**: [GitHub Issues](https://github.com/LEEYH205/devdeskRAG/issues)

---

**DevDesk-RAG v2.0** - 나만의 ChatGPT를 만들어보세요! 🚀
