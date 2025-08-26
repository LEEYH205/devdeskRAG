# 🚀 DevDesk-RAG v2.1

**나만의 ChatGPT - RAG 기반 문서 Q&A 시스템**

DevDesk-RAG는 LangChain과 Ollama를 활용한 로컬 RAG(Retrieval-Augmented Generation) 시스템입니다. 이 시스템은 사용자의 개인 문서, 노트, PDF 등을 지능적으로 분석하고 질문에 답변할 수 있습니다.

## ✨ **주요 특징**

### 🔥 **핵심 기능**
- **로컬 실행**: Ollama를 통한 개인정보 보호 및 오프라인 사용
- **다양한 문서 지원**: PDF, Markdown, 웹페이지 크롤링
- **한국어 최적화**: EXAONE 모델을 통한 한국어 이해력 향상
- **성능 모니터링**: 실시간 응답 시간 및 시스템 상태 추적

### 🚀 **v2.1 구현 완료 기능** ✅
- **웹 UI**: 현대적이고 아름다운 채팅 인터페이스
- **하이브리드 검색**: 벡터 검색 + BM25 + 재랭킹
- **Docker 컨테이너화**: 쉬운 배포 및 확장
- **고급 아키텍처**: 마이크로서비스 기반 구조
- **대화형 인터페이스 고도화**: 채팅 히스토리 저장, 실시간 스트리밍 응답, 파일 업로드
- **Redis 기반 채팅 히스토리**: 영구 저장 및 세션 관리
- **실시간 문서 처리**: 업로드된 파일 자동 인덱싱

### 🔮 **v2.2+ 계획 기능** (개발 예정)
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

## 🚀 **성능 최적화**

### **권장 설정**

```bash
# 고성능 환경
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
SEARCH_K=6
TEMPERATURE=0.05

# 메모리 제한 환경
CHUNK_SIZE=600
CHUNK_OVERLAP=80
SEARCH_K=3
TEMPERATURE=0.1
```

### **모니터링 지표**

- **응답 시간**: 평균 2-5초
- **검색 정확도**: 85-95%
- **메모리 사용량**: 2-4GB
- **동시 사용자**: 단일 사용자 최적화

## 🔧 **Redis 채팅 히스토리 관리**

### **Redis 백업 파일 위치**
```
devdesk-rag/
└── redis_backup/
    └── dump.rdb          # 채팅 히스토리 백업 파일
```

### **Redis 설정**
- **포트**: 6379 (기본)
- **백업 위치**: 프로젝트 폴더 내 `redis_backup/`
- **데이터 지속성**: RDB 스냅샷 방식
- **Git 추적**: 백업 폴더는 `.gitignore`에 포함

### **채팅 히스토리 기능**
- **세션 관리**: 사용자별 대화 세션 생성 및 관리
- **메시지 저장**: 질문과 답변의 완전한 기록
- **검색 기능**: 과거 대화 내용 검색
- **자동 정리**: 오래된 세션 자동 삭제

## 🔍 **문제 해결**

### **일반적인 문제**

#### **Ollama 연결 오류**
```bash
# Ollama 서비스 상태 확인
brew services list | grep ollama

# Ollama 재시작
brew services restart ollama

# 모델 다운로드 확인
ollama list
```

#### **메모리 부족**
```bash
# 청킹 파라미터 조정
CHUNK_SIZE=600
CHUNK_OVERLAP=80

# 검색 범위 축소
SEARCH_K=3
```

#### **Docker 컨테이너 오류**
```bash
# 컨테이너 로그 확인
docker-compose logs devdesk-rag

# 컨테이너 재시작
docker-compose restart devdesk-rag
```

#### **Redis 연결 오류**
```bash
# Redis 서비스 상태 확인
brew services list | grep redis

# Redis 재시작
brew services restart redis

# Redis 백업 파일 확인
ls -la redis_backup/
```

#### **채팅 히스토리 손실**
```bash
# Redis 백업 파일 복원
cp redis_backup/dump.rdb /opt/homebrew/var/db/redis/

# Redis 재시작
brew services restart redis
```

### **로그 확인**

```bash
# 애플리케이션 로그
tail -f app.log

# Docker 로그
docker-compose logs -f

# Nginx 로그
docker exec devdesk-nginx tail -f /var/log/nginx/access.log
```

## 📈 **모니터링 및 알림**

### **시스템 메트릭**

- **API 응답 시간**: `/health` 엔드포인트
- **메모리 사용량**: Docker stats
- **검색 품질**: 사용자 피드백 기반

### **알림 설정**

```bash
# 헬스체크 스크립트
while true; do
  if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "시스템 오류 발생: $(date)" | mail -s "DevDesk-RAG Alert" admin@example.com
  fi
  sleep 300
done
```

## 🛣️ **로드맵**

### **v2.1 - 대화형 인터페이스 고도화** ✅ **완료**
- [x] **채팅 히스토리 저장**: Redis 기반 대화 기록 관리
- [x] **실시간 스트리밍 응답**: Server-Sent Events를 통한 점진적 답변 생성
- [x] **파일 업로드 기능**: 드래그 앤 드롭으로 문서 추가
- [x] **Redis 백업 관리**: 프로젝트 폴더 내 채팅 히스토리 보관
- [ ] **사용자 인증 및 권한 관리**: API 키 기반 보안 시스템
- [ ] **문서 버전 관리**: 변경 이력 추적 및 롤백 기능

### **v2.2 - 검색 및 분석 강화** 🚧 **개발 중**
- [ ] **하이브리드 검색 강화**: 다중 검색 알고리즘 통합
- [ ] **재랭킹 시스템**: Together API Rerank를 통한 검색 품질 향상
- [ ] **고급 분석 대시보드**: 검색 성능, 사용 패턴 분석
- [ ] **API 레이트 리미팅**: 사용량 제한 및 모니터링
- [ ] **검색 결과 필터링**: 날짜, 출처, 문서 타입별 필터

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
