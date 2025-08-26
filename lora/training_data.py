#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DevDesk-RAG 특화 LoRA 훈련 데이터 생성기
DevDesk-RAG 시스템에 특화된 한국어 모델을 훈련하기 위한 데이터셋을 생성합니다.
"""

import json
from typing import List, Dict

def create_devdesk_rag_training_data() -> List[Dict]:
    """DevDesk-RAG 특화 훈련 데이터 생성"""
    
    training_data = [
        {
            "instruction": "DevDesk-RAG에서 파일을 업로드하는 방법을 알려주세요",
            "input": "",
            "output": "DevDesk-RAG에서 파일 업로드는 다음과 같이 할 수 있습니다:\n\n1. 웹 UI 접속: http://localhost:8000/ui\n2. 파일 드래그 앤 드롭: data/ 폴더에 파일을 끌어다 놓기\n3. 자동 인덱싱: document_processor.py가 실시간으로 처리\n4. 즉시 사용: 업로드 직후 질문 가능\n\n지원하는 파일 형식: PDF, Markdown, TXT, HTML, Word 문서\n\nAPI를 통한 업로드도 가능합니다:\n- POST /upload: 단일 파일 업로드\n- POST /upload/multiple: 다중 파일 업로드"
        },
        {
            "instruction": "DevDesk-RAG의 청킹 설정을 최적화하는 방법을 설명해주세요",
            "input": "",
            "output": "DevDesk-RAG의 청킹 최적화 방법:\n\n현재 설정:\n- CHUNK_SIZE=800 (기본값)\n- CHUNK_OVERLAP=120 (기본값)\n\n환경별 최적화:\n1. N100 미니PC (8GB RAM):\n   - CHUNK_SIZE=600\n   - CHUNK_OVERLAP=80\n\n2. 표준 환경 (16GB RAM):\n   - CHUNK_SIZE=800\n   - CHUNK_OVERLAP=120\n\n3. 고성능 환경 (32GB RAM):\n   - CHUNK_SIZE=1000\n   - CHUNK_OVERLAP=150\n\n설정 방법:\n- .env 파일 수정\n- 런타임 환경변수 설정\n- Docker 환경변수 설정\n\n실제로 현재 시스템의 설정을 확인하려면:\ncurl http://localhost:8000/config"
        },
        {
            "instruction": "DevDesk-RAG에서 Redis 채팅 히스토리는 어떻게 관리되나요?",
            "input": "",
            "output": "DevDesk-RAG의 Redis 채팅 히스토리 관리:\n\n1. 저장 위치: redis_backup/dump.rdb\n2. 관리 모듈: chat_history.py\n3. 주요 기능:\n   - 세션별 대화 기록 저장\n   - 메시지 검색 및 필터링\n   - 자동 세션 정리\n   - 영구 저장 및 백업\n\n4. 설정: REDIS_URL 환경변수로 연결\n5. 백업: 프로젝트 폴더 내 안전한 보관\n6. Git 추적: .gitignore에 포함하여 민감정보 보호\n\n7. API 엔드포인트:\n   - GET /chat/history/{session_id}: 세션별 대화 기록\n   - DELETE /chat/history/{session_id}: 세션 삭제\n   - GET /chat/search: 대화 내용 검색"
        },
        {
            "instruction": "DevDesk-RAG의 벡터 데이터베이스는 어떻게 구성되어 있나요?",
            "input": "",
            "output": "DevDesk-RAG의 벡터 데이터베이스 구성:\n\n1. 벡터 DB: ChromaDB 사용\n2. 저장 위치: chroma_db/ 폴더\n3. 임베딩 모델: BAAI/bge-m3\n4. 문서 처리: document_processor.py\n5. 주요 기능:\n   - 실시간 문서 인덱싱\n   - 중복 문서 처리\n   - 벡터 검색 최적화\n   - 하이브리드 검색 지원\n\n6. 검색 방식:\n   - 벡터 유사도 검색\n   - BM25 키워드 검색\n   - 재랭킹 시스템\n   - 다중 검색 알고리즘 통합\n\n7. 관리 API:\n   - GET /documents/status: 벡터 DB 상태 확인\n   - POST /documents/refresh: 전체 시스템 새로고침\n   - DELETE /files/{filename}: 특정 문서 제거"
        },
        {
            "instruction": "DevDesk-RAG에서 Ollama 모델을 변경하는 방법을 알려주세요",
            "input": "",
            "output": "DevDesk-RAG에서 Ollama 모델 변경 방법:\n\n1. 환경변수 설정:\n   - .env 파일: OLLAMA_MODEL=exaone-deep:7.8b\n   - 런타임: export OLLAMA_MODEL=exaone-deep:7.8b\n\n2. 지원 모델:\n   - exaone3.5:7.8b (기본, 한국어 최적화)\n   - exaone-deep:7.8b (고성능)\n   - exaone3.5:7.8b-q4_0 (4bit 양자화)\n   - exaone3.5:7.8b-q8_0 (8bit 양자화)\n\n3. 모델 다운로드:\n   - ollama pull exaone-deep:7.8b\n   - ollama pull exaone3.5:7.8b-q4_0\n\n4. 서버 재시작: python app.py\n\n5. 모델 확인:\n   - ollama list\n   - curl http://localhost:8000/config"
        },
        {
            "instruction": "DevDesk-RAG의 하이브리드 검색은 어떻게 작동하나요?",
            "input": "",
            "output": "DevDesk-RAG의 하이브리드 검색 작동 방식:\n\n1. 검색 알고리즘 조합:\n   - 벡터 검색: 임베딩 기반 유사도 검색\n   - BM25 검색: 키워드 기반 검색\n   - 재랭킹: Together API를 통한 결과 순서 최적화\n\n2. 검색 과정:\n   - 사용자 질문 분석\n   - 벡터 검색으로 관련 문서 후보 선별\n   - BM25로 키워드 매칭 강화\n   - 재랭킹으로 최종 순서 결정\n   - 상위 결과를 LLM에 전달하여 답변 생성\n\n3. 설정 파라미터:\n   - SEARCH_K: 검색할 문서 수 (기본값: 4)\n   - TEMPERATURE: 답변 창의성 (기본값: 0.1)\n   - CHUNK_SIZE: 문서 청킹 크기\n\n4. 성능 최적화:\n   - 검색 정확도: 85-95%\n   - 응답 시간: 평균 2-5초\n   - 메모리 사용량: 2-4GB"
        },
        {
            "instruction": "DevDesk-RAG에서 Docker를 사용한 배포 방법을 알려주세요",
            "input": "",
            "output": "DevDesk-RAG Docker 배포 방법:\n\n1. Docker Compose 서비스 구성:\n   - devdesk-rag: 메인 RAG 애플리케이션\n   - ollama: 로컬 LLM 서버\n   - redis: 캐싱 및 세션 관리\n   - nginx: 리버스 프록시 및 정적 파일 서빙\n\n2. 배포 스크립트 사용:\n   - ./deploy.sh start: 서비스 시작\n   - ./deploy.sh stop: 서비스 중지\n   - ./deploy.sh restart: 서비스 재시작\n   - ./deploy.sh status: 상태 확인\n   - ./deploy.sh logs: 로그 확인\n\n3. 환경 설정:\n   - docker-compose.yml에서 환경변수 설정\n   - .env 파일을 통한 설정 관리\n   - 볼륨 마운트로 데이터 영속성 보장\n\n4. 포트 구성:\n   - 8000: FastAPI 애플리케이션\n   - 6379: Redis\n   - 11434: Ollama\n   - 80: Nginx\n\n5. 데이터 백업:\n   - chroma_db/: 벡터 데이터베이스\n   - redis_backup/: 채팅 히스토리\n   - data/: 업로드된 문서들"
        },
        {
            "instruction": "DevDesk-RAG의 성능을 모니터링하는 방법은 무엇인가요?",
            "input": "",
            "output": "DevDesk-RAG 성능 모니터링 방법:\n\n1. API 엔드포인트:\n   - GET /health: 시스템 상태 확인\n   - GET /config: 현재 설정 확인\n   - GET /documents/status: 문서 처리 상태\n\n2. 실시간 모니터링:\n   - 응답 시간 측정: 각 API 호출별 소요 시간\n   - 메모리 사용량: Docker stats 명령어\n   - 검색 품질: 사용자 피드백 기반\n\n3. 로그 확인:\n   - 애플리케이션 로그: app.py 실행 로그\n   - Docker 로그: docker-compose logs -f\n   - Nginx 로그: 접근 및 오류 로그\n\n4. 성능 지표:\n   - 평균 응답 시간: 2-5초\n   - 검색 정확도: 85-95%\n   - 메모리 사용량: 2-4GB\n   - 동시 사용자: 단일 사용자 최적화\n\n5. 최적화 팁:\n   - 청킹 파라미터 조정\n   - 검색 범위 최적화\n   - 모델 양자화 적용\n   - 하드웨어 리소스 모니터링"
        },
        {
            "instruction": "DevDesk-RAG에서 문서 처리 오류가 발생했을 때 어떻게 해결하나요?",
            "input": "",
            "output": "DevDesk-RAG 문서 처리 오류 해결 방법:\n\n1. 일반적인 오류 유형:\n   - 파일 형식 지원 안됨\n   - 메모리 부족\n   - 임베딩 모델 로딩 실패\n   - 벡터 DB 연결 오류\n\n2. 단계별 해결 과정:\n   - 로그 확인: app.py 실행 로그 확인\n   - 파일 형식 검증: 지원 확장자 확인\n   - 메모리 상태 확인: 시스템 리소스 점검\n   - 서비스 재시작: 관련 서비스 재시작\n\n3. 구체적 해결 방법:\n   - 지원 파일 형식: PDF, MD, TXT, HTML, DOC\n   - 메모리 부족 시: CHUNK_SIZE 줄이기\n   - 임베딩 오류: 모델 재다운로드\n   - DB 오류: chroma_db 폴더 정리\n\n4. 디버깅 도구:\n   - GET /documents/status: 처리 상태 확인\n   - GET /health: 시스템 상태 점검\n   - 로그 레벨 조정: DEBUG 모드 활성화\n\n5. 예방 방법:\n   - 정기적인 시스템 점검\n   - 로그 모니터링\n   - 백업 데이터 관리\n   - 성능 최적화 설정"
        },
        {
            "instruction": "DevDesk-RAG의 보안 기능은 어떻게 구성되어 있나요?",
            "input": "",
            "output": "DevDesk-RAG 보안 기능 구성:\n\n1. 기본 보안 설정:\n   - 로컬 네트워크 접근 제한\n   - CORS 설정으로 웹 보안 강화\n   - 파일 업로드 확장자 검증\n   - API 요청 검증 및 필터링\n\n2. 데이터 보안:\n   - 민감 정보 Git 추적 제외\n   - Redis 백업 파일 보안 관리\n   - 업로드 파일 검증 및 격리\n   - 벡터 DB 접근 제어\n\n3. 네트워크 보안:\n   - 기본 포트 8000 사용\n   - 방화벽 설정 권장\n   - HTTPS 적용 가능 (프록시 설정)\n   - API 키 기반 인증 준비\n\n4. 파일 보안:\n   - 지원 확장자만 허용\n   - 파일 크기 제한\n   - 악성 파일 검사 준비\n   - 업로드 디렉토리 격리\n\n5. 향후 보안 계획:\n   - 사용자 인증 시스템\n   - API 레이트 리미팅\n   - 감사 로그 시스템\n   - 암호화 저장소 지원"
        },
        {
            "instruction": "DevDesk-RAG를 클라우드에 배포하려면 어떻게 해야 하나요?",
            "input": "",
            "output": "DevDesk-RAG 클라우드 배포 방법:\n\n1. 클라우드 플랫폼 선택:\n   - AWS: EC2, ECS, Lambda\n   - GCP: Compute Engine, Cloud Run\n   - Azure: Virtual Machines, Container Instances\n   - DigitalOcean: Droplets, App Platform\n\n2. 배포 아키텍처:\n   - 컨테이너 기반: Docker + Kubernetes\n   - 서버리스: AWS Lambda + API Gateway\n   - 가상머신: EC2 + Docker Compose\n   - 관리형 서비스: ECS, Cloud Run\n\n3. 필수 구성 요소:\n   - Ollama 서버: LLM 모델 실행\n   - Redis: 채팅 히스토리 관리\n   - ChromaDB: 벡터 데이터베이스\n   - Nginx: 리버스 프록시\n\n4. 환경 설정:\n   - 환경변수 관리: 클라우드 시크릿 매니저\n   - 데이터 영속성: 클라우드 스토리지\n   - 로그 관리: CloudWatch, Stackdriver\n   - 모니터링: Prometheus + Grafana\n\n5. CI/CD 파이프라인:\n   - GitHub Actions: 자동 빌드 및 배포\n   - Docker Hub: 이미지 저장소\n   - 테스트 자동화: 단위/통합 테스트\n   - 롤백 전략: 이전 버전 복구"
        }
    ]
    
    return training_data

def format_training_data(data: List[Dict]) -> List[Dict]:
    """훈련 데이터를 EXAONE 모델에 맞게 포맷팅"""
    formatted_data = []
    
    for item in data:
        # EXAONE 모델 입력 형식
        formatted_item = {
            "text": f"### 질문: {item['instruction']}\n\n{item['input']}\n\n### 답변:\n{item['output']}"
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def save_training_data(data: List[Dict], filename: str):
    """훈련 데이터를 JSONL 형식으로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    """메인 실행 함수"""
    print("🚀 DevDesk-RAG 특화 LoRA 훈련 데이터 생성 시작...")
    
    # 1. 훈련 데이터 생성
    print("📚 DevDesk-RAG 특화 훈련 데이터 생성 중...")
    training_data = create_devdesk_rag_training_data()
    print(f"✅ {len(training_data)}개의 훈련 샘플 생성 완료")
    
    # 2. 데이터 포맷팅
    print("🔧 EXAONE 모델용 데이터 포맷팅 중...")
    formatted_data = format_training_data(training_data)
    print("✅ 데이터 포맷팅 완료")
    
    # 3. JSONL 파일로 저장
    print("💾 훈련 데이터 저장 중...")
    save_training_data(formatted_data, "devdesk_rag_training.jsonl")
    print("✅ devdesk_rag_training.jsonl 저장 완료")
    
    # 4. 샘플 데이터 출력
    print("\n📖 샘플 훈련 데이터:")
    print("=" * 50)
    for i, item in enumerate(formatted_data[:2], 1):
        print(f"\n--- 샘플 {i} ---")
        print(item['text'][:200] + "...")
    
    print(f"\n🎉 DevDesk-RAG 특화 훈련 데이터 생성 완료!")
    print(f"📊 총 {len(formatted_data)}개 샘플")
    print(f"📁 파일: devdesk_rag_training.jsonl")
    print(f"🚀 이제 LoRA 훈련을 시작할 수 있습니다!")

if __name__ == "__main__":
    main()
