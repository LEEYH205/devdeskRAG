#!/bin/bash

# DevDesk-RAG 배포 스크립트
# 사용법: ./deploy.sh [start|stop|restart|build|logs|status]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 환경 확인
check_environment() {
    log_info "환경 확인 중..."
    
    # Docker 확인
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되지 않았습니다."
        exit 1
    fi
    
    # Docker Compose 확인
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose가 설치되지 않았습니다."
        exit 1
    fi
    
    # 필요한 파일 확인
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml 파일을 찾을 수 없습니다."
        exit 1
    fi
    
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile을 찾을 수 없습니다."
        exit 1
    fi
    
    log_success "환경 확인 완료"
}

# 서비스 시작
start_service() {
    log_info "DevDesk-RAG 서비스 시작 중..."
    
    # 기존 컨테이너 정리
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # 서비스 시작
    docker-compose up -d
    
    # 서비스 상태 확인
    sleep 10
    check_service_status
    
    log_success "서비스 시작 완료"
    log_info "웹 UI: http://localhost"
    log_info "API: http://localhost:8000"
}

# 서비스 중지
stop_service() {
    log_info "DevDesk-RAG 서비스 중지 중..."
    docker-compose down
    log_success "서비스 중지 완료"
}

# 서비스 재시작
restart_service() {
    log_info "DevDesk-RAG 서비스 재시작 중..."
    stop_service
    start_service
}

# 이미지 빌드
build_image() {
    log_info "Docker 이미지 빌드 중..."
    docker-compose build --no-cache
    log_success "이미지 빌드 완료"
}

# 로그 확인
show_logs() {
    log_info "서비스 로그 확인 중..."
    docker-compose logs -f
}

# 서비스 상태 확인
check_service_status() {
    log_info "서비스 상태 확인 중..."
    
    # 컨테이너 상태
    echo
    docker-compose ps
    
    # 헬스체크
    echo
    log_info "헬스체크 결과:"
    
    # API 헬스체크
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API 서비스: 정상"
    else
        log_error "API 서비스: 오류"
    fi
    
    # 웹 UI 헬스체크
    if curl -s http://localhost > /dev/null 2>&1; then
        log_success "웹 UI: 정상"
    else
        log_error "웹 UI: 오류"
    fi
    
    # Ollama 헬스체크
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama: 정상"
    else
        log_warning "Ollama: 연결 안됨"
    fi
}

# 데이터 백업
backup_data() {
    log_info "데이터 백업 중..."
    
    BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # 벡터 데이터베이스 백업
    if [ -d "chroma_db" ]; then
        cp -r chroma_db "$BACKUP_DIR/"
        log_success "벡터 데이터베이스 백업 완료"
    fi
    
    # 문서 데이터 백업
    if [ -d "data" ]; then
        cp -r data "$BACKUP_DIR/"
        log_success "문서 데이터 백업 완료"
    fi
    
    # 설정 파일 백업
    cp .env "$BACKUP_DIR/" 2>/dev/null || true
    cp docker-compose.yml "$BACKUP_DIR/"
    
    log_success "백업 완료: $BACKUP_DIR"
}

# 데이터 복원
restore_data() {
    if [ -z "$1" ]; then
        log_error "복원할 백업 디렉토리를 지정해주세요."
        log_info "사용법: ./deploy.sh restore <backup_directory>"
        exit 1
    fi
    
    BACKUP_DIR="$1"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "백업 디렉토리를 찾을 수 없습니다: $BACKUP_DIR"
        exit 1
    fi
    
    log_info "데이터 복원 중: $BACKUP_DIR"
    
    # 서비스 중지
    stop_service
    
    # 데이터 복원
    if [ -d "$BACKUP_DIR/chroma_db" ]; then
        rm -rf chroma_db
        cp -r "$BACKUP_DIR/chroma_db" .
        log_success "벡터 데이터베이스 복원 완료"
    fi
    
    if [ -d "$BACKUP_DIR/data" ]; then
        rm -rf data
        cp -r "$BACKUP_DIR/data" .
        log_success "문서 데이터 복원 완료"
    fi
    
    # 서비스 시작
    start_service
    
    log_success "데이터 복원 완료"
}

# 도움말
show_help() {
    echo "DevDesk-RAG 배포 스크립트"
    echo
    echo "사용법: $0 [명령어]"
    echo
    echo "명령어:"
    echo "  start     - 서비스 시작"
    echo "  stop      - 서비스 중지"
    echo "  restart   - 서비스 재시작"
    echo "  build     - Docker 이미지 빌드"
    echo "  logs      - 로그 확인"
    echo "  status    - 서비스 상태 확인"
    echo "  backup    - 데이터 백업"
    echo "  restore   - 데이터 복원"
    echo "  help      - 도움말 표시"
    echo
    echo "예시:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 backup"
    echo "  $0 restore backup_20241201_143022"
}

# 메인 로직
main() {
    case "${1:-help}" in
        start)
            check_environment
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            check_environment
            restart_service
            ;;
        build)
            check_environment
            build_image
            ;;
        logs)
            show_logs
            ;;
        status)
            check_service_status
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "알 수 없는 명령어: $1"
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"
