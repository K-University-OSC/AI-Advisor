#!/bin/bash
# =============================================================================
# AI Advisor OSC - 배포 스크립트
# =============================================================================
#
# 사용법:
#   ./scripts/deploy.sh dev       # 개발 환경
#   ./scripts/deploy.sh prod      # 프로덕션 환경
#   ./scripts/deploy.sh test      # 테스트 환경
#   ./scripts/deploy.sh k8s       # Kubernetes 배포
#   ./scripts/deploy.sh scale 3   # Backend 스케일링
#   ./scripts/deploy.sh stop      # 중지
#   ./scripts/deploy.sh logs      # 로그 확인
#   ./scripts/deploy.sh status    # 상태 확인
#
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 함수 정의
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

# 환경 변수 확인
check_env() {
    if [ ! -f ".env" ] && [ ! -f "backend/.env.docker" ]; then
        log_warning ".env 파일이 없습니다. .env.example을 복사합니다."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_info ".env 파일을 생성했습니다. API 키를 설정해주세요."
        fi
    fi
}

# 개발 환경 배포
deploy_dev() {
    log_info "=== 개발 환경 배포 ==="
    check_env

    docker compose up -d

    log_info "서비스 상태 확인 중..."
    sleep 5
    docker compose ps

    log_success "개발 환경 배포 완료!"
    log_info "Frontend: http://localhost:10310"
    log_info "Backend:  http://localhost:10311"
    log_info "API Docs: http://localhost:10311/docs"
}

# 프로덕션 환경 배포
deploy_prod() {
    log_info "=== 프로덕션 환경 배포 ==="
    check_env

    # 이미지 빌드
    log_info "Docker 이미지 빌드 중..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml build

    # 배포
    log_info "서비스 시작 중..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

    log_info "서비스 상태 확인 중..."
    sleep 10
    docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

    log_success "프로덕션 환경 배포 완료!"
}

# 테스트 환경 배포
deploy_test() {
    log_info "=== 테스트 환경 배포 ==="

    docker compose -f docker-compose.yml -f docker-compose.test.yml up -d postgres redis qdrant backend frontend

    log_info "서비스 시작 대기 중..."
    sleep 15

    log_success "테스트 환경 준비 완료!"
    log_info "테스트 실행: ./scripts/run-tests.sh"
}

# Kubernetes 배포
deploy_k8s() {
    log_info "=== Kubernetes 배포 ==="

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl이 설치되어 있지 않습니다."
        exit 1
    fi

    log_info "Namespace 생성..."
    kubectl apply -f k8s/namespace.yaml

    log_info "ConfigMap & Secrets 적용..."
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml

    log_info "데이터베이스 서비스 배포..."
    kubectl apply -f k8s/postgres-deployment.yaml
    kubectl apply -f k8s/redis-deployment.yaml
    kubectl apply -f k8s/qdrant-deployment.yaml

    log_info "애플리케이션 배포..."
    kubectl apply -f k8s/backend-deployment.yaml
    kubectl apply -f k8s/frontend-deployment.yaml

    log_info "HPA 설정..."
    kubectl apply -f k8s/backend-hpa.yaml

    log_info "Ingress 설정..."
    kubectl apply -f k8s/ingress.yaml

    log_info "Network Policy 설정..."
    kubectl apply -f k8s/network-policy.yaml

    log_success "Kubernetes 배포 완료!"
    kubectl get pods -n advisor-osc
}

# 스케일링
scale_backend() {
    REPLICAS=${1:-3}
    log_info "=== Backend 스케일링: $REPLICAS 개 ==="

    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale backend=$REPLICAS

    log_success "스케일링 완료!"
    docker compose ps
}

# 서비스 중지
stop_services() {
    log_info "=== 서비스 중지 ==="
    docker compose down
    log_success "서비스 중지 완료!"
}

# 로그 확인
show_logs() {
    SERVICE=${1:-""}
    if [ -z "$SERVICE" ]; then
        docker compose logs -f --tail=100
    else
        docker compose logs -f --tail=100 "$SERVICE"
    fi
}

# 상태 확인
show_status() {
    log_info "=== 서비스 상태 ==="
    docker compose ps

    echo ""
    log_info "=== 헬스체크 ==="

    # Frontend
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:10310 | grep -q "200"; then
        log_success "Frontend (10310): OK"
    else
        log_error "Frontend (10310): FAILED"
    fi

    # Backend
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:10311/api/health | grep -q "200"; then
        log_success "Backend (10311): OK"
    else
        log_error "Backend (10311): FAILED"
    fi

    echo ""
    log_info "=== 리소스 사용량 ==="
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
}

# 헬프
show_help() {
    echo "AI Advisor OSC 배포 스크립트"
    echo ""
    echo "사용법: ./scripts/deploy.sh [명령] [옵션]"
    echo ""
    echo "명령:"
    echo "  dev       개발 환경 배포"
    echo "  prod      프로덕션 환경 배포"
    echo "  test      테스트 환경 배포"
    echo "  k8s       Kubernetes 배포"
    echo "  scale N   Backend를 N개로 스케일링"
    echo "  stop      서비스 중지"
    echo "  logs      로그 확인 (logs [서비스명])"
    echo "  status    상태 확인"
    echo "  help      도움말"
}

# 메인
case "${1:-help}" in
    dev)
        deploy_dev
        ;;
    prod)
        deploy_prod
        ;;
    test)
        deploy_test
        ;;
    k8s)
        deploy_k8s
        ;;
    scale)
        scale_backend "$2"
        ;;
    stop)
        stop_services
        ;;
    logs)
        show_logs "$2"
        ;;
    status)
        show_status
        ;;
    help|*)
        show_help
        ;;
esac
