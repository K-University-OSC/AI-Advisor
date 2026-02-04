#!/bin/bash
# =============================================================================
# AI Advisor OSC - 테스트 실행 스크립트
# =============================================================================
#
# 사용법:
#   ./scripts/run-tests.sh all        # 모든 테스트
#   ./scripts/run-tests.sh unit       # Unit 테스트
#   ./scripts/run-tests.sh api        # API 테스트
#   ./scripts/run-tests.sh e2e        # E2E 테스트 (Playwright)
#   ./scripts/run-tests.sh isolation  # 테넌트 격리 테스트
#   ./scripts/run-tests.sh perf       # 성능 테스트
#   ./scripts/run-tests.sh security   # 보안 테스트
#
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 프로젝트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# 테스트 결과 디렉토리 생성
mkdir -p tests/e2e/test-results
mkdir -p tests/api/test-results
mkdir -p tests/performance/test-results

# Unit 테스트
run_unit_tests() {
    log_info "=== Unit 테스트 실행 (pytest) ==="
    cd backend

    if [ ! -d "venv" ]; then
        log_warning "가상환경이 없습니다. 직접 실행합니다."
        python -m pytest tests/unit -v --cov=. --cov-report=html --cov-report=term-missing
    else
        source venv/bin/activate
        python -m pytest tests/unit -v --cov=. --cov-report=html --cov-report=term-missing
        deactivate
    fi

    cd ..
    log_success "Unit 테스트 완료!"
}

# API 테스트
run_api_tests() {
    log_info "=== API 통합 테스트 실행 (pytest + httpx) ==="

    # 서비스 상태 확인
    if ! curl -s http://localhost:10311/api/health > /dev/null; then
        log_error "Backend가 실행 중이 아닙니다. 먼저 서비스를 시작하세요."
        exit 1
    fi

    cd tests/api
    python -m pytest -v \
        --html=test-results/api-report.html \
        --self-contained-html \
        --tb=short
    cd ../..

    log_success "API 테스트 완료!"
    log_info "리포트: tests/api/test-results/api-report.html"
}

# E2E 테스트 (Playwright)
run_e2e_tests() {
    log_info "=== E2E 테스트 실행 (Playwright) ==="

    # Docker Compose로 Playwright 실행
    if [ "$1" == "docker" ]; then
        docker compose -f docker-compose.yml -f docker-compose.test.yml run --rm playwright
    else
        # 로컬 실행
        cd tests/e2e

        if [ ! -d "node_modules" ]; then
            log_info "의존성 설치 중..."
            npm ci
        fi

        log_info "Playwright 브라우저 설치 확인..."
        npx playwright install --with-deps chromium

        log_info "테스트 실행..."
        npx playwright test --reporter=html

        cd ../..
    fi

    log_success "E2E 테스트 완료!"
    log_info "리포트: tests/e2e/test-results/index.html"
}

# 테넌트 격리 테스트
run_isolation_tests() {
    log_info "=== 테넌트 격리 테스트 실행 ==="

    if ! curl -s http://localhost:10311/api/health > /dev/null; then
        log_error "Backend가 실행 중이 아닙니다."
        exit 1
    fi

    cd tests/api
    python -m pytest test_tenant_isolation.py -v \
        --html=test-results/isolation-report.html \
        --self-contained-html \
        -x  # 첫 번째 실패에서 중단
    cd ../..

    log_success "테넌트 격리 테스트 완료!"
}

# 성능 테스트
run_performance_tests() {
    log_info "=== 성능 테스트 실행 (Locust) ==="

    USERS=${1:-50}
    SPAWN_RATE=${2:-10}
    RUN_TIME=${3:-1m}

    log_info "설정: Users=$USERS, Spawn Rate=$SPAWN_RATE, Run Time=$RUN_TIME"

    if command -v locust &> /dev/null; then
        locust -f tests/performance/locustfile.py \
            --headless \
            -u $USERS \
            -r $SPAWN_RATE \
            --run-time $RUN_TIME \
            --host http://localhost:10311 \
            --html=tests/performance/test-results/locust-report.html
    else
        # Docker로 실행
        docker compose -f docker-compose.yml -f docker-compose.test.yml run --rm locust
    fi

    log_success "성능 테스트 완료!"
    log_info "리포트: tests/performance/test-results/locust-report.html"
}

# 보안 테스트
run_security_tests() {
    log_info "=== 보안 테스트 실행 ==="

    cd tests/security
    python -m pytest security_scan.py -v \
        --html=../api/test-results/security-report.html \
        --self-contained-html
    cd ../..

    log_success "보안 테스트 완료!"
}

# 모든 테스트
run_all_tests() {
    log_info "=========================================="
    log_info "         모든 테스트 실행                 "
    log_info "=========================================="

    FAILED=0

    # 1. Unit Tests
    log_info ""
    run_unit_tests || { log_error "Unit 테스트 실패!"; FAILED=1; }

    # 2. API Tests
    log_info ""
    run_api_tests || { log_error "API 테스트 실패!"; FAILED=1; }

    # 3. Tenant Isolation Tests
    log_info ""
    run_isolation_tests || { log_error "테넌트 격리 테스트 실패!"; FAILED=1; }

    # 4. E2E Tests
    log_info ""
    run_e2e_tests || { log_error "E2E 테스트 실패!"; FAILED=1; }

    # 5. Security Tests
    log_info ""
    run_security_tests || { log_error "보안 테스트 실패!"; FAILED=1; }

    # 6. Performance Tests (선택적)
    if [ "$SKIP_PERF" != "true" ]; then
        log_info ""
        run_performance_tests 30 5 30s || { log_warning "성능 테스트 실패 (비필수)"; }
    fi

    log_info ""
    log_info "=========================================="
    if [ $FAILED -eq 0 ]; then
        log_success "모든 테스트 통과!"
    else
        log_error "일부 테스트 실패!"
        exit 1
    fi
}

# 헬프
show_help() {
    echo "AI Advisor OSC 테스트 스크립트"
    echo ""
    echo "사용법: ./scripts/run-tests.sh [명령] [옵션]"
    echo ""
    echo "명령:"
    echo "  all        모든 테스트 실행"
    echo "  unit       Unit 테스트 (pytest)"
    echo "  api        API 통합 테스트"
    echo "  e2e        E2E 테스트 (Playwright)"
    echo "  isolation  테넌트 격리 테스트"
    echo "  perf       성능 테스트 (Locust)"
    echo "  security   보안 테스트"
    echo ""
    echo "옵션:"
    echo "  e2e docker    Docker로 E2E 테스트 실행"
    echo "  perf 100 20   성능 테스트 (100 users, 20 spawn rate)"
}

# 메인
case "${1:-help}" in
    all)
        run_all_tests
        ;;
    unit)
        run_unit_tests
        ;;
    api)
        run_api_tests
        ;;
    e2e)
        run_e2e_tests "$2"
        ;;
    isolation)
        run_isolation_tests
        ;;
    perf)
        run_performance_tests "$2" "$3" "$4"
        ;;
    security)
        run_security_tests
        ;;
    help|*)
        show_help
        ;;
esac
