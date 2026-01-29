# AI Advisor Changelog

모든 주요 변경 사항이 이 파일에 기록됩니다.

---

## [2026-01-04] Docker Compose 및 모니터링/자동복구 시스템 구축

### 추가됨 (Added)

#### Docker Compose 완전 격리 환경
- `docker-compose.yml` - 개발 환경 전용 Docker Compose 설정
  - Backend, Frontend, PostgreSQL, Redis, Qdrant 완전 격리
  - GPU 할당 (cuda:0)
  - 헬스체크 설정
  - 리소스 제한 (메모리)

- `backend/Dockerfile` - Backend 컨테이너 이미지
  - Python 3.10 기반
  - 의존성 자동 설치

- `frontend/Dockerfile` - Frontend 컨테이너 이미지
  - Node 18 빌드 → Nginx 서빙
  - SPA 라우팅 지원

- `docker/init-db.sql` - PostgreSQL 초기화 스크립트
  - advisor_central, tenant_hallym DB 생성
  - 테스트 계정 자동 생성

#### 모니터링 시스템
- `monitor.sh` - 개발 환경 상태 모니터링
  - Backend API (8890) 헬스체크
  - Frontend (8891) 상태 확인
  - PostgreSQL 연결 상태
  - Qdrant, Redis 상태
  - GPU 0 메모리/프로세스 현황
  - Docker Compose 컨테이너 상태
  - `--watch` 옵션으로 실시간 모니터링 (5초 간격)

#### 자동 복구 시스템
- `auto_recovery.sh` - 자동 복구 스크립트
  - Backend 다운 시 자동 재시작
  - Frontend 다운 시 자동 재시작
  - PostgreSQL 연결 풀 관리 (idle 연결 정리)
  - Docker Compose 컨테이너 헬스체크
  - GPU 점유 감지 (경고)
  - 복구 로그 기록 (`recovery.log`)

#### Cron 작업 등록
- 1시간마다 자동 복구 스크립트 실행
  ```
  0 * * * * /home/aiedu/workspace/advisor/auto_recovery.sh >> /home/aiedu/workspace/advisor/recovery.log 2>&1
  ```

### 포트 구성 (개발 환경)

| 서비스 | 포트 | 컨테이너명 |
|--------|------|-----------|
| Backend | 8890 | advisor-dev-backend |
| Frontend | 8891 | advisor-dev-frontend |
| PostgreSQL | 15432 | advisor-dev-postgres |
| Redis | 16380 | advisor-dev-redis |
| Qdrant | 16333 | advisor-dev-qdrant |

### 사용법

```bash
# 모니터링
./monitor.sh              # 1회 상태 확인
./monitor.sh --watch      # 실시간 모니터링

# 수동 복구
./auto_recovery.sh

# Docker Compose
docker compose up -d      # 전체 스택 시작
docker compose down       # 전체 스택 중지
docker compose logs -f    # 로그 확인
docker compose up -d --build  # 재빌드 후 시작
```

### 프로덕션 환경 영향
- **영향 없음**: 프로덕션 환경(8893, 8610)은 별도로 운영됨
- 프로덕션 모니터링/복구 스크립트는 `/home/aiedu/production/advisor/`에 별도 존재

---

## [이전 변경 사항]

### [2025-12-28] V7.7.2 Hybrid Smart Child Fallback
- RAG 파이프라인 성능 개선 (93.3% 정확도)
- 4Method Voting 테스트 결과: V7 81.7%

### [2025-12-27] 멀티테넌트 아키텍처 구현
- Database Per Tenant 패턴 적용
- Central DB 분리 (`advisor_central_prod`)
- 테넌트별 DB URL 템플릿

### [2025-12-24] 초기 프로젝트 설정
- LLM Chatbot 기반 AI Advisor 프로젝트 시작
- 개발/프로덕션 환경 분리
