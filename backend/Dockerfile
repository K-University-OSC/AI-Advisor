# =============================================================================
# AI Advisor Backend - Dockerfile
# =============================================================================
#
# 빌드: docker build -t advisor-backend .
# 실행: docker run -p 8890:8890 --gpus '"device=0"' advisor-backend
#
# =============================================================================

FROM python:3.10-slim

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# shared_libs 경로 설정 (런타임에 마운트됨)
# 패키지 구조: shared_libs/personalization_core/personalization_core/
#              shared_libs/mh_rag_core/mh_rag_core/
ENV PYTHONPATH="/app/shared_libs/personalization_core:/app/shared_libs/mh_rag_core:/app/shared_libs:$PYTHONPATH"

# uploads 디렉토리 생성
RUN mkdir -p /app/uploads

# 헬스체크용 curl 설치 확인
RUN curl --version

# 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["python", "-u", "server.py"]
