"""
Gunicorn 프로덕션 설정
200명 이상 동시 사용자 지원을 위한 최적화
"""
import multiprocessing
import os

# GPU 설정 (5번 GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 서버 바인딩
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8600")

# 워커 설정
# 테스트 환경: 4개 고정 (운영 환경에서는 CPU 코어 수 * 2 + 1 권장)
workers = int(os.getenv("GUNICORN_WORKERS", 4))
worker_class = "uvicorn.workers.UvicornWorker"

# 워커 타임아웃 (LLM 응답 대기 시간 고려)
timeout = 300  # 5분
graceful_timeout = 60
keepalive = 5

# 동시 연결 수 (워커당)
worker_connections = 1000

# 메모리 누수 방지: 요청 N개 처리 후 워커 재시작
max_requests = 1000
max_requests_jitter = 50

# 프리포크 (워커 사전 생성)
preload_app = True

# 로깅
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = "access.log"
errorlog = "error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 프로세스 이름
proc_name = "llm-chatbot"

# 임시 파일 디렉토리 (메모리 기반)
worker_tmp_dir = "/dev/shm"

# 요청 제한
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190


def on_starting(server):
    """서버 시작 시 호출"""
    print(f"🚀 Starting Gunicorn with {workers} workers...")


def on_exit(server):
    """서버 종료 시 호출"""
    print("👋 Gunicorn shutdown complete")


def worker_int(worker):
    """워커 인터럽트 시 호출"""
    print(f"Worker {worker.pid} interrupted")


def worker_abort(worker):
    """워커 비정상 종료 시 호출"""
    print(f"Worker {worker.pid} aborted")
