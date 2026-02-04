"""
AI Advisor OSC - 성능 테스트 (Locust)

실행 방법:
    locust -f tests/performance/locustfile.py --host http://localhost:10311

    # Headless 모드
    locust -f tests/performance/locustfile.py --headless -u 50 -r 10 --run-time 1m --host http://localhost:10311
"""
import os
import random
from locust import HttpUser, task, between, events


class AdvisorUser(HttpUser):
    """일반 사용자 시뮬레이션"""

    wait_time = between(1, 3)  # 요청 간 대기 시간 (1-3초)

    # 테스트 메시지 목록
    TEST_MESSAGES = [
        "안녕하세요",
        "학칙에 대해 알려주세요",
        "휴학 절차가 어떻게 되나요?",
        "장학금 신청 방법을 알려주세요",
        "수강신청은 언제 하나요?",
        "졸업 요건이 무엇인가요?",
        "성적 확인은 어디서 하나요?",
    ]

    def on_start(self):
        """사용자 세션 시작 시 로그인"""
        self.token = None
        self.tenant_id = os.getenv("TENANT_ID", "default")

        # 로그인 시도
        response = self.client.post("/api/auth/login",
            json={
                "username": os.getenv("TEST_USERNAME", "testuser"),
                "password": os.getenv("TEST_PASSWORD", "test1234")
            },
            headers={"X-Tenant-ID": self.tenant_id},
            name="/api/auth/login"
        )

        if response.status_code == 200:
            self.token = response.json().get("access_token")
            self.headers = {
                "Authorization": f"Bearer {self.token}",
                "X-Tenant-ID": self.tenant_id
            }
        else:
            self.headers = {"X-Tenant-ID": self.tenant_id}

    @task(10)
    def chat(self):
        """채팅 API 호출 (가장 빈번한 작업)"""
        if not self.token:
            return

        message = random.choice(self.TEST_MESSAGES)

        self.client.post("/api/chat",
            json={"message": message},
            headers=self.headers,
            name="/api/chat"
        )

    @task(5)
    def get_chat_history(self):
        """채팅 기록 조회"""
        if not self.token:
            return

        self.client.get("/api/chat/history",
            headers=self.headers,
            name="/api/chat/history"
        )

    @task(2)
    def get_documents(self):
        """문서 목록 조회"""
        if not self.token:
            return

        self.client.get("/api/documents",
            headers=self.headers,
            name="/api/documents"
        )

    @task(1)
    def health_check(self):
        """헬스체크 (인증 불필요)"""
        self.client.get("/api/health", name="/api/health")

    @task(1)
    def get_user_info(self):
        """현재 사용자 정보 조회"""
        if not self.token:
            return

        self.client.get("/api/auth/me",
            headers=self.headers,
            name="/api/auth/me"
        )


class AdminUser(HttpUser):
    """관리자 사용자 시뮬레이션"""

    wait_time = between(3, 8)  # 관리 작업은 덜 빈번함
    weight = 1  # 일반 사용자보다 적은 비율

    def on_start(self):
        """관리자 세션 시작"""
        self.token = None
        self.tenant_id = os.getenv("TENANT_ID", "default")

        # 관리자 로그인
        response = self.client.post("/api/auth/login",
            json={
                "username": os.getenv("ADMIN_USERNAME", "admin"),
                "password": os.getenv("ADMIN_PASSWORD", "admin1234")
            },
            headers={"X-Tenant-ID": self.tenant_id},
            name="/api/auth/login (admin)"
        )

        if response.status_code == 200:
            self.token = response.json().get("access_token")
            self.headers = {
                "Authorization": f"Bearer {self.token}",
                "X-Tenant-ID": self.tenant_id
            }
        else:
            self.headers = {"X-Tenant-ID": self.tenant_id}

    @task(3)
    def get_all_users(self):
        """사용자 목록 조회 (관리자)"""
        if not self.token:
            return

        self.client.get("/api/admin/users",
            headers=self.headers,
            name="/api/admin/users"
        )

    @task(2)
    def get_statistics(self):
        """통계 조회 (관리자)"""
        if not self.token:
            return

        self.client.get("/api/admin/statistics",
            headers=self.headers,
            name="/api/admin/statistics"
        )

    @task(1)
    def get_documents(self):
        """문서 관리"""
        if not self.token:
            return

        self.client.get("/api/documents",
            headers=self.headers,
            name="/api/documents (admin)"
        )


# 이벤트 핸들러
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=== 성능 테스트 시작 ===")
    print(f"Host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("=== 성능 테스트 종료 ===")

    # 결과 요약
    stats = environment.stats
    print(f"총 요청 수: {stats.total.num_requests}")
    print(f"실패 수: {stats.total.num_failures}")
    print(f"평균 응답 시간: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
