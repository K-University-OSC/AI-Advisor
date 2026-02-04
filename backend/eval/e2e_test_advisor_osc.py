# -*- coding: utf-8 -*-
"""
Advisor OSC E2E 테스트

브라우저 사용자처럼 모든 기능을 테스트:
1. 회원가입/로그인
2. 채팅 (일반)
3. 문서 업로드 (RAG)
4. RAG 기반 질문 응답
5. 관리자 기능

실행: python e2e_test_advisor_osc.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import httpx

# 서비스 URL
BASE_URL = "http://localhost:10311"
FRONTEND_URL = "http://localhost:10310"

# 테스트 계정
TEST_USER = {
    "username": f"testuser_{int(time.time())}",
    "email": f"test_{int(time.time())}@example.com",
    "password": "TestPassword123!",
    "full_name": "E2E Test User"
}

TEST_ADMIN = {
    "username": "admin",
    "password": "admin123!"
}


class AdvisorOSCTester:
    """Advisor OSC E2E 테스터"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.user_token = None
        self.admin_token = None
        self.session_id = None
        self.results = []

    async def close(self):
        await self.client.aclose()

    def log(self, test_name: str, status: str, message: str = ""):
        """테스트 결과 로깅"""
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{icon} [{status}] {test_name}: {message}")
        self.results.append({
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    # ========================================================================
    # 1. Health Check
    # ========================================================================
    async def test_health_check(self):
        """서버 상태 확인"""
        try:
            response = await self.client.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                self.log("Health Check", "PASS", f"status={data.get('status')}, rag_based={data.get('rag_based')}")
                return True
            else:
                self.log("Health Check", "FAIL", f"status_code={response.status_code}")
                return False
        except Exception as e:
            self.log("Health Check", "FAIL", str(e))
            return False

    # ========================================================================
    # 2. 회원가입
    # ========================================================================
    async def test_signup(self):
        """회원가입 테스트"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/auth/signup",
                json=TEST_USER
            )
            if response.status_code == 200:
                data = response.json()
                self.log("회원가입", "PASS", f"user_id={data.get('user_id')}")
                return True
            elif response.status_code == 400:
                # 이미 존재하는 사용자
                self.log("회원가입", "WARN", "이미 존재하는 사용자 (기존 계정 사용)")
                return True
            else:
                self.log("회원가입", "FAIL", f"status={response.status_code}, {response.text}")
                return False
        except Exception as e:
            self.log("회원가입", "FAIL", str(e))
            return False

    # ========================================================================
    # 3. 로그인
    # ========================================================================
    async def test_login(self):
        """로그인 테스트"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/auth/login",
                json={
                    "username": TEST_USER["username"],
                    "password": TEST_USER["password"]
                }
            )
            if response.status_code == 200:
                data = response.json()
                self.user_token = data.get("access_token")
                self.log("로그인", "PASS", f"token_type={data.get('token_type')}")
                return True
            else:
                # 새 사용자로 다시 시도
                response = await self.client.post(
                    f"{self.base_url}/api/auth/signup",
                    json=TEST_USER
                )
                if response.status_code == 200:
                    response = await self.client.post(
                        f"{self.base_url}/api/auth/login",
                        json={
                            "username": TEST_USER["username"],
                            "password": TEST_USER["password"]
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        self.user_token = data.get("access_token")
                        self.log("로그인", "PASS", f"새 사용자로 로그인 성공")
                        return True
                self.log("로그인", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("로그인", "FAIL", str(e))
            return False

    # ========================================================================
    # 4. 사용자 정보 조회
    # ========================================================================
    async def test_me(self):
        """사용자 정보 조회 테스트"""
        if not self.user_token:
            self.log("사용자 정보", "SKIP", "토큰 없음")
            return False
        try:
            response = await self.client.get(
                f"{self.base_url}/api/auth/me",
                headers={"Authorization": f"Bearer {self.user_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                self.log("사용자 정보", "PASS", f"username={data.get('username')}, email={data.get('email')}")
                return True
            else:
                self.log("사용자 정보", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("사용자 정보", "FAIL", str(e))
            return False

    # ========================================================================
    # 5. 채팅 - 일반 대화
    # ========================================================================
    async def test_chat_general(self):
        """일반 채팅 테스트"""
        if not self.user_token:
            self.log("일반 채팅", "SKIP", "토큰 없음")
            return False
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat/send",
                headers={"Authorization": f"Bearer {self.user_token}"},
                json={
                    "message": "안녕하세요! 자기소개 해주세요.",
                    "session_id": None
                }
            )
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                answer = data.get("answer", "")[:100]
                self.log("일반 채팅", "PASS", f"session_id={self.session_id}, answer={answer}...")
                return True
            else:
                self.log("일반 채팅", "FAIL", f"status={response.status_code}, {response.text[:200]}")
                return False
        except Exception as e:
            self.log("일반 채팅", "FAIL", str(e))
            return False

    # ========================================================================
    # 6. 채팅 - 후속 질문
    # ========================================================================
    async def test_chat_followup(self):
        """후속 질문 테스트"""
        if not self.user_token or not self.session_id:
            self.log("후속 질문", "SKIP", "세션 없음")
            return False
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat/send",
                headers={"Authorization": f"Bearer {self.user_token}"},
                json={
                    "message": "방금 말한 내용을 한 문장으로 요약해줘.",
                    "session_id": self.session_id
                }
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")[:100]
                self.log("후속 질문", "PASS", f"answer={answer}...")
                return True
            else:
                self.log("후속 질문", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("후속 질문", "FAIL", str(e))
            return False

    # ========================================================================
    # 7. 세션 목록 조회
    # ========================================================================
    async def test_sessions_list(self):
        """세션 목록 조회 테스트"""
        if not self.user_token:
            self.log("세션 목록", "SKIP", "토큰 없음")
            return False
        try:
            response = await self.client.get(
                f"{self.base_url}/api/chat/sessions",
                headers={"Authorization": f"Bearer {self.user_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                sessions = data.get("sessions", [])
                self.log("세션 목록", "PASS", f"sessions_count={len(sessions)}")
                return True
            else:
                self.log("세션 목록", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("세션 목록", "FAIL", str(e))
            return False

    # ========================================================================
    # 8. 문서 업로드 (RAG)
    # ========================================================================
    async def test_document_upload(self):
        """문서 업로드 테스트"""
        if not self.user_token:
            self.log("문서 업로드", "SKIP", "토큰 없음")
            return False

        # 테스트용 PDF 파일 찾기
        test_pdf = Path(__file__).parent / "allganize/files/finance/WP22-05.pdf"
        if not test_pdf.exists():
            self.log("문서 업로드", "SKIP", "테스트 PDF 파일 없음")
            return False

        try:
            with open(test_pdf, "rb") as f:
                files = {"file": (test_pdf.name, f, "application/pdf")}
                response = await self.client.post(
                    f"{self.base_url}/api/chat/upload",
                    headers={"Authorization": f"Bearer {self.user_token}"},
                    files=files,
                    data={"domain": "finance"}
                )

            if response.status_code == 200:
                data = response.json()
                self.log("문서 업로드", "PASS", f"filename={data.get('filename')}, chunks={data.get('chunks_created', 'N/A')}")
                return True
            elif response.status_code == 400 and "already exists" in response.text.lower():
                self.log("문서 업로드", "WARN", "이미 업로드된 문서")
                return True
            else:
                self.log("문서 업로드", "FAIL", f"status={response.status_code}, {response.text[:200]}")
                return False
        except Exception as e:
            self.log("문서 업로드", "FAIL", str(e))
            return False

    # ========================================================================
    # 9. RAG 검색
    # ========================================================================
    async def test_rag_search(self):
        """RAG 검색 테스트"""
        if not self.user_token:
            self.log("RAG 검색", "SKIP", "토큰 없음")
            return False
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat/search",
                headers={"Authorization": f"Bearer {self.user_token}"},
                json={
                    "query": "연금 시장 규모",
                    "domain": "finance",
                    "top_k": 5
                }
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                self.log("RAG 검색", "PASS", f"results_count={len(results)}")
                return True
            else:
                self.log("RAG 검색", "FAIL", f"status={response.status_code}, {response.text[:200]}")
                return False
        except Exception as e:
            self.log("RAG 검색", "FAIL", str(e))
            return False

    # ========================================================================
    # 10. RAG 기반 질문 응답
    # ========================================================================
    async def test_rag_chat(self):
        """RAG 기반 질문 응답 테스트"""
        if not self.user_token:
            self.log("RAG 채팅", "SKIP", "토큰 없음")
            return False
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat/send",
                headers={"Authorization": f"Bearer {self.user_token}"},
                json={
                    "message": "한국 연금 시장의 규모는 얼마인가요?",
                    "session_id": None,
                    "use_rag": True,
                    "domain": "finance"
                }
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")[:150]
                sources = data.get("sources", [])
                self.log("RAG 채팅", "PASS", f"sources={len(sources)}, answer={answer}...")
                return True
            else:
                self.log("RAG 채팅", "FAIL", f"status={response.status_code}, {response.text[:200]}")
                return False
        except Exception as e:
            self.log("RAG 채팅", "FAIL", str(e))
            return False

    # ========================================================================
    # 11. 관리자 로그인
    # ========================================================================
    async def test_admin_login(self):
        """관리자 로그인 테스트"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/auth/login",
                json=TEST_ADMIN
            )
            if response.status_code == 200:
                data = response.json()
                self.admin_token = data.get("access_token")
                self.log("관리자 로그인", "PASS", f"token_type={data.get('token_type')}")
                return True
            else:
                self.log("관리자 로그인", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("관리자 로그인", "FAIL", str(e))
            return False

    # ========================================================================
    # 12. 관리자 대시보드
    # ========================================================================
    async def test_admin_dashboard(self):
        """관리자 대시보드 테스트"""
        if not self.admin_token:
            self.log("관리자 대시보드", "SKIP", "관리자 토큰 없음")
            return False
        try:
            response = await self.client.get(
                f"{self.base_url}/api/admin/dashboard",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                self.log("관리자 대시보드", "PASS", f"total_users={data.get('total_users')}, total_sessions={data.get('total_sessions')}")
                return True
            else:
                self.log("관리자 대시보드", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("관리자 대시보드", "FAIL", str(e))
            return False

    # ========================================================================
    # 13. 사용자 목록 조회
    # ========================================================================
    async def test_admin_users_list(self):
        """관리자 사용자 목록 조회 테스트"""
        if not self.admin_token:
            self.log("사용자 목록", "SKIP", "관리자 토큰 없음")
            return False
        try:
            response = await self.client.get(
                f"{self.base_url}/api/admin/users",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                users = data.get("users", [])
                self.log("사용자 목록", "PASS", f"users_count={len(users)}")
                return True
            else:
                self.log("사용자 목록", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("사용자 목록", "FAIL", str(e))
            return False

    # ========================================================================
    # 14. 문서 목록 조회
    # ========================================================================
    async def test_admin_documents_list(self):
        """관리자 문서 목록 조회 테스트"""
        if not self.admin_token:
            self.log("문서 목록", "SKIP", "관리자 토큰 없음")
            return False
        try:
            response = await self.client.get(
                f"{self.base_url}/api/admin/data/documents",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                self.log("문서 목록", "PASS", f"documents_count={len(documents)}")
                return True
            else:
                self.log("문서 목록", "FAIL", f"status={response.status_code}")
                return False
        except Exception as e:
            self.log("문서 목록", "FAIL", str(e))
            return False

    # ========================================================================
    # 메인 테스트 실행
    # ========================================================================
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("=" * 60)
        print("Advisor OSC E2E 테스트")
        print("=" * 60)
        print(f"서버: {self.base_url}")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 테스트 순서
        tests = [
            ("서버 상태", self.test_health_check),
            ("회원가입", self.test_signup),
            ("로그인", self.test_login),
            ("사용자 정보", self.test_me),
            ("일반 채팅", self.test_chat_general),
            ("후속 질문", self.test_chat_followup),
            ("세션 목록", self.test_sessions_list),
            ("문서 업로드", self.test_document_upload),
            ("RAG 검색", self.test_rag_search),
            ("RAG 채팅", self.test_rag_chat),
            ("관리자 로그인", self.test_admin_login),
            ("관리자 대시보드", self.test_admin_dashboard),
            ("사용자 목록 (관리자)", self.test_admin_users_list),
            ("문서 목록 (관리자)", self.test_admin_documents_list),
        ]

        print("\n테스트 실행 중...\n")

        passed = 0
        failed = 0
        skipped = 0

        for name, test_func in tests:
            try:
                result = await test_func()
                if result:
                    passed += 1
                else:
                    # 결과 확인
                    last_result = self.results[-1] if self.results else {}
                    if last_result.get("status") == "SKIP":
                        skipped += 1
                    elif last_result.get("status") == "WARN":
                        passed += 1  # WARN은 통과로 처리
                    else:
                        failed += 1
            except Exception as e:
                self.log(name, "FAIL", f"예외 발생: {e}")
                failed += 1

        # 결과 요약
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        total = passed + failed + skipped
        print(f"총 테스트: {total}")
        print(f"✅ 성공: {passed}")
        print(f"❌ 실패: {failed}")
        print(f"⚠️ 스킵: {skipped}")
        print(f"성공률: {passed/(total-skipped)*100:.1f}%" if total > skipped else "N/A")
        print("=" * 60)
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {"passed": passed, "failed": failed, "skipped": skipped}


async def main():
    tester = AdvisorOSCTester()
    try:
        results = await tester.run_all_tests()
        return results
    finally:
        await tester.close()


if __name__ == "__main__":
    results = asyncio.run(main())
    sys.exit(0 if results["failed"] == 0 else 1)
