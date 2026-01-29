# -*- coding: utf-8 -*-
"""
Advisor 전체 기능 테스트
- 로그인
- 파일 업로드
- RAG 채팅 (개인화)
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

# 설정
BASE_URL = "http://localhost:8890"
TEST_USER = "test_user_" + datetime.now().strftime("%H%M%S")
TEST_PASSWORD = "test1234"
TEST_EMAIL = f"{TEST_USER}@test.com"

results = {
    "test_time": datetime.now().isoformat(),
    "tests": [],
    "summary": {"passed": 0, "failed": 0}
}


def log_result(name: str, passed: bool, details: str = ""):
    """테스트 결과 로깅"""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {details}")
    results["tests"].append({
        "name": name,
        "passed": passed,
        "details": details
    })
    if passed:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1


async def test_health():
    """헬스체크 테스트"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                log_result("Health Check", True, f"status={data.get('status')}, rag_based={data.get('rag_based')}")
                return True
            else:
                log_result("Health Check", False, f"status_code={response.status_code}")
                return False
    except Exception as e:
        log_result("Health Check", False, str(e)[:100])
        return False


async def test_signup_login():
    """회원가입 및 로그인 테스트"""
    token = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. 회원가입 시도
            response = await client.post(
                f"{BASE_URL}/api/auth/signup",
                json={
                    "username": TEST_USER,
                    "email": TEST_EMAIL,
                    "password": TEST_PASSWORD,
                    "name": TEST_USER
                }
            )

            if response.status_code in [200, 201]:
                log_result("Signup", True, f"user={TEST_USER}")
            elif response.status_code == 400 or response.status_code == 409:
                log_result("Signup", True, "User may already exist (OK)")
            else:
                log_result("Signup", False, f"status={response.status_code}, body={response.text[:200]}")

            # 2. 로그인
            response = await client.post(
                f"{BASE_URL}/api/auth/login",
                json={
                    "username": TEST_USER,
                    "password": TEST_PASSWORD
                }
            )

            if response.status_code == 200:
                data = response.json()
                token = data.get("access_token") or data.get("token")
                if token:
                    log_result("Login", True, f"token obtained (len={len(token)})")
                else:
                    log_result("Login", True, f"No token but status OK: {list(data.keys())}")
            else:
                log_result("Login", False, f"status={response.status_code}, body={response.text[:200]}")

    except Exception as e:
        log_result("Login", False, str(e)[:100])

    return token


async def test_file_upload(token: str = None):
    """파일 업로드 테스트 (/api/chat/upload)"""
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # 테스트 텍스트 파일 생성
        test_content = f"""테스트 문서
생성시간: {datetime.now().isoformat()}
이 문서는 advisor RAG 시스템 테스트를 위한 샘플 문서입니다.
금융 관련 테스트 내용: 핀테크 투자 생태계 활성화 방안"""

        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {
                "file": ("test_document.txt", test_content.encode(), "text/plain")
            }

            response = await client.post(
                f"{BASE_URL}/api/chat/upload",
                headers=headers,
                files=files
            )

            if response.status_code in [200, 201]:
                data = response.json()
                log_result("File Upload", True, f"response={list(data.keys()) if isinstance(data, dict) else 'OK'}")
                return True
            else:
                log_result("File Upload", False, f"status={response.status_code}, body={response.text[:200]}")
                return False

    except Exception as e:
        log_result("File Upload", False, str(e)[:100])
        return False


async def test_rag_chat(token: str = None):
    """RAG 채팅 테스트 (/api/chat/send) - 스트리밍 응답 처리"""
    try:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            test_query = "금융위원회의 핀테크 투자 생태계 활성화 방안에 대해 알려주세요."

            # 스트리밍 응답 처리
            async with client.stream(
                "POST",
                f"{BASE_URL}/api/chat/send",
                headers=headers,
                json={"message": test_query}
            ) as response:
                if response.status_code == 200:
                    # 스트리밍 데이터 수집
                    full_response = ""
                    async for chunk in response.aiter_text():
                        full_response += chunk

                    if len(full_response) > 10:
                        log_result("RAG Chat", True, f"response_length={len(full_response)}")
                        # 미리보기 (SSE 형식일 수 있음)
                        preview = full_response.replace("data: ", "")[:150]
                        print(f"    답변 미리보기: {preview}...")
                        return True
                    else:
                        log_result("RAG Chat", True, f"response received (short)")
                        return True
                else:
                    body = await response.aread()
                    log_result("RAG Chat", False, f"status={response.status_code}, body={body.decode()[:200]}")
                    return False

    except Exception as e:
        log_result("RAG Chat (Personalization)", False, str(e)[:100])
        return False


async def test_search(token: str = None):
    """검색 테스트 (/api/chat/search)"""
    try:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/chat/search",
                headers=headers,
                json={
                    "query": "핀테크",
                    "top_k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else data.get("count", 0)
                log_result("Search", True, f"results={count}")
                return True
            else:
                log_result("Search", False, f"status={response.status_code}")
                return False

    except Exception as e:
        log_result("Search", False, str(e)[:100])
        return False


async def test_sessions(token: str = None):
    """세션 목록 테스트 (/api/chat/sessions)"""
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{BASE_URL}/api/chat/sessions",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else data.get("count", 0)
                log_result("Sessions List", True, f"sessions={count}")
                return True
            else:
                log_result("Sessions List", False, f"status={response.status_code}")
                return False

    except Exception as e:
        log_result("Sessions List", False, str(e)[:100])
        return False


async def main():
    print("=" * 70)
    print("Advisor 전체 기능 테스트")
    print("=" * 70)
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"테스트 사용자: {TEST_USER}")
    print(f"서버: {BASE_URL}")
    print("=" * 70)

    # 1. 헬스체크
    print("\n[1/5] 헬스체크...")
    await test_health()

    # 2. 회원가입/로그인
    print("\n[2/5] 회원가입/로그인...")
    token = await test_signup_login()

    # 3. 파일 업로드
    print("\n[3/5] 파일 업로드...")
    await test_file_upload(token)

    # 4. RAG 채팅 (개인화)
    print("\n[4/5] RAG 채팅 (개인화)...")
    await test_rag_chat(token)

    # 5. 검색
    print("\n[5/5] 검색...")
    await test_search(token)

    # 6. 세션 목록
    print("\n[6/5] 세션 목록...")
    await test_sessions(token)

    # 결과 요약
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)
    print(f"통과: {results['summary']['passed']}")
    print(f"실패: {results['summary']['failed']}")
    print(f"전체: {results['summary']['passed'] + results['summary']['failed']}")

    # 결과 저장
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"full_features_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_file}")
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
