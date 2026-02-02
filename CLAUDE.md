# Advisor Chatbot

대학교용 학사 정보용  챗봇 시스템

## 개선점 관련 주의 사항
- 정보가 더 필요하면 인터넷 검색해서 개선점을 찾아도 되..
- 편향적이지 않고 보편적인 방법이어야 해


## 주의사항
- API 키는 .env 파일에서만 관리
- 기본 비밀번호 하드코딩 금지
- 새 LLM/DB 등 필요 모듈 추가 시 기존 코드 수정하지 말고 Provider 추가
- Git 업로드는 사용자 요청시에만 commit , push 진행


## 코딩 규칙

### 기본 원칙
- 모든 기능은 모듈형으로 여기서도 붙였다 띠었다 쉽고 다른곳에서도 가져다 쓰기 쉽게 만들어야해
- 포트, db , 벡터 db, 소스코드등 개발에 필요한 모든것들을 다른것들과 격리가 되어 야해
-다른 프로젝트 db를 사용한다거나, 벡터 db등을 사용하면 안되 
- 작업이 오래 걸리는건 async/await 선호
- 에러는 반드시 try-catch로 처리
- 테스트 커버리지 80% 이상 유지

### 외부 의존성 처리 (Provider 패턴)
**모든 외부 서비스(LLM, DB, 검색엔진 등)는 Provider 패턴 + 의존성 주입으로 구현해야 함**

1. 인터페이스(Base 클래스) 먼저 정의
2. 각 구현체는 인터페이스를 상속
3. 서비스는 인터페이스에만 의존 (구체 클래스 직접 참조 금지)
4. 설정 파일(config.py)에서 사용할 Provider 선택

```python
# 나쁜 예 (직접 의존)
from openai import OpenAI
client = OpenAI(api_key="...")

# 좋은 예 (인터페이스 의존)
from providers.llm import get_llm_provider
provider = get_llm_provider()  # 설정에 따라 OpenAI/Claude/Gemini 반환
```

이렇게 하면 외부 서비스 교체 시 기존 코드 수정 없이 새 Provider만 추가하면 됨

### 설정 관리
- 모든 설정은 `backend/config.py`에서 중앙 관리
- 모델명, API 키, 서비스 설정 등 하드코딩 금지
- 환경변수는 `.env`에서 로드
- 이미 사용하고 있는 서비스의 포트를 정지 시키고 사용하면 안되
- 다른 서비스에 문제가 없도록 사용포트는 사용포트 정보.txt 에서 사용하는걸 사용하면 안되. 단 해당서비스가 쓰던건 그걸 사용해도 되
## 환경 격리 원칙

**개발(dev), 스테이징(staging), 운영(production) 환경은 모든 리소스가 완전히 격리되어야 함**

| 리소스 | 개발 | 운영 |
|--------|------|------|
| Backend Port | 10311 | - |
| Frontend Port | 10310 | - |
| PostgreSQL DB | advisor_osc_db | - |
| PostgreSQL Port | 10312 | - |
| Redis Port | 10313 | - |
| Qdrant Port | 10314 | - |

### 격리 원칙
1. **데이터베이스 격리**: 각 환경별 별도 DB 사용 (테넌트 DB 포함)
2. **캐시 격리**: Redis는 별도 포트/컨테이너 사용 (DB 번호만 다른 것은 불충분)
3. **벡터DB 격리**: Qdrant는 환경별 별도 컬렉션 사용
4. **설정 파일 분리**: 각 환경별 `.env` 파일로 관리
5. **포트 분리**: 개발과 운영은 다른 포트 사용

### 새 환경 추가 시 체크리스트
- [ ] PostgreSQL 데이터베이스 생성
- [ ] Redis 컨테이너 생성 (별도 포트)
- [ ] Qdrant 컬렉션명 설정
- [ ] .env 파일 설정
- [ ] 사용포트정보.txt 업데이트

## GPU 사용 규칙

**GPU가 필요한 작업 수행 시 다음 규칙을 따라야 함**

### 기본 원칙
1. **지정된 GPU 우선 사용**: 설정에서 지정된 GPU를 먼저 확인
2. **사용 불가 시 대체 GPU 탐색**: 지정 GPU가 사용 중이면 사용 가능한 다른 GPU 탐색
3. **GPU 상태 확인 명령어**: `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv`

### GPU 선택 로직
```python
import subprocess

def get_available_gpu(preferred_gpu: int = 0) -> int:
    """사용 가능한 GPU 반환. 지정 GPU 우선, 불가시 대체 GPU 탐색"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )

    gpus = []
    for line in result.stdout.strip().split('\n'):
        idx, used, total = map(float, line.split(','))
        available = total - used
        gpus.append((int(idx), available))

    # 지정 GPU가 사용 가능하면 우선 사용
    for idx, available in gpus:
        if idx == preferred_gpu and available > 2000:  # 2GB 이상 여유
            return preferred_gpu

    # 가장 여유로운 GPU 반환
    gpus.sort(key=lambda x: x[1], reverse=True)
    return gpus[0][0] if gpus else 0
```

### 사용 예시
```bash
# 환경변수로 GPU 지정
CUDA_VISIBLE_DEVICES=$(python -c "from gpu_utils import get_available_gpu; print(get_available_gpu(0))") python train.py
```

## 주요 디렉토리
- `backend/config.py` - 중앙 설정
- `backend/providers/` - Provider 구현체들
- `backend/services/` - 비즈니스 로직
- `docs/` - 문서

## RAG 개선 이력

### Issue #13: Smart Child Fallback 구현 (2025-12-28) ✅ 해결

**문제**: Multi-page 섹션에서 Parent chunk가 `parent_chunk_size=2000`으로 인해 truncated되어 후반부 페이지 정보 누락
- 예: Parent가 18-22페이지를 커버하지만, 실제 content는 18페이지만 포함 (1938자)
- Child는 21페이지의 변액연금 정보를 가지고 있으나 LLM에 전달되지 않음

**해결**: Smart Child Fallback 구현 (V7.7)
1. **Truncation 메타데이터 기록** (`hierarchical_chunker.py`)
   - Parent 생성 시 `is_truncated`, `original_length` 메타데이터 추가
2. **조건부 Child 보완** (`hierarchical_retriever.py`)
   - Parent가 truncated된 경우에만 Child 내용 추가
   - Child 내용이 Parent에 이미 포함되어 있으면 추가하지 않음 (중복 방지)

**V7.6.3/V7.6.4와의 차이점**:
- V7.6.3/V7.6.4: 무조건 Child 추가 → 중복 노이즈로 성능 하락
- V7.7: Truncated Parent에 대해서만 Child 보완 → 노이즈 최소화

**적용 범위**: 모든 문서 유형에 보편적으로 적용 가능

### Issue #11: 복합 질문 및 인과관계 추론 개선 (2025-12-28) ✅ 해결

**문제**: 복합 질문(A와 B를 동시에 묻는 질문)이나 인과관계/미래 영향 질문에 대한 답변 품질 부족
- Index 12: "후견제도 지원신탁 명칭과 출시 배경" - 통계 데이터 누락
- Index 37: "부동산 PF 대출 연체율의 미래 영향" - 인과관계 추론 부족

**해결**: 3가지 개선 적용
1. **검색 범위 확대** (`RetrievalConfig`)
   - `top_k`: 8 → 12, `rerank_top_k`: 25 → 30, `num_queries`: 3 → 4
2. **Query Decomposition** (`query_enhancer.py`)
   - 복합 질문 분해: "A와 B" → "A", "B" 별도 검색
   - 배경 질문 시 통계/사회적 상황 검색 추가
   - 영향 질문 시 인과관계/전망 검색 추가
3. **시스템 프롬프트 추론 지침** (`rag_chain.py`)
   - 복합 질문 처리: 각 부분 순서대로 답변
   - 배경/원인 질문: 통계, 사회적 상황, 정책적 맥락 포함
   - 인과관계 추론: 원인 → 결과 → 파급 영향 → 전망 순서
   - 미래 영향 질문: "예상", "전망", "우려" 표현 찾아 추론

**결과**: Pass Rate 93.3% → **96.7%** (+3.4%p)

### Issue #10: 검색(Retrieval) 범위 부족 문제 (2025-12-28) ✅ 해결

**문제**: 질문에 대한 정답이 여러 섹션에 분산되어 있을 때, 단일 쿼리로 모든 관련 청크를 검색하지 못함
- 예: "후견제도 지원신탁 출시 배경" → 법적 배경만 검색, 사회적 배경(통계) 누락

**해결**: Multi-Query Retrieval 구현
- 파일: `backend/rag/retriever/hierarchical_retriever.py`
- 원본 쿼리를 여러 관점으로 변환하여 검색 범위 확대
- 검색 결과 병합 후 BGE Reranker로 최종 순위 결정
- 설정: `RetrievalConfig.use_multi_query=True`, `num_queries=4`

### Issue #9: LLM 의미론적 평가 도입 (2025-12-28) ✅ 해결

**문제**: 키워드 기반 평가가 너무 엄격하여 의미적으로 동일한 답변도 FAIL 처리
- Pass Rate: 36.7% (키워드 평가)

**해결**: GPT-4o-mini 기반 의미론적 평가 도입
- 파일: `eval/allganize/test_finance_paragraph.py`
- 평가 기준: 정확성(40%), 완전성(30%), 숫자정확도(30%)
- 임계값: 70% 이상 PASS
- Pass Rate: 93.3% (LLM 평가)

### Issue #8: Parent-Child 컨텍스트 구축 문제 (2025-12-27) ✅ 해결

**문제**: Child 청크의 내용이 Parent에 포함되지 않는 경우 핵심 정보 누락

**해결**: `_build_context()` 메서드 개선
- Parent에 Child 내용이 없으면 Child도 추가로 포함
- 파일: `backend/rag/retriever/hierarchical_retriever.py:233-238`

### Issue #7: LLM 과잉 응답 문제 (2025-12-28) ✅ 해결

**문제**: "특정 국가"를 묻는 질문에 관련 없는 다른 국가 정보까지 포함

**해결**: 시스템 프롬프트 개선
- 파일: `backend/rag/chain/rag_chain.py`
- "질문 범위 엄수" 원칙 추가
- "특정 국가", "해당 기관" 등 질문에서 한정한 범위 내 정보만 답변하도록 지시

### 현재 성능 (Finance 도메인, Paragraph 유형)
| 지표 | 값 |
|------|-----|
| Pass Rate | 96.7% (29/30) |
| 평균 의미점수 | 93.8% |
| 평균 응답시간 | 9.6초 |
| 출처 일치율 | 93.3% |