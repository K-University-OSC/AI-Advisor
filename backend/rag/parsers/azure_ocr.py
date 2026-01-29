"""
Azure Document Intelligence OCR 모듈
이미지에서 텍스트와 테이블을 추출
"""

import os
import base64
import subprocess
from typing import Dict, Any, Optional


def _load_azure_key() -> str:
    """Azure Document Intelligence API 키 로드"""
    # bashrc에서 먼저 시도
    try:
        result = subprocess.run(
            ['bash', '-c', 'source ~/.bashrc && echo $AZURE_DOCUMENT_INTELLEGENCE_KEY'],
            capture_output=True, text=True
        )
        key = result.stdout.strip()
        if key:
            return key
    except:
        pass

    # 환경변수에서 로드
    return os.getenv("AZURE_DOCUMENT_INTELLEGENCE_KEY", "")


class AzureOCR:
    """
    Azure Document Intelligence OCR 클라이언트
    이미지에서 텍스트와 테이블을 추출
    """

    DEFAULT_ENDPOINT = "https://rag-di.cognitiveservices.azure.com/"

    def __init__(self, endpoint: str = None, key: str = None):
        """
        Args:
            endpoint: Azure Document Intelligence 엔드포인트
            key: API 키 (없으면 환경변수에서 로드)
        """
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.key = key or _load_azure_key()
        self.client = None
        self._initialized = False

    def initialize(self) -> bool:
        """클라이언트 초기화"""
        if self._initialized:
            return True

        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            if not self.key:
                print("Azure Document Intelligence API 키가 설정되지 않았습니다.")
                return False

            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            self._initialized = True
            print("Azure OCR 초기화 완료")
            return True

        except ImportError:
            print("azure-ai-documentintelligence 패키지를 설치하세요")
            return False
        except Exception as e:
            print(f"Azure OCR 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        이미지 OCR 분석

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            {'text': str, 'confidence': float, 'tables': list, 'word_count': int}
        """
        if not self._initialized:
            return {'error': 'Not initialized', 'text': '', 'confidence': 0.0, 'tables': []}

        try:
            # 이미지 크기 검증
            from PIL import Image
            import io

            try:
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size
                if width < 50 or height < 50:
                    return {'error': 'Image too small', 'text': '', 'confidence': 0.0, 'tables': []}
                if width > 10000 or height > 10000:
                    return {'error': 'Image too large', 'text': '', 'confidence': 0.0, 'tables': []}
            except:
                pass

            # OCR 실행 (레이아웃 모드로 테이블도 추출)
            img_base64 = base64.b64encode(image_bytes).decode()

            poller = self.client.begin_analyze_document(
                "prebuilt-layout",
                body={"base64Source": img_base64}
            )
            result = poller.result()

            full_text = result.content if result.content else ""
            confidence = self._get_average_confidence(result)

            # 테이블 추출
            tables = []
            if result.tables:
                for table in result.tables:
                    table_data = self._extract_table_data(table)
                    markdown = self._table_to_markdown(table_data)
                    tables.append({
                        'data': table_data,
                        'markdown': markdown
                    })

            return {
                'text': full_text,
                'confidence': confidence,
                'tables': tables,
                'word_count': len(full_text.split()) if full_text else 0
            }

        except Exception as e:
            return {'error': str(e)[:100], 'text': '', 'confidence': 0.0, 'tables': []}

    def _extract_table_data(self, table) -> Dict[str, Any]:
        """테이블 객체에서 데이터 추출"""
        table_data = {
            'row_count': table.row_count,
            'column_count': table.column_count,
            'cells': []
        }

        if table.cells:
            for cell in table.cells:
                table_data['cells'].append({
                    'row_index': cell.row_index,
                    'column_index': cell.column_index,
                    'content': cell.content,
                    'row_span': getattr(cell, 'row_span', 1),
                    'column_span': getattr(cell, 'column_span', 1)
                })

        return table_data

    def _table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """테이블 데이터를 마크다운으로 변환"""
        if not table_data.get('cells'):
            return ""

        rows = table_data['row_count']
        cols = table_data['column_count']

        # 그리드 생성
        grid = [["" for _ in range(cols)] for _ in range(rows)]

        # 셀 데이터 채우기
        for cell in table_data['cells']:
            r = cell['row_index']
            c = cell['column_index']
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = cell['content']

        # 마크다운 생성
        lines = []
        if grid:
            # 헤더
            lines.append("| " + " | ".join(grid[0]) + " |")
            # 구분선
            lines.append("| " + " | ".join(["---"] * cols) + " |")
            # 데이터 행
            for row in grid[1:]:
                lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _get_average_confidence(self, result) -> float:
        """OCR 결과의 평균 신뢰도 계산"""
        confidences = []
        if result.pages:
            for page in result.pages:
                if page.words:
                    for word in page.words:
                        if hasattr(word, 'confidence'):
                            confidences.append(word.confidence)
        return sum(confidences) / len(confidences) if confidences else 0.0
