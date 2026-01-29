# -*- coding: utf-8 -*-
"""
PDF 하이라이트 모듈

PDF 문서에서 키워드를 검색하고 하이라이트 처리
- 키워드 추출 (불용어/조사 제거)
- PDF 페이지 추출
- 텍스트 하이라이팅
"""

import re
import urllib.parse
from io import BytesIO
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class HighlightConfig:
    """하이라이트 설정"""
    border_color: tuple = (1, 0.8, 0)     # 주황색 테두리
    fill_color: tuple = (1, 1, 0.7)       # 연한 노란색 배경
    border_width: float = 2
    fill_opacity: float = 0.3
    padding_x: float = 20                  # 좌우 여백
    padding_y: float = 15                  # 상하 여백
    use_full_width: bool = True            # 페이지 전체 너비 사용


class KeywordExtractor:
    """질문에서 핵심 키워드 추출"""

    # 한국어 불용어
    STOPWORDS = {
        '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
        '와', '과', '도', '만', '까지', '부터', '대해', '대한', '관한',
        '어떤', '무엇', '어디', '언제', '어떻게', '왜', '그', '저',
        '이런', '그런', '있는', '있다', '없는', '없다', '하는', '되는',
        '된', '할', '될', '것', '수', '등', '및', '또는', '그리고',
        '하지만', '그러나', '때문', '위해', '통해', '따라', '관련',
        '경우', '점', '중', '후', '전', '간', '내', '외'
    }

    # 조사 패턴 (단어 끝에 붙는 조사 제거)
    SUFFIX_PATTERN = re.compile(
        r'(에서|으로|에게|한테|께서|에는|에도|와는|과는|'
        r'은|는|이|가|을|를|의|에|로|와|과|도|만|서|라|다|나|며|면|고|지|든|게)$'
    )

    def __init__(self, min_length: int = 2, max_keywords: int = 8):
        """
        Args:
            min_length: 최소 키워드 길이
            max_keywords: 최대 키워드 개수
        """
        self.min_length = min_length
        self.max_keywords = max_keywords

    def extract(self, text: str) -> list[str]:
        """
        텍스트에서 키워드 추출

        Args:
            text: 원본 텍스트 (질문)

        Returns:
            추출된 키워드 리스트
        """
        # 구두점 제거
        cleaned = text.replace('?', ' ').replace(',', ' ').replace('.', ' ')

        keywords = []
        for word in cleaned.split():
            # 조사 제거
            clean_word = self.SUFFIX_PATTERN.sub('', word)
            # 길이 체크 및 불용어 필터링
            if len(clean_word) >= self.min_length and clean_word not in self.STOPWORDS:
                keywords.append(clean_word)

        # 중복 제거 및 개수 제한
        return list(dict.fromkeys(keywords))[:self.max_keywords]

    def to_pipe_separated(self, keywords: list[str]) -> str:
        """키워드를 파이프(|)로 구분된 문자열로 변환"""
        return '|'.join(keywords)

    def from_pipe_separated(self, text: str) -> list[str]:
        """파이프(|)로 구분된 문자열에서 키워드 추출"""
        return [kw.strip() for kw in text.split('|') if kw.strip()]


class PDFHighlighter:
    """PDF 하이라이트 처리기"""

    def __init__(self, config: Optional[HighlightConfig] = None):
        """
        Args:
            config: 하이라이트 설정 (None이면 기본값 사용)
        """
        self.config = config or HighlightConfig()

    def highlight_keywords(
        self,
        pdf_page,  # fitz.Page
        keywords: list[str]
    ) -> list:
        """
        PDF 페이지에서 키워드를 찾아 하이라이트

        Args:
            pdf_page: PyMuPDF 페이지 객체
            keywords: 하이라이트할 키워드 리스트

        Returns:
            발견된 영역 리스트 (fitz.Rect)
        """
        import fitz

        all_rects = []

        # 각 키워드 검색
        for keyword in keywords:
            text_instances = pdf_page.search_for(keyword)
            all_rects.extend(text_instances)

        if not all_rects:
            return []

        # 바운딩 박스 계산
        min_x = min(r.x0 for r in all_rects)
        min_y = min(r.y0 for r in all_rects)
        max_x = max(r.x1 for r in all_rects)
        max_y = max(r.y1 for r in all_rects)

        # 하이라이트 영역 결정
        if self.config.use_full_width:
            page_width = pdf_page.rect.width
            rect = fitz.Rect(
                self.config.padding_x,
                max(0, min_y - self.config.padding_y),
                page_width - self.config.padding_x,
                min(pdf_page.rect.height, max_y + self.config.padding_y)
            )
        else:
            rect = fitz.Rect(
                max(0, min_x - self.config.padding_x),
                max(0, min_y - self.config.padding_y),
                min(pdf_page.rect.width, max_x + self.config.padding_x),
                min(pdf_page.rect.height, max_y + self.config.padding_y)
            )

        # 하이라이트 그리기
        shape = pdf_page.new_shape()
        shape.draw_rect(rect)
        shape.finish(
            color=self.config.border_color,
            fill=self.config.fill_color,
            width=self.config.border_width,
            fill_opacity=self.config.fill_opacity
        )
        shape.commit()

        return all_rects


class PDFPageExtractor:
    """PDF 페이지 추출 및 하이라이트 처리"""

    def __init__(
        self,
        highlighter: Optional[PDFHighlighter] = None,
        keyword_extractor: Optional[KeywordExtractor] = None
    ):
        """
        Args:
            highlighter: PDF 하이라이터 (None이면 기본값 사용)
            keyword_extractor: 키워드 추출기 (None이면 기본값 사용)
        """
        self.highlighter = highlighter or PDFHighlighter()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()

    def extract_pages(
        self,
        pdf_path: str,
        center_page: int,
        range_pages: int = 3,
        highlight_text: Optional[str] = None
    ) -> tuple[BytesIO, dict]:
        """
        PDF에서 특정 페이지 범위 추출 및 하이라이트

        Args:
            pdf_path: PDF 파일 경로
            center_page: 중심 페이지 번호 (1-based)
            range_pages: 추출할 총 페이지 수
            highlight_text: 하이라이트할 텍스트 (파이프 구분 또는 일반 문자열)

        Returns:
            (PDF 바이트 스트림, 메타정보 딕셔너리)
        """
        import fitz

        # PDF 열기
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # 페이지 범위 계산 (1-based → 0-based)
        half_range = range_pages // 2
        start_page = max(0, center_page - 1 - half_range)
        end_page = min(total_pages, center_page - 1 + half_range + 1)

        # 새 PDF 문서 생성
        new_doc = fitz.open()

        for i in range(start_page, end_page):
            new_doc.insert_pdf(doc, from_page=i, to_page=i)

        # 하이라이트 처리
        highlighted_count = 0
        if highlight_text:
            # URL 디코딩
            decoded_text = urllib.parse.unquote(highlight_text)

            # 키워드 파싱 (파이프 구분 또는 직접 추출)
            if '|' in decoded_text:
                keywords = self.keyword_extractor.from_pipe_separated(decoded_text)
            else:
                keywords = self.keyword_extractor.extract(decoded_text)

            logger.info(f"PDF Highlighting keywords: {keywords}")

            # 각 페이지에 하이라이트 적용
            for page_idx in range(len(new_doc)):
                pdf_page = new_doc[page_idx]
                found_rects = self.highlighter.highlight_keywords(pdf_page, keywords)
                if found_rects:
                    highlighted_count += 1

        # 메모리에 저장
        output = BytesIO()
        new_doc.save(output)
        output.seek(0)

        # 정리
        new_doc.close()
        doc.close()

        # 메타정보
        metadata = {
            "total_pages": total_pages,
            "start_page": start_page + 1,  # 1-based
            "end_page": end_page,          # 1-based
            "requested_page": center_page,
            "highlighted_pages": highlighted_count
        }

        return output, metadata

    def extract_and_highlight_from_query(
        self,
        pdf_path: str,
        page: int,
        query: str,
        range_pages: int = 3
    ) -> tuple[BytesIO, dict]:
        """
        질문에서 키워드를 추출하여 PDF 하이라이트

        Args:
            pdf_path: PDF 파일 경로
            page: 중심 페이지 번호 (1-based)
            query: 원본 질문 (키워드 추출 대상)
            range_pages: 추출할 총 페이지 수

        Returns:
            (PDF 바이트 스트림, 메타정보 딕셔너리)
        """
        # 질문에서 키워드 추출
        keywords = self.keyword_extractor.extract(query)
        keywords_str = self.keyword_extractor.to_pipe_separated(keywords)

        return self.extract_pages(
            pdf_path=pdf_path,
            center_page=page,
            range_pages=range_pages,
            highlight_text=keywords_str
        )


# 편의 함수들
def extract_keywords_from_query(query: str, max_keywords: int = 8) -> list[str]:
    """질문에서 키워드 추출 (편의 함수)"""
    extractor = KeywordExtractor(max_keywords=max_keywords)
    return extractor.extract(query)


def get_keywords_string(query: str, max_keywords: int = 8) -> str:
    """질문에서 키워드 추출하여 파이프 구분 문자열 반환"""
    extractor = KeywordExtractor(max_keywords=max_keywords)
    keywords = extractor.extract(query)
    return extractor.to_pipe_separated(keywords)
