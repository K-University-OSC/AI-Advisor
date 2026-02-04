# -*- coding: utf-8 -*-
"""
문서 파서 팩토리

여러 파서 옵션 지원:
- gpt4o: GPT-4o Vision OCR (기본값, 고품질)
- gemini: Gemini 3 Flash OCR (고품질)
- docling: Docling (IBM 오픈소스, 레이아웃 분석 특화)
- pymupdf: PyMuPDF4LLM (무료, 빠름)
"""

import os
from typing import Optional, Literal

from .document_parser import DocumentParser


ParserType = Literal["gemini", "gpt4o", "docling", "pymupdf"]


def get_document_parser(
    parser_type: Optional[ParserType] = None,
    **kwargs
) -> DocumentParser:
    """
    문서 파서 인스턴스 생성

    Args:
        parser_type: 파서 타입 (기본값: 환경변수 또는 gpt4o)
            - "gpt4o": GPT-4o Vision OCR (고품질, v1.6.8 최고 성능)
            - "gemini": Gemini 3 Flash OCR (고품질, OCR Arena 1위)
            - "docling": Docling (IBM 오픈소스, 레이아웃 분석 특화, 무료)
            - "pymupdf": PyMuPDF4LLM (무료, 빠름, 마크다운 지원)
        **kwargs: 파서별 추가 설정

    Returns:
        DocumentParser: 문서 파서 인스턴스

    Examples:
        >>> parser = get_document_parser()  # 기본값 (gpt4o)
        >>> parser = get_document_parser("pymupdf")  # PyMuPDF 사용
        >>> parser = get_document_parser("gemini", model="gemini-3-flash-preview")
    """
    # 환경변수에서 기본값 읽기
    if parser_type is None:
        parser_type = os.getenv("DOCUMENT_PARSER", "gpt4o").lower()

    if parser_type == "gpt4o":
        from .gpt4o_ocr_parser import GPT4oOCRParser
        return GPT4oOCRParser(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "gpt-4o"),
            dpi=kwargs.get("dpi", 150),
            ocr_concurrency=kwargs.get("ocr_concurrency", 5),
        )

    elif parser_type == "gemini":
        from .gemini_ocr_parser import GeminiOCRParser
        return GeminiOCRParser(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "gemini-3-flash-preview"),
            extract_tables=kwargs.get("extract_tables", True),
            extract_images=kwargs.get("extract_images", True),
            output_format=kwargs.get("output_format", "markdown"),
        )

    elif parser_type == "docling":
        from .docling_parser import DoclingParser
        return DoclingParser(
            extract_tables=kwargs.get("extract_tables", True),
            extract_images=kwargs.get("extract_images", True),
            ocr_enabled=kwargs.get("ocr_enabled", True),
            output_format=kwargs.get("output_format", "markdown"),
        )

    elif parser_type == "pymupdf":
        from .pymupdf_parser import PyMuPDFParser
        return PyMuPDFParser(
            extract_images=kwargs.get("extract_images", True),
            extract_tables=kwargs.get("extract_tables", True),
            page_chunks=kwargs.get("page_chunks", False),
            write_images=kwargs.get("write_images", False),
            image_path=kwargs.get("image_path"),
        )

    else:
        raise ValueError(
            f"지원하지 않는 파서 타입: {parser_type}. "
            f"지원 타입: gpt4o, gemini, docling, pymupdf"
        )


# 파서 정보
PARSER_INFO = {
    "gpt4o": {
        "name": "GPT-4o Vision OCR",
        "description": "OpenAI GPT-4o Vision 기반 고품질 OCR, v1.6.8 최고 성능",
        "cost": "유료 ($5/1M 입력 토큰, 약 $0.05/페이지)",
        "speed": "보통 (1-2s, 병렬 처리)",
        "features": ["고정밀 OCR", "마크다운", "테이블", "이미지 설명", "병렬 처리"],
        "accuracy": "finance 테스트 73.3%, image 70%",
    },
    "gemini": {
        "name": "Gemini 3 Flash OCR",
        "description": "Google Gemini 3 Flash 기반 고품질 OCR, OCR Arena 1위",
        "cost": "유료 ($0.50/1M 토큰, 약 $0.01/페이지)",
        "speed": "빠름 (0.5-1s)",
        "features": ["고정밀 OCR", "마크다운", "테이블", "이미지 분석", "손글씨"],
        "accuracy": "인쇄물 95%+, 손글씨 85%+",
    },
    "docling": {
        "name": "Docling (IBM)",
        "description": "IBM 오픈소스 문서 분석, DocLayNet 기반 레이아웃 분석",
        "cost": "무료 (MIT 라이선스)",
        "speed": "보통 (1-3s)",
        "features": ["레이아웃 분석", "테이블", "이미지", "OCR", "마크다운", "다중 포맷"],
        "supported_formats": ["PDF", "DOCX", "PPTX", "XLSX", "HTML", "Images"],
        "github_stars": "30K+",
    },
    "pymupdf": {
        "name": "PyMuPDF4LLM",
        "description": "빠르고 정확한 무료 PDF 파서, RAG에 최적화된 마크다운 출력",
        "cost": "무료",
        "speed": "매우 빠름 (0.12s)",
        "features": ["마크다운", "테이블", "이미지", "기본 OCR"],
    },
}


def list_available_parsers() -> dict:
    """사용 가능한 파서 목록 반환"""
    return PARSER_INFO
