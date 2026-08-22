from __future__ import annotations

from pathlib import Path

import pytest

from app.services.pdf_pipeline import PdfPage, PdfPipeline, PdfRegion


class FakeExtractor:
    def extract(self, path: Path) -> list[PdfPage]:
        return [PdfPage(1, "", b"page-1"), PdfPage(2, "electronic text", b"page-2")]


class FakeLayout:
    def analyze(self, page: PdfPage) -> list[PdfRegion]:
        if page.number == 1:
            return [
                PdfRegion("image", 1, page=1, bbox=None),
                PdfRegion("table", 2, page=1, bbox=None),
                PdfRegion("text", 3, "OCR text", page=1, bbox=None),
            ]
        return [PdfRegion("text", 0, page.text, page=2)]


class FakeOcr:
    def text(self, image: bytes) -> str:
        return "OCR fallback"


class FakeTable:
    def parse(self, image: bytes, payload=None) -> str:
        return "| a | b |\n| - | - |\n| 1 | 2 |"


class FakeVision:
    async def describe(self, image: bytes) -> str:
        return "示例图片"


class FakeAssets:
    def put(self, image: bytes, key: str) -> str:
        return f"https://oss.test/{key}"


@pytest.mark.asyncio
async def test_pdf_pipeline_keeps_region_order_and_asset_markdown():
    result = await PdfPipeline(
        extractor=FakeExtractor(), layout=FakeLayout(), ocr=FakeOcr(), table=FakeTable(), vision=FakeVision(), assets=FakeAssets()
    ).parse(Path("demo.pdf"))

    assert [item["page"] for item in result] == [1, 1, 1, 2]
    assert "![示例图片](https://oss.test/" in str(result[0]["text"])
    assert "| a | b |" in str(result[1]["text"])
    assert result[2]["text"] == "OCR text"


@pytest.mark.asyncio
async def test_text_only_pdf_does_not_require_layout_adapter():
    class TextExtractor:
        def extract(self, path: Path) -> list[PdfPage]:
            return [PdfPage(1, "hello", b"page")]

    result = await PdfPipeline(extractor=TextExtractor()).parse(Path("demo.pdf"))
    assert result == [{"text": "hello", "page": 1, "order": 0}]
