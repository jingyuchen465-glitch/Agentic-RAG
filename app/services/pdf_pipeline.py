from __future__ import annotations

import asyncio
import base64
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.core.config import get_settings
from app.core.errors import AppError


@dataclass(slots=True)
class PdfPage:
    number: int
    text: str
    image: bytes
    width: int = 0
    height: int = 0
    has_images: bool = False


@dataclass(slots=True)
class PdfRegion:
    kind: str
    order: int
    text: str = ""
    image: bytes | None = None
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    payload: Any = None


class PageExtractor(Protocol):
    def extract(self, path: Path) -> list[PdfPage]: ...


class LayoutAnalyzer(Protocol):
    def analyze(self, page: PdfPage) -> list[PdfRegion]: ...


class OcrEngine(Protocol):
    def text(self, image: bytes) -> str: ...


class TableParser(Protocol):
    def parse(self, image: bytes, payload: Any = None) -> str: ...


class VisionDescriber(Protocol):
    async def describe(self, image: bytes) -> str: ...


class AssetStore(Protocol):
    def put(self, image: bytes, key: str) -> str: ...


class PyMuPdfExtractor:
    """Extract page text and a PNG rendering without importing PyMuPDF at startup."""

    def __init__(self, dpi: int = 144):
        self.dpi = dpi

    def extract(self, path: Path) -> list[PdfPage]:
        try:
            import fitz
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install pymupdf to ingest PDF files.", 503) from exc
        pages: list[PdfPage] = []
        try:
            document = fitz.open(str(path))
            scale = self.dpi / 72
            matrix = fitz.Matrix(scale, scale)
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pages.append(
                    PdfPage(
                        number=index + 1,
                        text=page.get_text("text").strip(),
                        image=pixmap.tobytes("png"),
                        width=pixmap.width,
                        height=pixmap.height,
                        has_images=bool(page.get_images(full=True)),
                    )
                )
            document.close()
            return pages
        except AppError:
            raise
        except Exception as exc:
            raise AppError("PDF_PARSE_FAILED", f"Unable to read PDF: {exc}", 422) from exc


class PaddleLayoutAnalyzer:
    """Adapter for PaddleOCR/PP-Structure; output normalization tolerates API versions."""

    def __init__(self, language: str = "ch"):
        try:
            import paddleocr
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install paddleocr to analyze PDF layout.", 503) from exc
        self._engine = None
        for name in ("PPStructureV3", "PPStructure"):
            factory = getattr(paddleocr, name, None)
            if factory:
                try:
                    self._engine = factory(lang=language)
                except TypeError:
                    self._engine = factory()
                break
        if self._engine is None:
            raise AppError("DEPENDENCY_MISSING", "PaddleOCR layout model is unavailable.", 503)

    def analyze(self, page: PdfPage) -> list[PdfRegion]:
        result = self._engine.predict(page.image) if hasattr(self._engine, "predict") else self._engine(page.image)
        if hasattr(result, "__iter__") and not isinstance(result, (dict, str, bytes)):
            result = next(iter(result), [])
        items = result.get("res", result) if isinstance(result, dict) else result
        if not isinstance(items, list):
            items = []
        regions: list[PdfRegion] = []
        for order, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", item.get("label", "text"))).lower()
            kind = "table" if "table" in kind else "image" if kind in {"image", "figure", "img"} else "text"
            box = item.get("bbox") or item.get("box")
            bbox = tuple(float(v) for v in box[:4]) if isinstance(box, (list, tuple)) and len(box) >= 4 else None
            payload = item.get("table_res") or item.get("table") or item.get("res")
            regions.append(PdfRegion(kind=kind, order=order, text=str(item.get("text", "")), page=page.number, bbox=bbox, payload=payload))
        return regions or [PdfRegion(kind="text", order=0, text=page.text, page=page.number)]


class PaddleOcrEngine:
    def __init__(self, language: str = "ch"):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install paddleocr to OCR scanned PDF pages.", 503) from exc
        try:
            self._engine = PaddleOCR(lang=language)
        except TypeError:
            self._engine = PaddleOCR()

    def text(self, image: bytes) -> str:
        result = self._engine.predict(image) if hasattr(self._engine, "predict") else self._engine.ocr(image)
        values: list[str] = []
        def walk(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, dict):
                for key in ("text", "rec_texts"):
                    if key in value:
                        walk(value[key])
            elif isinstance(value, (list, tuple)):
                for child in value:
                    walk(child)
        walk(result)
        return "\n".join(dict.fromkeys(values))


class PaddleTableParser:
    def __init__(self):
        self._engine = None

    def parse(self, image: bytes, payload: Any = None) -> str:
        if isinstance(payload, dict):
            for key in ("markdown", "html", "text"):
                if payload.get(key):
                    return str(payload[key])
            for key in ("table_html", "html"):  # Paddle 3 result variants
                value = payload.get(key)
                if value:
                    return str(value)
        if self._engine is None:
            try:
                import paddleocr
                factory = getattr(paddleocr, "TableRecognitionPipeline", None) or getattr(paddleocr, "TableSystem", None)
                self._engine = factory() if factory else False
            except ImportError as exc:
                raise AppError("DEPENDENCY_MISSING", "Install paddleocr to parse PDF tables.", 503) from exc
        if self._engine:
            result = self._engine.predict(image) if hasattr(self._engine, "predict") else self._engine(image)
            if hasattr(result, "__iter__") and not isinstance(result, (dict, str, bytes)):
                result = next(iter(result), {})
            if isinstance(result, dict):
                for key in ("markdown", "html", "table_html", "text"):
                    if result.get(key):
                        return str(result[key])
        raise AppError("TABLE_PARSER_UNAVAILABLE", "PaddleOCR table parser is unavailable.", 503)


class OpenAIVisionDescriber:
    def __init__(self):
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from app.services.llm import _get_chat_client
            self._client = _get_chat_client()
        return self._client

    async def describe(self, image: bytes) -> str:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AppError("MODEL_NOT_CONFIGURED", "OPENAI_API_KEY is required for PDF image descriptions.", 503)
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install langchain-openai for PDF image descriptions.", 503) from exc
        encoded = base64.b64encode(image).decode("ascii")
        response = await self._ensure_client().ainvoke([HumanMessage(content=[
            {"type": "text", "text": "请简洁描述这张图片的关键信息，用于知识库检索。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ])])
        return response.content if isinstance(response.content, str) else str(response.content)


class AliyunOssAssetStore:
    def __init__(self):
        settings = get_settings()
        required = [settings.oss_access_key_id, settings.oss_access_key_secret, settings.oss_bucket, settings.oss_endpoint, settings.oss_public_base_url]
        if not all(required):
            raise AppError("OSS_NOT_CONFIGURED", "OSS settings are required for PDF image assets.", 503)
        try:
            import oss2
        except ImportError as exc:
            raise AppError("DEPENDENCY_MISSING", "Install oss2 to store PDF image assets.", 503) from exc
        self._base = settings.oss_public_base_url.rstrip("/")
        self._bucket = oss2.Bucket(oss2.Auth(settings.oss_access_key_id, settings.oss_access_key_secret), settings.oss_endpoint, settings.oss_bucket)

    def put(self, image: bytes, key: str) -> str:
        result = self._bucket.put_object(key, image)
        if not result.status < 300:
            raise AppError("OSS_UPLOAD_FAILED", f"OSS upload failed with status {result.status}.", 502)
        return f"{self._base}/{key}"


def _crop(page: PdfPage, bbox: tuple[float, float, float, float] | None) -> bytes:
    if not bbox:
        return page.image
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(page.image))
        box = tuple(max(0, int(v)) for v in bbox)
        output = io.BytesIO()
        image.crop(box).save(output, format="PNG")
        return output.getvalue()
    except ImportError:
        return page.image


class PdfPipeline:
    def __init__(self, extractor: PageExtractor | None = None, layout: LayoutAnalyzer | None = None, ocr: OcrEngine | None = None, table: TableParser | None = None, vision: VisionDescriber | None = None, assets: AssetStore | None = None):
        settings = get_settings()
        self.extractor = extractor or PyMuPdfExtractor(int(getattr(settings, "pdf_render_dpi", 144)))
        self.layout = layout
        self.ocr = ocr
        self.table = table
        self.vision = vision
        self.assets = assets
        self.language = getattr(settings, "pdf_ocr_language", "ch")
        self._limit = asyncio.Semaphore(int(getattr(settings, "pdf_concurrency", 4)))
        self._layout_adapter = layout
        self._ocr_adapter = ocr
        self._table_adapter = table
        self._vision_adapter = vision
        self._assets_adapter = assets

    def _layout(self) -> LayoutAnalyzer:
        if self._layout_adapter is None:
            self._layout_adapter = PaddleLayoutAnalyzer(self.language)
        return self._layout_adapter

    def _ocr(self) -> OcrEngine:
        if self._ocr_adapter is None:
            self._ocr_adapter = PaddleOcrEngine(self.language)
        return self._ocr_adapter

    def _table(self) -> TableParser:
        if self._table_adapter is None:
            self._table_adapter = PaddleTableParser()
        return self._table_adapter

    def _vision(self) -> VisionDescriber:
        if self._vision_adapter is None:
            self._vision_adapter = OpenAIVisionDescriber()
        return self._vision_adapter

    def _assets(self) -> AssetStore:
        if self._assets_adapter is None:
            self._assets_adapter = AliyunOssAssetStore()
        return self._assets_adapter

    async def _region(self, page: PdfPage, region: PdfRegion, document_key: str) -> PdfRegion:
        async with self._limit:
            image = _crop(page, region.bbox)
            if region.kind == "text":
                if region.text.strip():
                    return region
                region.text = await asyncio.to_thread(self._ocr().text, image)
            elif region.kind == "table":
                region.text = await asyncio.to_thread(self._table().parse, image, region.payload or region.text)
            elif region.kind == "image":
                description = await self._vision().describe(image)
                url = await asyncio.to_thread(self._assets().put, image, f"pdf/{document_key}/{uuid.uuid4()}.png")
                region.text = f"![{description}]({url})\n{description}"
            return region

    async def parse(self, path: Path) -> list[dict[str, object]]:
        pages = await asyncio.to_thread(self.extractor.extract, path)
        output: list[dict[str, object]] = []
        document_key = path.stem
        # 纯电子文本 PDF 直接保留 PyMuPDF 文本，不强制加载 Paddle 模型。
        if pages and all(page.text.strip() and not page.has_images for page in pages) and self.layout is None:
            return [{"text": page.text.strip(), "page": page.number, "order": 0} for page in pages]
        layout = self._layout()
        for page in pages:
            regions = await asyncio.to_thread(layout.analyze, page)
            if not page.text.strip() and not regions:
                regions = [PdfRegion(kind="text", order=0, page=page.number)]
            if page.text.strip() and len(regions) == 1 and regions[0].kind == "text" and not regions[0].text.strip():
                regions[0].text = page.text
            processed = await asyncio.gather(*(self._region(page, region, document_key) for region in regions))
            for region in sorted(processed, key=lambda item: item.order):
                if region.text.strip():
                    output.append({"text": region.text.strip(), "page": page.number, "order": region.order})
        return output
