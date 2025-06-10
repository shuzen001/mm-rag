import os
import sys
import types
from pathlib import Path

import pytest


def ensure_dummy_docling():
    if "docling" not in sys.modules:
        docling = types.ModuleType("docling")
        dc_mod = types.ModuleType("docling.document_converter")

        class DummyConverter:
            def convert(self, path):
                pass

        dc_mod.DocumentConverter = DummyConverter
        docling.document_converter = dc_mod
        sys.modules["docling"] = docling
        sys.modules["docling.document_converter"] = dc_mod

    if "docling_core" not in sys.modules:
        docling_core = types.ModuleType("docling_core")
        types_mod = types.ModuleType("docling_core.types")
        doc_mod = types.ModuleType("docling_core.types.doc")
        base_mod = types.ModuleType("docling_core.types.doc.base")

        class ImageRefMode:
            PLACEHOLDER = "placeholder"

        base_mod.ImageRefMode = ImageRefMode
        doc_mod.base = base_mod
        types_mod.doc = doc_mod
        docling_core.types = types_mod
        sys.modules["docling_core"] = docling_core
        sys.modules["docling_core.types"] = types_mod
        sys.modules["docling_core.types.doc"] = doc_mod
        sys.modules["docling_core.types.doc.base"] = base_mod


class DummyDoc:
    def __init__(self, markdown: str, images: list[str]):
        self._markdown = markdown
        self._images = [Path(p) for p in images]

    def export_to_markdown(self, **kwargs):
        return self._markdown

    def _list_images_on_disk(self):
        return self._images


class DummyResult:
    def __init__(self, doc):
        self.legacy_document = doc


def test_convert_file_to_markdown(monkeypatch, tmp_path):
    ensure_dummy_docling()
    from utils import docling_markdown

    dummy_doc = DummyDoc("Text <!-- image --> end", [tmp_path / "img.png"])
    dummy_res = DummyResult(dummy_doc)

    def fake_convert(self, path):
        return dummy_res

    monkeypatch.setattr(docling_markdown.DocumentConverter, "convert", fake_convert)
    monkeypatch.setattr(docling_markdown, "_summarize_image", lambda p: "summary")

    result = docling_markdown.convert_file_to_markdown("dummy")
    assert "summary" in result


def test_convert_file_to_markdown_real_pdf(monkeypatch):
    """Integration test using a user-supplied PDF via the DOC_TEST_PDF env var."""
    pytest.importorskip("docling")
    import importlib

    module = importlib.import_module("docling_core")
    spec = getattr(module, "__spec__", None)
    if spec is None or spec.origin is None:
        pytest.skip("docling_core not installed")

    from utils import docling_markdown

    pdf_path_env = os.environ.get("DOC_TEST_PDF")
    if not pdf_path_env:
        pytest.skip("DOC_TEST_PDF not set")

    pdf_path = Path(pdf_path_env)
    monkeypatch.setattr(docling_markdown, "_summarize_image", lambda p: "summary")
    result = docling_markdown.convert_file_to_markdown(str(pdf_path))
    assert "summary" in result
