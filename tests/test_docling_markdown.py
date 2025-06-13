import importlib
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub external dependencies before importing the module under test
stub_converter_module = types.ModuleType("docling.document_converter")
stub_converter_module.DocumentConverter = lambda: None
sys.modules["docling.document_converter"] = stub_converter_module

stub_base_module = types.ModuleType("docling_core.types.doc.base")
stub_base_module.ImageRefMode = object
sys.modules["docling_core.types.doc.base"] = stub_base_module

import utils.docling_markdown as dm


class DummyDoc:
    def export_to_markdown(self, image_placeholder="<!-- image -->"):
        return f"Intro {image_placeholder} Outro {image_placeholder}"

    def _list_images_on_disk(self):
        return [Path("img1.png"), Path("img2.png")]


class DummyConverter:
    def convert(self, path):
        return types.SimpleNamespace(document=DummyDoc())


def test_convert_file_to_markdown_replaces_placeholders(monkeypatch):
    monkeypatch.setattr(dm, "DocumentConverter", lambda: DummyConverter())
    summaries = ["summary1", "summary2"]

    def fake_summarize_image(image_path):
        return summaries.pop(0)

    monkeypatch.setattr(dm, "_summarize_image", fake_summarize_image)
    markdown = dm.convert_file_to_markdown("dummy.pdf")
    assert markdown == "Intro summary1 Outro summary2"
