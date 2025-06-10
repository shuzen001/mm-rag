"""Utilities for converting documents to Markdown with image summaries using docling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import ImageRefMode

from utils.summarize import encode_image


def _summarize_image(image_path: str) -> str:
    """Return a short summary for the given image using OpenAI."""
    from utils.LLM_Tool import gpt_4o_for_summary

    llm = gpt_4o_for_summary()
    img_b64 = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
                {"type": "text", "text": "以繁體中文簡短摘要這張圖片的內容。"},
            ],
        }
    ]
    return llm.invoke(messages).content.strip()


def convert_file_to_markdown(path: str) -> str:
    """Convert a document to Markdown and insert image summaries in-place."""
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.legacy_document

    placeholder = "<!-- image -->"
    markdown = doc.export_to_markdown(
        image_placeholder=placeholder, image_mode=ImageRefMode.PLACEHOLDER
    )

    images: List[Path] = doc._list_images_on_disk()
    for img_path in images:
        summary = _summarize_image(str(img_path))
        markdown = markdown.replace(placeholder, summary, 1)

    # remove any remaining placeholders
    markdown = re.sub(r"<!-- image -->", "", markdown)
    return markdown
