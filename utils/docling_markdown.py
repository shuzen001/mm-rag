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
    # return llm.invoke(messages).content.strip()
    return "這是一張圖片的摘要。"  # Placeholder for actual LLM response


def convert_file_to_markdown(path: str) -> str:
    """Convert a document to Markdown and insert image summaries in-place."""
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document  # Changed from result.legacy_document
    # print(f"doc object: {doc}")
    # print(f"type of doc: {type(doc)}")  
    # if hasattr(doc, 'pages'):
    #     print(f"doc.pages: {doc.pages}")
    #     print(f"type of doc.pages: {type(doc.pages)}")
    #     if doc.pages and isinstance(doc.pages, list) and len(doc.pages) > 0:
    #         print(f"First page element: {doc.pages[0]}")
    #         print(f"Type of first page element: {type(doc.pages[0])}")
    # else:
    #     print("doc object has no 'pages' attribute")
    placeholder = "<!-- image -->"
    markdown = doc.export_to_markdown(
        image_placeholder=placeholder
    )

    if placeholder not in markdown:
        return markdown

    # New logic to extract image paths from document pages
    image_paths: List[Path] = []
    if hasattr(doc, 'pages') and doc.pages:
        for page in doc.pages:
            if hasattr(page, 'images') and page.images:
                for image_file_info in page.images: # Assuming page.images contains objects/dicts with path info
                    # Attempt to get path, assuming image_file_info has a 'path' attribute or is the path itself
                    path_attr = getattr(image_file_info, 'path', None)
                    if path_attr:
                        image_paths.append(Path(path_attr))
                    elif isinstance(image_file_info, (str, Path)): # Fallback if image_file_info is a path string/object
                        image_paths.append(Path(image_file_info))
    
    images = image_paths # Use the collected image_paths

    for img_path in images:
        summary = _summarize_image(str(img_path))
        markdown = markdown.replace(placeholder, summary, 1)

    # remove any remaining placeholders
    # markdown = re.sub(r"<!-- image -->", "", markdown)
    return markdown
