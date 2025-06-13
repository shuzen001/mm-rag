"""Shared processing helpers."""

from __future__ import annotations

import os
import time
from typing import List, Tuple

from unstructured.partition.docx import partition_docx

from .pdf_processing import convert_pdf_to_page_images, extract_pdf_elements
from .pptx_processing import (
    convert_pptx_to_pdf,
    convert_pptx_to_slide_images,
    extract_pptx_elements,
)

__all__ = ["categorize_elements", "process_single_file"]


def categorize_elements(raw_elements: List) -> Tuple[List[str], List[str]]:
    """Categorize unstructured elements into texts and tables."""
    tables: List[str] = []
    texts: List[str] = []
    for element in raw_elements:
        element_type = str(type(element))
        if "unstructured.documents.elements.Table" in element_type:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in element_type:
            texts.append(str(element))
        elif "unstructured.documents.elements.Title" in element_type:
            texts.append(str(element))
        elif "unstructured.documents.elements.NarrativeText" in element_type:
            texts.append(str(element))
    return texts, tables


def process_single_file(
    fname: str, fpath: str, database_dir: str = "./database"
) -> dict:
    """Process a single file and return extracted information."""
    file_start_time = time.time()
    print(f"\n📄 處理文件: {fname}")
    result = {
        "texts": [],
        "tables": [],
        "page_summaries": [],
        "page_identifiers": [],
        "temp_files": [],
        "file_type": None,
        "processing_time": 0,
    }
    try:
        if fname.lower().endswith(".pdf"):
            result["file_type"] = "pdf"
            extract_start = time.time()
            print(f"  🔄 開始提取 PDF 元素: {fname}")
            raw_elements = extract_pdf_elements(fpath, fname, database_dir=database_dir)
            texts, tables = categorize_elements(raw_elements)
            result["texts"] = texts
            result["tables"] = tables
            print(f"  ✓ 提取元素: {time.time() - extract_start:.2f} 秒")
            page_conversion_start = time.time()
            page_image_paths = convert_pdf_to_page_images(
                fpath, fname, database_dir=database_dir
            )
            print(f"  ✓ 頁面轉換: {time.time() - page_conversion_start:.2f} 秒")
            if page_image_paths:
                from utils.summarize import generate_pdf_page_summaries

                summary_start = time.time()
                page_summaries, page_identifiers = generate_pdf_page_summaries(
                    fpath, fname, page_image_paths
                )
                result["page_summaries"] = page_summaries
                result["page_identifiers"] = page_identifiers
                print(f"  ✓ 頁面摘要生成: {time.time() - summary_start:.2f} 秒")
        elif fname.lower().endswith(".pptx") or fname.lower().endswith(".ppt"):
            result["file_type"] = "pptx"
            pptx_start = time.time()
            raw_elements = extract_pptx_elements(
                fpath, fname, database_dir=database_dir
            )
            texts, tables = categorize_elements(raw_elements)
            result["texts"] = texts
            result["tables"] = tables
            print(f"  ✓ PPTX 元素提取: {time.time() - pptx_start:.2f} 秒")
            conversion_start = time.time()
            pdf_path = convert_pptx_to_pdf(fpath, fname)
            if pdf_path:
                result["temp_files"].append(pdf_path)
                pdf_filename = os.path.basename(pdf_path)
                pdf_dir = os.path.dirname(pdf_path)
                print(f"  ✓ PPTX 轉 PDF: {time.time() - conversion_start:.2f} 秒")
                page_conversion_start = time.time()
                page_image_paths = convert_pdf_to_page_images(
                    pdf_dir, pdf_filename, database_dir=database_dir
                )
                print(f"  ✓ PDF 頁面轉換: {time.time() - page_conversion_start:.2f} 秒")
                if page_image_paths:
                    from utils.summarize import generate_pdf_page_summaries

                    summary_start = time.time()
                    page_summaries, page_identifiers = generate_pdf_page_summaries(
                        pdf_dir, fname, page_image_paths
                    )
                    slide_identifiers = [
                        pid.replace("_page_", "_slide_") for pid in page_identifiers
                    ]
                    result["page_summaries"] = page_summaries
                    result["page_identifiers"] = slide_identifiers
                    print(f"  ✓ 幻燈片摘要生成: {time.time() - summary_start:.2f} 秒")
            else:
                print("  ❌ PPTX 轉 PDF 失敗，無法處理幻燈片")
        elif fname.lower().endswith(".docx"):
            result["file_type"] = "docx"
            print("  ⚠️ DOCX 處理功能尚未完全實現，僅提取文本")
            try:
                full_path = os.path.join(fpath, fname)
                elements = partition_docx(
                    filename=full_path, extract_images_in_tables=True
                )
                texts, tables = categorize_elements(elements)
                result["texts"] = texts
                result["tables"] = tables
                print(
                    f"  ✓ DOCX 元素提取完成，獲取了 {len(texts)} 個文本段落和 {len(tables)} 個表格"
                )
            except Exception as docx_err:
                print(f"  ❌ DOCX 處理出錯: {docx_err}")
    except Exception as e:
        print(f"  ❌ 處理文件 {fname} 時出錯: {str(e)}")
        import traceback

        traceback.print_exc()
    processing_time = time.time() - file_start_time
    result["processing_time"] = processing_time
    print(f"  ✓ 文件 {fname} 處理完成: {processing_time:.2f} 秒")
    return result
