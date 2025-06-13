"""PDF related processing utilities."""

import os

from pdf2image import convert_from_path
from unstructured.partition.pdf import partition_pdf

__all__ = ["convert_pdf_to_page_images", "extract_pdf_elements"]


def convert_pdf_to_page_images(
    file_path: str, file_name: str, database_dir: str = "./database"
):
    """Convert each page of a PDF to an image and return the paths."""
    try:
        output_dir = f"{database_dir}/figures/{os.path.splitext(file_name)[0]}/pages"
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(file_path, file_name)
        print(f"🔄 開始處理 PDF: {full_path}")
        if not os.path.exists(full_path):
            print(f"❌ 錯誤: PDF 文件不存在: {full_path}")
            return []
        file_size = os.path.getsize(full_path) / (1024 * 1024)
        print(f"📄 PDF 文件大小: {file_size:.2f} MB")
        try:
            print(f"🔄 正在將 {file_name} 轉換為頁面圖片...")
            images = convert_from_path(full_path, dpi=200, thread_count=1)  # Reduced thread_count from 20 to 1
            print(f"✅ 成功讀取 {len(images)} 頁")
        except Exception as e:  # pragma: no cover - conversion can fail in CI
            print(f"❌ 轉換 PDF 頁面時出錯: {str(e)}")
            import traceback

            traceback.print_exc()
            return []
        page_image_paths = []
        for i, image in enumerate(images):
            try:
                page_num = i + 1
                image_path = os.path.join(output_dir, f"page_{page_num}.jpg")
                image.save(image_path, "JPEG")
                page_image_paths.append((page_num, image_path))
            except Exception as e:  # pragma: no cover - file system issues
                print(f"❌ 儲存頁面 {i+1} 時出錯: {str(e)}")
        print(f"✅ 已將 {file_name} 的 {len(images)} 頁轉換為圖片")
        return page_image_paths
    except Exception as e:  # pragma: no cover - unexpected errors
        print(f"❌ 處理 PDF 文件 {file_name} 時發生未預期的錯誤: {str(e)}")
        import traceback

        traceback.print_exc()
        return []


def extract_pdf_elements(
    file_path: str, file_name: str, database_dir: str = "./database"
):
    """Extract text, tables and images from a PDF."""
    output_dir = f"{database_dir}/figures/{os.path.splitext(file_name)[0]}"
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(file_path, file_name)
    elements = partition_pdf(
        filename=full_path,
        extract_images_in_pdf=False,
        infer_table_structure=False,  # Changed from True to False
        chunking_strategy="by_title",
        max_characters=8192,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir,
        extract_image_block_output_dir=output_dir,
        strategy="hi_res",
        hi_res_model_name="yolox",  # This might be irrelevant if table structure is not inferred
        languages=["eng", "chi_tra", "chi_tra_vert"],
    )
    return elements
