import base64
import glob
import io
import os

from PIL import Image

from utils.logging_config import get_logger

logger = get_logger(__name__)
import concurrent.futures  # 添加對 concurrent.futures 的導入
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from utils.LLM_Tool import gpt_4o, gpt_4o_for_summary, text_embedding_3_large


# 為單個幻燈片生成摘要的輔助函數
def _process_single_slide(args):
    """
    為單個 PPTX 幻燈片生成摘要的輔助函數
    args: 包含 (llm, slide_num, img_path, file_name) 的元組
    回傳：(summary, slide_identifier, slide_num) 的元組
    """
    llm, slide_num, img_path, file_name = args

    try:
        # 將圖片轉換為 base64
        img_base64 = encode_image(img_path)

        # 創建幻燈片標識符
        slide_identifier = f"{os.path.splitext(file_name)[0]}_slide_{slide_num}"

        # 準備提示
        system_prompt = """您是一名詳細的簡報分析專家。您的任務是對PowerPoint幻燈片進行全面分析和摘要。
        請提供詳細的摘要，包含以下內容：
        1. 該幻燈片的主要內容和主題
        2. 幻燈片上所有圖片的詳細描述（如果有）
        3. 幻燈片上所有表格的內容和結構（如果有）
        4. 該幻燈片的關鍵概念和要點
        
        請使用繁體中文回答，並提供一個結構清晰的詳細摘要。"""

        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
            {
                "type": "text",
                "text": f"這是 {file_name} 的第 {slide_num} 張幻燈片。請對這張幻燈片進行詳細分析和摘要，包含幻燈片主要內容、圖片描述、表格內容和關鍵概念。",
            },
        ]

        # 創建消息對象
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 呼叫 GPT-4o 生成摘要
        response = llm.invoke(messages)
        summary = response.content

        logger.info(f"✅ 成功為 {file_name} 第 {slide_num} 張幻燈片生成摘要")
        return summary, slide_identifier, slide_num
    except Exception as e:
        logger.error(f"❌ 處理 {file_name} 第 {slide_num} 張幻燈片時出錯: {e}")
        return None, None, slide_num


# 新增的 PPTX 幻燈片摘要生成函數
def generate_pptx_slide_summaries(
    file_path: str,
    file_name: str,
    slide_image_paths: List[Tuple[int, str]],
    max_workers: int = 4,
) -> Tuple[List[str], List[str]]:
    """
    為 PPTX 每一張幻燈片生成詳細摘要
    file_path: PPTX 檔案所在資料夾路徑
    file_name: PPTX 檔案名稱
    slide_image_paths: 包含(幻燈片編號, 圖片路徑)的列表
    max_workers: 最大並行工作線程數
    回傳：摘要列表和對應的幻燈片標識符列表
    """
    if not slide_image_paths:
        logger.error(f"❌ 未找到 {file_name} 的幻燈片圖片，無法生成摘要")
        return [], []

    # 初始化結果列表
    slide_summaries = []
    slide_identifiers = []  # 格式: "filename_slide_N"

    # 初始化 GPT-4o 模型 (只初始化一次)
    llm = gpt_4o_for_summary()

    # 準備工作參數
    tasks = [
        (llm, slide_num, img_path, file_name)
        for slide_num, img_path in slide_image_paths
    ]

    # 設定最大工作線程數，不超過幻燈片數量
    actual_max_workers = min(max_workers, len(slide_image_paths))
    logger.info(
        f"🚀 開始並行處理 {file_name} 的 {len(slide_image_paths)} 張幻燈片，使用 {actual_max_workers} 個並行工作線程"
    )

    # 使用線程池並行處理
    processed_slides = 0
    failed_slides = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=actual_max_workers
    ) as executor:
        # 提交所有任務
        future_to_slide = {
            executor.submit(_process_single_slide, task): task[1] for task in tasks
        }

        # 按完成順序處理結果
        for future in concurrent.futures.as_completed(future_to_slide):
            slide_num = future_to_slide[future]
            try:
                summary, slide_identifier, _ = future.result()
                if summary and slide_identifier:
                    slide_summaries.append(summary)
                    slide_identifiers.append(slide_identifier)
                    processed_slides += 1
                else:
                    failed_slides += 1
            except Exception as e:
                logger.error(
                    f"❌ 獲取 {file_name} 第 {slide_num} 張幻燈片的摘要結果時出錯: {e}"
                )
                failed_slides += 1

    logger.info(
        f"📊 並行處理完成：成功為 {processed_slides}/{len(slide_image_paths)} 張幻燈片生成了摘要，失敗 {failed_slides} 張"
    )

    # 確保結果按幻燈片順序排序
    # 由於並行處理可能導致結果順序混亂，這裡重新按幻燈片編號排序
    if len(slide_summaries) > 0:
        # 從slide_identifiers中提取幻燈片編號
        slide_numbers = [int(sid.split("_")[-1]) for sid in slide_identifiers]
        # 按幻燈片編號排序
        sorted_results = sorted(
            zip(slide_summaries, slide_identifiers, slide_numbers), key=lambda x: x[2]
        )
        # 解包排序後的結果
        slide_summaries = [result[0] for result in sorted_results]
        slide_identifiers = [result[1] for result in sorted_results]

    return slide_summaries, slide_identifiers


# 為單個頁面生成摘要的輔助函數
def _process_single_page(args):
    """
    為單個 PDF 頁面生成摘要的輔助函數
    args: 包含 (llm, page_num, img_path, file_name) 的元組
    回傳：(summary, page_identifier, page_num) 的元組
    """
    llm, page_num, img_path, file_name = args

    try:
        # 將圖片轉換為 base64
        img_base64 = encode_image(img_path)

        # 創建頁面標識符
        page_identifier = f"{os.path.splitext(file_name)[0]}_page_{page_num}"

        # 準備提示
        system_prompt = """您是一名詳細的文件分析專家。您的任務是對PDF的每一頁進行全面分析和摘要。
        請提供詳細的摘要，包含以下內容：
        1. 該頁面的主要內容和主題
        2. 頁面上所有圖片的詳細描述（如果有）
        3. 頁面上所有表格的內容和結構（如果有）
        4. 該頁面的關鍵概念和要點
        
        請使用繁體中文回答，並提供一個結構清晰的詳細摘要。"""

        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
            {
                "type": "text",
                "text": f"這是 {file_name} 的第 {page_num} 頁。請對這一頁進行詳細分析和摘要，包含頁面主要內容、圖片描述、表格內容和關鍵概念。",
            },
        ]

        # 創建消息對象
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 呼叫 GPT-4o 生成摘要
        response = llm.invoke(messages)
        summary = response.content

        logger.info(f"✅ 成功為 {file_name} 第 {page_num} 頁生成摘要")
        return summary, page_identifier, page_num
    except Exception as e:
        logger.error(f"❌ 處理 {file_name} 第 {page_num} 頁時出錯: {e}")
        return None, None, page_num


# 新增的 PDF 頁面摘要生成函數 (並行版本)
def generate_pdf_page_summaries(
    file_path: str,
    file_name: str,
    page_image_paths: List[Tuple[int, str]],
    max_workers: int = 6,
) -> Tuple[List[str], List[str]]:
    """
    為 PDF 每一頁生成詳細摘要 (並行處理版本)
    file_path: PDF 檔案所在資料夾路徑
    file_name: PDF 檔案名稱
    page_image_paths: 包含(頁碼, 圖片路徑)的列表
    max_workers: 最大並行工作線程數
    回傳：摘要列表和對應的頁面標識符列表
    """
    if not page_image_paths:
        logger.error(f"❌ 未找到 {file_name} 的頁面圖片，無法生成摘要")
        return [], []

    # 初始化結果列表
    page_summaries = []
    page_identifiers = []  # 格式: "filename_page_N"

    # 初始化 GPT-4o 模型 (只初始化一次)
    llm = gpt_4o_for_summary()

    # 準備工作參數
    tasks = [
        (llm, page_num, img_path, file_name) for page_num, img_path in page_image_paths
    ]

    # 設定最大工作線程數，不超過頁面數量
    actual_max_workers = min(max_workers, len(page_image_paths))
    logger.info(
        f"🚀 開始並行處理 {file_name} 的 {len(page_image_paths)} 頁，使用 {actual_max_workers} 個並行工作線程"
    )

    # 使用線程池並行處理
    processed_pages = 0
    failed_pages = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=actual_max_workers
    ) as executor:
        # 提交所有任務
        future_to_page = {
            executor.submit(_process_single_page, task): task[1] for task in tasks
        }

        # 按完成順序處理結果
        for future in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                summary, page_identifier, _ = future.result()
                if summary and page_identifier:
                    page_summaries.append(summary)
                    page_identifiers.append(page_identifier)
                    processed_pages += 1
                else:
                    failed_pages += 1
            except Exception as e:
                logger.error(
                    f"❌ 獲取 {file_name} 第 {page_num} 頁的摘要結果時出錯: {e}"
                )
                failed_pages += 1

    logger.info(
        f"📊 並行處理完成：成功為 {processed_pages}/{len(page_image_paths)} 頁生成了摘要，失敗 {failed_pages} 頁"
    )

    # 確保結果按頁碼順序排序
    # 由於並行處理可能導致結果順序混亂，這裡重新按頁碼排序
    if len(page_summaries) > 0:
        # 從page_identifiers中提取頁碼
        page_numbers = [int(pid.split("_")[-1]) for pid in page_identifiers]
        # 按頁碼排序
        sorted_results = sorted(
            zip(page_summaries, page_identifiers, page_numbers), key=lambda x: x[2]
        )
        # 解包排序後的結果
        page_summaries = [result[0] for result in sorted_results]
        page_identifiers = [result[1] for result in sorted_results]

    return page_summaries, page_identifiers


def generate_text_summaries(texts_4k_token, table_strings, summarize_texts=True):
    """
    為文字和表格生成摘要
    texts_4k_token: 文字區塊列表
    table_strings: 表格字串列表
    summarize_texts: 是否為文字生成摘要
    回傳：文字摘要列表、表格摘要列表
    """
    # 提示
    prompt_text = """您是一名助手，負責摘要表格和文本。 \
    這些摘要將被作為embedding並用於檢索原始文本或表格元素。 \
    請提供針對檢索優化的詳細摘要。表格或文本：{element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # 文本摘要鏈
    model = gpt_4o_for_summary()
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # 初始化摘要列表
    text_summaries = []
    table_summaries = []

    # 如果提供文本且需要摘要，則應用於文本
    if texts_4k_token and summarize_texts:
        text_summaries = summarize_chain.batch(texts_4k_token, {"max_concurrency": 4})
    elif texts_4k_token:
        text_summaries = texts_4k_token

    # 如果提供表格，則應用於表格
    if table_strings:
        table_summaries = summarize_chain.batch(table_strings, {"max_concurrency": 4})

    return text_summaries, table_summaries


def encode_image(image_path):
    """將圖片編碼為 base64 字串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_meaningful_image(image_path, min_width=100, min_height=100, min_file_size_kb=10):
    """
    判斷圖片是否有意義
    image_path: 圖片路徑
    min_width: 最小寬度 (像素)
    min_height: 最小高度 (像素)
    min_file_size_kb: 最小文件大小 (KB)
    回傳：圖片是否有意義
    """
    try:
        # 檢查文件大小
        file_size_kb = os.path.getsize(image_path) / 1024
        if file_size_kb < min_file_size_kb:
            return False

        # 檢查圖片尺寸
        with Image.open(image_path) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return False

            # 檢查圖片是否為空白或單色
            if img.mode == "RGB":
                # 採樣檢查圖片複雜度
                colors = img.getcolors(maxcolors=256)
                # 如果顏色種類非常少，可能是裝飾圖片
                if colors is not None and len(colors) < 5:
                    return False

        return True
    except Exception as e:
        logger.info(f"檢查圖片時出錯 {image_path}: {e}")
        return False


def get_image_hash(image_path):
    """
    獲取圖片的哈希值，用於去重
    image_path: 圖片路徑
    回傳：圖片哈希值
    """
    try:
        with Image.open(image_path) as img:
            # 調整大小以加快哈希計算
            img = img.resize((64, 64), Image.LANCZOS)
            # 轉為灰度
            if img.mode != "L":
                img = img.convert("L")
            # 計算哈希
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if pixel > avg else "0" for pixel in pixels)
            hexadecimal = hex(int(bits, 2))[2:][:16].zfill(16)
            return hexadecimal
    except Exception as e:
        logger.info(f"計算圖片哈希時出錯 {image_path}: {e}")
        return None


def extract_image_paths(figures_dir: str, filter_images=True) -> list:
    """
    從圖片目錄提取圖片路徑，並過濾無意義的圖片
    figures_dir: 圖片目錄路徑
    filter_images: 是否過濾無意義的圖片
    回傳：圖片路徑列表
    """
    DATABASE_DIR = "./database"
    image_paths = []

    # 檢查 figures_dir 是否存在
    if not os.path.exists(figures_dir):
        logger.warning(f"⚠️ 圖片目錄不存在: {figures_dir}")
        return []

    # 獲取所有子目錄
    subdirs = [
        d
        for d in os.listdir(figures_dir)
        if os.path.isdir(os.path.join(figures_dir, d))
    ]

    # 處理每個子目錄中的圖片
    for subdir in subdirs:
        subdir_path = os.path.join(figures_dir, subdir)

        # 獲取子目錄中的所有圖片
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
        for ext in image_extensions:
            pattern = os.path.join(subdir_path, ext)
            image_paths.extend(glob.glob(pattern))

    # 如果沒有找到圖片，記錄警告
    if not image_paths:
        logger.warning(f"⚠️ 在 {figures_dir} 及其子目錄中未找到任何圖片")
        return []

    # 過濾無意義的圖片
    if filter_images:
        logger.info(f"🔍 開始過濾圖片，原始圖片數量: {len(image_paths)}")

        filtered_paths = []
        image_hashes = set()  # 用於存儲已處理過的圖片哈希值

        for img_path in image_paths:
            # 檢查圖片是否有意義
            if is_meaningful_image(img_path):
                # 計算圖片哈希值用於去重
                img_hash = get_image_hash(img_path)
                if img_hash and img_hash not in image_hashes:
                    image_hashes.add(img_hash)
                    filtered_paths.append(img_path)

        logger.info(
            f"🔍 過濾完成，過濾後圖片數量: {len(filtered_paths)}/{len(image_paths)}"
        )
        return filtered_paths

    return image_paths


def generate_img_summaries(
    figures_dir: str, filter_images=True
) -> Tuple[List[str], List[str]]:
    """
    為圖片生成摘要
    figures_dir: 圖片目錄路徑
    filter_images: 是否過濾無意義的圖片
    回傳：摘要列表和對應的圖片文件名列表
    """
    # 獲取所有圖片路徑，並過濾無意義的圖片
    image_paths = extract_image_paths(figures_dir, filter_images=filter_images)
    if not image_paths:
        logger.error("❌ 沒有找到任何有意義的圖片，無法生成摘要")
        return [], []

    img_summaries = []
    img_filenames = []  # 用於存儲相對路徑

    # 為每張圖片生成摘要
    for img_path in image_paths:
        # 轉換為相對路徑 (從 database/figures/ 開始)
        img_relative_path = os.path.relpath(img_path, os.path.dirname(figures_dir))

        # 將圖片轉換為 base64
        try:
            img_base64 = encode_image(img_path)

            # 初始化 GPT-4o 模型
            llm = gpt_4o_for_summary()

            # 準備消息
            system_message = {
                "role": "system",
                "content": "用繁體中文回答和使用直覺簡明的方式說明圖片內容：請描述這張圖片中有什麼，盡量細節到位。回答要全部都只有一段話，不要使用條列式或是項目符號。",
            }
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
                {
                    "type": "text",
                    "text": "簡明扼要地描述這張圖片顯示了什麼內容。回覆只需要一段文字，不要使用條列式。",
                },
            ]

            # 創建消息對象
            messages = [HumanMessage(content=user_content)]

            # 呼叫 GPT-4o 生成摘要
            response = llm.invoke(messages)
            summary = response.content

            # 添加摘要和文件名到列表
            img_summaries.append(summary)
            img_filenames.append(img_relative_path)

            logger.info(f"✅ 成功為圖片 {img_relative_path} 生成摘要")
        except Exception as e:
            logger.error(f"❌ 處理圖片 {img_path} 時出錯: {e}")

    logger.info(f"📊 共為 {len(img_summaries)}/{len(image_paths)} 張圖片生成了摘要")
    return img_summaries, img_filenames


if __name__ == "__main__":
    logger.info(
        "This module is part of the processing pipeline; invoke it via app.py or main.py."
    )
