import os
import json
import uuid
import base64
import io
import re
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional  # Added Any, Optional
from PIL import Image
from IPython.display import display, HTML

from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS  # Added FAISS import

# 新增資料庫資料夾路徑常數
DATABASE_DIR = "./database"

def create_multi_vector_retriever(
    vectorstore: Optional[Any],  # Modified to Optional[Any]
    text_summaries: List[str], 
    text_contents: List[str], 
    table_summaries: List[str], 
    table_contents: List[str],
    image_summaries: List[str],
    image_filenames: List[str],
    page_summaries: Optional[List[str]] = None,
    page_identifiers: Optional[List[str]] = None,
    slide_summaries: Optional[List[str]] = None,
    slide_identifiers: Optional[List[str]] = None,
    docstore_path: str = f"{DATABASE_DIR}/docstore_mapping.json",
    update_existing: bool = False,
    # Added FAISS specific parameters
    vectorstore_type: Optional[str] = None, 
    faiss_index_path: Optional[str] = None,
    faiss_index_name: Optional[str] = None,
    embedding_function: Optional[Any] = None
):
    """
    建立多向量檢索器
    vectorstore: 向量存儲 (FAISS instance, or None)
    text_summaries: 文字摘要列表
    text_contents: 原始文字內容列表
    table_summaries: 表格摘要列表
    table_contents: 原始表格內容列表
    image_summaries: 圖片摘要列表
    image_filenames: 圖片檔名列表
    page_summaries: PDF頁面摘要列表
    page_identifiers: PDF頁面標識符列表
    slide_summaries: PPTX幻燈片摘要列表
    slide_identifiers: PPTX幻燈片標識符列表
    docstore_path: 儲存對應關係的 JSON 檔案路徑
    update_existing: 是否更新現有的向量存儲
    vectorstore_type: Type of vectorstore ('faiss', etc.)
    faiss_index_path: Path to FAISS index folder
    faiss_index_name: Name of FAISS index files
    embedding_function: Embedding function to use for FAISS
    回傳：多向量檢索器
    """
    # 初始化文件存儲層，使用記憶體存儲
    store = InMemoryStore()
    # 定義文件唯一識別鍵名稱
    id_key = "doc_id"

    # 如果需要更新現有的 docstore 映射
    existing_mappings = {}
    if update_existing and os.path.exists(docstore_path):
        try:
            with open(docstore_path, 'r', encoding='utf-8') as f:
                existing_mappings = json.load(f)
            print(f"✅ 載入現有 docstore 映射 ({len(existing_mappings)} 條記錄)")
            
            # 將現有映射加入到存儲層
            store.mset(list(existing_mappings.items()))
        except Exception as e:
            print(f"⚠️ 載入現有 docstore 映射時出錯: {e}")
            # 如果出錯，創建新的映射
            existing_mappings = {}

    # Initialize vectorstore if it's None and type is FAISS
    if vectorstore is None and vectorstore_type == 'faiss':
        if not all([faiss_index_path, faiss_index_name, embedding_function]):
            raise ValueError("FAISS parameters (faiss_index_path, faiss_index_name, embedding_function) must be provided for FAISS type when vectorstore is None.")
        
        faiss_actual_index_file = os.path.join(faiss_index_path, f"{faiss_index_name}.faiss")
        # Ensure directory exists for loading or creating
        os.makedirs(faiss_index_path, exist_ok=True)

        if os.path.exists(faiss_actual_index_file):
            try:
                print(f"[FAISS create_multi_vector_retriever] Attempting to load existing FAISS index from {faiss_index_path}/{faiss_index_name}")
                vectorstore = FAISS.load_local(
                    folder_path=faiss_index_path,
                    embeddings=embedding_function,
                    index_name=faiss_index_name,
                    allow_dangerous_deserialization=True
                )
                print(f"[FAISS create_multi_vector_retriever] Successfully loaded FAISS index.")
            except Exception as e:
                print(f"⚠️ [FAISS create_multi_vector_retriever] Failed to load existing index: {e}. Creating a new one.")
                # Create with a placeholder, as FAISS can't be empty initially for some operations
                vectorstore = FAISS.from_texts(
                    texts=["Initial placeholder for FAISS index"], 
                    embedding=embedding_function
                )
                vectorstore.save_local(folder_path=faiss_index_path, index_name=faiss_index_name)
                print(f"[FAISS create_multi_vector_retriever] Created and saved a new placeholder FAISS index because load failed.")
        else:
            print(f"[FAISS create_multi_vector_retriever] No existing FAISS index found at {faiss_index_path}/{faiss_index_name}. Creating a new one.")
            vectorstore = FAISS.from_texts(
                texts=["Initial placeholder for FAISS index"], 
                embedding=embedding_function
            )
            vectorstore.save_local(folder_path=faiss_index_path, index_name=faiss_index_name)
            print(f"[FAISS create_multi_vector_retriever] Created and saved a new placeholder FAISS index.")
            
    elif vectorstore is None and vectorstore_type is not None and vectorstore_type != 'faiss':
        raise ValueError(f"Vectorstore is None and type is '{vectorstore_type}'. This configuration is not handled yet.")
    elif vectorstore is not None and vectorstore_type == 'faiss':
        if not isinstance(vectorstore, FAISS):
            raise ValueError("Provided vectorstore is not a FAISS instance, but vectorstore_type is 'faiss'.")
        print("[FAISS create_multi_vector_retriever] Using provided FAISS vectorstore.")
    elif vectorstore is None and vectorstore_type is None:
        raise ValueError("Vectorstore is None and vectorstore_type is also None. Cannot proceed.")


    # 建立多向量檢索器，結合向量庫與文件存儲層
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 用來儲存所有 id -> content 的映射
    all_mappings = existing_mappings.copy() if update_existing else {}

    # 定義輔助函式，將摘要與原始內容加入向量庫與文件存儲
    def add_documents(current_retriever, doc_summaries, doc_contents, doc_filenames=None, doc_type=None):
        if not doc_contents: # Avoid error if contents are empty
            return
        # 生成與內容數量相同的唯一 UUID 列表，作為文件 ID
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        # 建立摘要文件列表，每個摘要對應一個 Document，並附帶唯一 doc_id 元資料
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        # 將摘要文件加入向量庫，建立向量索引
        current_retriever.vectorstore.add_documents(summary_docs)

        # Prepare content for docstore
        if doc_type == "page":  # 如果是頁面類型
            # 存儲頁面信息和摘要
            store_contents = [{'page_id': doc_contents[i], 'summary': doc_summaries[i], 'type': 'pdf_page'}
                             for i in range(len(doc_contents))]
        elif doc_type == "slide":  # 新增：如果是幻燈片類型
            # 存儲幻燈片信息和摘要
            store_contents = [{'slide_id': doc_contents[i], 'summary': doc_summaries[i], 'type': 'pptx_slide'}
                             for i in range(len(doc_contents))]
        elif doc_filenames:  # 如果提供了文件名（對於圖片）
            # 存儲帶有文件名和摘要的字典
            store_contents = [{'filename': doc_contents[i], 'summary': doc_summaries[i], 'type': 'image'}
                             for i in range(len(doc_contents))]
        else:  # 對於文字和表格，直接存儲內容
            # 添加類型信息
            doc_type = "text" if doc_summaries == text_summaries else "table"
            store_contents = [{'content': content, 'type': doc_type} for content in doc_contents]

        # 將原始內容以 doc_id 為鍵，存入文件存儲層
        mappings = list(zip(doc_ids, store_contents))
        retriever.docstore.mset(mappings)
        # 更新總映射
        all_mappings.update(dict(mappings))


    if text_summaries and text_contents:
        add_documents(retriever, text_summaries, text_contents)

    if table_summaries and table_contents:
        add_documents(retriever, table_summaries, table_contents)

    if image_summaries and image_filenames:
        add_documents(retriever, image_summaries, image_filenames, image_filenames)
    
    # 添加頁面摘要處理
    if page_summaries and page_identifiers:
        add_documents(retriever, page_summaries, page_identifiers, doc_type="page")
        print(f"✅ 已添加 {len(page_summaries)} 頁 PDF 頁面摘要到向量庫")
    
    # 新增：添加幻燈片摘要處理
    if slide_summaries and slide_identifiers:
        add_documents(retriever, slide_summaries, slide_identifiers, doc_type="slide")
        print(f"✅ 已添加 {len(slide_summaries)} 張 PPTX 幻燈片摘要到向量庫")

    # 保存映射到文件
    try:
        os.makedirs(os.path.dirname(docstore_path), exist_ok=True)
        with open(docstore_path, 'w', encoding='utf-8') as f:
            json.dump(all_mappings, f, ensure_ascii=False, indent=4)
        print(f"✅ Docstore mapping saved to {docstore_path}")
    except Exception as e:
        print(f"❌ Error saving docstore mapping: {e}")

    # Save the FAISS index if it was used and potentially modified
    if vectorstore_type == 'faiss' and isinstance(vectorstore, FAISS) and faiss_index_path and faiss_index_name:
        try:
            print(f"[FAISS create_multi_vector_retriever] Saving FAISS index to {faiss_index_path}/{faiss_index_name}...")
            vectorstore.save_local(folder_path=faiss_index_path, index_name=faiss_index_name)
            print(f"✅ [FAISS create_multi_vector_retriever] FAISS index saved successfully.")
        except Exception as e:
            print(f"❌ [FAISS create_multi_vector_retriever] Error saving FAISS index: {e}")
            # Potentially raise the error or handle it as critical
            raise e # Re-raise to make the calling function aware of save failure

    return retriever



def plt_img_base64(img_base64):
    """
    將 base64 字串以圖片方式顯示於 notebook。
    img_base64: 圖片 base64 字串
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

def looks_like_base64(sb):
    """
    判斷字串是否為 base64 格式。
    sb: 字串
    回傳：True/False
    """
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """
    判斷 base64 字串是否為圖片資料。
    b64data: base64 字串
    回傳：True/False
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


# 功能：調整 base64 圖片大小。
# 輸入：base64_string（圖片 base64 字串）、size（目標尺寸）
# 輸出：調整後的 base64 字串
# 目的：確保圖片輸入 LLM 時尺寸合適。
def resize_base64_image(base64_string, size=(128, 128)):
    """
    調整 base64 圖片大小。
    base64_string: 圖片 base64 字串
    size: 目標尺寸
    回傳：調整後的 base64 字串
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


if __name__ == "__main__":
    print("This module is used by the main workflow; run app.py or main.py instead of executing it directly.")
