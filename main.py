import os
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")


import json
import base64

from utils.vector_store import  resize_base64_image # Updated import
from utils.summarize import encode_image # Updated import
from utils.LLM_Tool import gpt_4o, text_embedding_3_large

from langchain.retrievers.multi_vector import MultiVectorRetriever 
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# 統一定義資料庫路徑 - 更新為 FAISS
DATABASE_DIR = "./database"
FAISS_INDEX_PATH = f"{DATABASE_DIR}/faiss_store"  # Changed from VECTORSTORE_PATH
FAISS_INDEX_NAME = "mm_rag_faiss_index"  # FAISS index name
DOCSTORE_MAPPING_PATH = f"{DATABASE_DIR}/docstore_mapping.json"
IMAGE_FIGURES_PATH = f"{DATABASE_DIR}/figures/"

def split_image_text_types(docs):
    """
    將 MultiVectorRetriever 檢索回來的原始內容分為圖片 (base64) 與文字/表格。
    docs: 檢索回來的文件列表 (包含原始文字、表格字串，或各種類型的字典)
    回傳：dict，含 images（base64 圖片列表）、texts（文字/表格列表）
    """
    b64_images = []
    texts = []
    page_summaries = []
    slide_summaries = []  # 新增：幻燈片摘要列表
    
    for doc in docs:
        if isinstance(doc, Document): # Handle cases where retriever might still wrap in Document
             doc_content = doc.page_content
        else:
             doc_content = doc # Assume it's the raw content from the docstore

        # 處理不同類型的字典內容
        if isinstance(doc_content, dict):
            # 檢查字典的類型
            doc_type = doc_content.get('type', '')
            
            # 處理圖片類型
            if doc_type == 'image' or ('filename' in doc_content and 'type' not in doc_content):
                img_filename = doc_content['filename']
                # 如果檔名包含文檔子資料夾路徑
                img_path = os.path.join(IMAGE_FIGURES_PATH, img_filename)
                
                # 檢查圖片是否存在
                if os.path.exists(img_path):
                    try:
                        # Encode the image from file path
                        b64_img = encode_image(img_path)
                        # Resize the base64 image
                        resized_b64_img = resize_base64_image(b64_img, size=(1300, 600)) # Adjust size as needed
                        b64_images.append(resized_b64_img)
                    except Exception as e:
                        print(f"⚠️ Error processing image file {img_path}: {e}")
                        # 嘗試記錄更多診斷信息
                        if os.path.isfile(img_path):
                            file_size = os.path.getsize(img_path)
                            print(f"   File exists, size: {file_size} bytes")
                            if file_size == 0:
                                print(f"   Warning: File size is 0 bytes!")
                        else:
                            print(f"   Strange: Path exists but is not a file!")
                else:
                    # 增強圖片未找到的診斷訊息
                    print(f"⚠️ Image file not found: {img_path}")
                    
                    # 檢查子資料夾是否存在
                    doc_folder = os.path.dirname(img_path)
                    if not os.path.exists(doc_folder):
                        print(f"   Document folder does not exist: {doc_folder}")
                    else:
                        print(f"   Document folder exists, but image file is missing")
                        # 列出子資料夾中的所有檔案以進行診斷
                        folder_files = os.listdir(doc_folder)
                        print(f"   Files in folder ({len(folder_files)}): {', '.join(folder_files[:5])}" + 
                              (f"... and {len(folder_files)-5} more" if len(folder_files) > 5 else ""))
            
            # 處理 PDF 頁面摘要類型
            elif doc_type == 'pdf_page':
                page_id = doc_content.get('page_id', 'unknown_page')
                summary = doc_content.get('summary', '無摘要')
                page_summaries.append(f"【頁面: {page_id}】\n{summary}")
            
            # 新增：處理 PPTX 幻燈片摘要類型
            elif doc_type == 'pptx_slide':
                slide_id = doc_content.get('slide_id', 'unknown_slide')
                summary = doc_content.get('summary', '無摘要')
                slide_summaries.append(f"【幻燈片: {slide_id}】\n{summary}")
            
            # 處理文本或表格類型
            elif doc_type in ['text', 'table']:
                content = doc_content.get('content', '')
                if content:
                    texts.append(content)
            
            # 處理未知類型的字典
            else:
                print(f"⚠️ Unknown dictionary type: {doc_content}")
        
        # 處理字符串類型 (向後兼容舊格式)
        elif isinstance(doc_content, str):
            texts.append(doc_content)
        else:
             print(f"⚠️ Unexpected document type received: {type(doc_content)}")

    # 增加處理結果總結
    if len(b64_images) == 0 and any(isinstance(d, dict) and 'filename' in d for d in docs):
        print("⚠️ Warning: Image references were found but no images could be loaded!")
        
    # 將頁面和幻燈片摘要也添加到文本列表中
    texts.extend(page_summaries)
    texts.extend(slide_summaries)  # 新增：加入幻燈片摘要
        
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    組合多模態 LLM 輸入 prompt (通用版本)。
    data_dict: dict，含 context 與 question
    回傳：LLM 輸入訊息列表
    將檢索到的圖片與文字組合成 LLM 可理解的格式。
    """
    # 從 context 中提取文字/表格內容
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    # 從 context 中提取圖片 (base64)
    images = data_dict["context"]["images"]
    # 獲取使用者的問題
    question = data_dict['question']

    # --- Prompt 設計 ---
    # 1. 設定角色/目標 (通用)
    role_description = "您是一位 AI 助手，擅長分析和整合提供的多模態資訊（包含文字、表格和圖像）。"

    # 2. 描述輸入
    input_description = "您將收到以下內容："
    if images:
        input_description += "\n- 一或多張圖像。"
    if formatted_texts.strip(): # 檢查是否有非空白文字
        input_description += "\n- 相關的文字段落和/或表格。"

    # 3. 設定任務指令
    task_instruction = f"""請根據以上提供的所有資訊（圖像、文字、表格），精確且僅基於這些資訊來回答以下問題：
"{question}"

請確保您的回答直接回應問題，並整合來自不同來源的相關細節，且一律使用繁體中文回答。如果提供的資訊不足以回答問題，請明確說明。"""
    # --- Prompt 組合 ---
    final_prompt_text = f"{role_description}\n\n{input_description}\n\n{task_instruction}\n\n提供的文字與表格內容：\n------\n{formatted_texts}\n------"

    # --- 建立 LangChain 訊息 ---
    messages = []

    # 1. 添加圖像訊息 (如果有的話)
    if images:
        for image in images:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # 2. 添加文字訊息 (包含完整的 prompt)
    text_message = {
        "type": "text",
        "text": final_prompt_text,
    }
    messages.append(text_message)

    # 返回 HumanMessage 列表
    return [HumanMessage(content=messages)]



def multi_modal_rag_chain(retriever):
    """
    建立多模態 RAG 推理鏈。
    retriever: 多向量檢索器
    回傳：chain（RAG 推理鏈）
    """

    # Multi-modal LLM
    model = gpt_4o()

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain



# --- Main Execution Logic ---

# 1. Load the Vector Store (FAISS)
print("🔄 Loading FAISS Vector Store...")
try:
    faiss_actual_index_file = os.path.join(FAISS_INDEX_PATH, f"{FAISS_INDEX_NAME}.faiss")
    if os.path.exists(faiss_actual_index_file):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            text_embedding_3_large,
            FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS Vector Store Loaded.")
    else:
        print("⚠️ FAISS index not found. Creating placeholder index...")
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        # Create a dummy FAISS index 
        initial_doc_texts = ["placeholder document for faiss initialization"]
        vectorstore = FAISS.from_texts(initial_doc_texts, text_embedding_3_large)
        vectorstore.save_local(FAISS_INDEX_PATH, FAISS_INDEX_NAME)
        print("✅ Placeholder FAISS Vector Store Created.")
except Exception as e:
    print(f"❌ Error loading FAISS vector store: {e}")
    print("Creating new placeholder FAISS index...")
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    initial_doc_texts = ["placeholder document for faiss initialization"]
    vectorstore = FAISS.from_texts(initial_doc_texts, text_embedding_3_large)
    vectorstore.save_local(FAISS_INDEX_PATH, FAISS_INDEX_NAME)
    print("✅ New FAISS Vector Store Created.")

# 2. Load the Docstore Mapping
print(f"🔄 Loading Docstore Mapping from {DOCSTORE_MAPPING_PATH}...")
try:
    with open(DOCSTORE_MAPPING_PATH, 'r', encoding='utf-8') as f:
        loaded_mapping = json.load(f)
    print("✅ Docstore Mapping Loaded.")
except FileNotFoundError:
    print(f"⚠️ Warning: Docstore mapping file not found at {DOCSTORE_MAPPING_PATH}. Creating empty mapping.")
    # 创建空的映射和必要的目录
    os.makedirs(os.path.dirname(DOCSTORE_MAPPING_PATH), exist_ok=True)
    loaded_mapping = {}
    with open(DOCSTORE_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(loaded_mapping, f, ensure_ascii=False, indent=4)
except json.JSONDecodeError:
    print(f"❌ Error: Could not decode JSON from {DOCSTORE_MAPPING_PATH}. Creating empty mapping.")
    loaded_mapping = {}
    with open(DOCSTORE_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(loaded_mapping, f, ensure_ascii=False, indent=4)
except Exception as e:
    print(f"❌ An unexpected error occurred while loading the mapping: {e}")
    print("Creating empty mapping.")
    loaded_mapping = {}
    with open(DOCSTORE_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(loaded_mapping, f, ensure_ascii=False, indent=4)


# 3. Reconstruct the InMemoryStore and Populate it
print("🔄 Reconstructing Docstore...")
store = InMemoryStore()
store.mset(list(loaded_mapping.items())) # Populate the store
print("✅ Docstore Reconstructed.")

# 4. Initialize the MultiVectorRetriever CORRECTLY
print("🔄 Initializing MultiVectorRetriever...")
retriever_multi_vector_img = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id", # Must match the key used when storing documents
)
print("✅ MultiVectorRetriever Initialized.")


# 5. Build the RAG Chain
print("🔄 Building RAG Chain...")
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
print("✅ RAG Chain Built.")

# 將測試查詢代碼包裝在 if __name__ == "__main__" 內，避免在被導入時執行
if __name__ == "__main__":
    # 6. Run a Query
    query = "" # Example query
    print(f"\n❓ Running Query: {query}\n")

    # Optional: Check what the retriever fetches directly

    print("--- Retriever Output ---")
    retrieved_docs = retriever_multi_vector_img.invoke(query)
    for i, item in enumerate(retrieved_docs):
        print(f"--- Item {i+1} ---") # 為每個項目加上編號和分隔線

        if isinstance(item, str):
            # 如果是字串，標示為文字區塊並印出內容
            print("[Type: Text Block]")
            print("Content:")
            print(item.strip()) # 使用 strip() 去除前後多餘空白

        elif isinstance(item, dict) and 'filename' in item:
            print("[Type: Image Reference]")
            filename = item['filename']
            # Safely get the summary, provide default if missing
            summary = item.get('summary', '[Summary not found in docstore]')
            print(f"Referenced Image File: {filename}")
            print(f"Image Summary: {summary}") # Print the summary
        

        else:
            # 處理未預期的類型
            print(f"[Type: Unknown ({type(item)})]")
            print(f"Content: {item}")

        print("-" * 20) # 每個項目之間的分隔線
        print() # 增加空行，讓輸出更清晰

    print("------------------------\n")


    print("--- RAG Chain Output ---")
    for chunk in chain_multimodal_rag.stream(query):
        print(chunk, flush=True, end="")
    print("\n------------------------")
    print("\n✅ Query Finished.")
