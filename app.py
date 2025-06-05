\
# uvicorn app:app --host 0.0.0.0 --port 1230 --reload
import os
import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import sys
import uuid
import shutil
import uvicorn
import json
import time
import threading
import asyncio
from concurrent.futures import ProcessPoolExecutor

# Import the RAG functionality from main.py
from main import multi_modal_rag_chain
# Import required utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.LLM_Tool import text_embedding_3_large
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_text_splitters import NLTKTextSplitter # Ensure NLTK data is downloaded

# Create FastAPI app
app = FastAPI(
    title="MM-RAG API (FAISS Version)",
    description="Multimodal Retrieval Augmented Generation API using FAISS",
    version="0.1.1", # Incremented version
    root_path="/mm_rag" # Changed to match the new directory structure
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
DATABASE_DIR = "./database"
FAISS_INDEX_PATH = f"{DATABASE_DIR}/faiss_store" # Changed from VECTORSTORE_PATH
FAISS_INDEX_NAME = "mm_rag_faiss_index" # Name for FAISS index files
FIGURE_PATH = f"{DATABASE_DIR}/figures"
DOCSTORE_MAPPING_PATH = f"{DATABASE_DIR}/docstore_mapping.json" # Remains the same
UPLOAD_DIR = "./uploads"

# Simple in-memory status tracking
document_status = {}

# Global variables for the retriever and refresh logic
retriever_lock = threading.RLock()
last_refresh_time = 0
current_retriever = None
current_chain = None
refresh_interval = 15

process_pool = ProcessPoolExecutor(max_workers=4)
processing_semaphore = asyncio.Semaphore(4)
active_processing_tasks = {}

# Pydantic models (remain the same)
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    message: Optional[str] = None

class QueryResponse(BaseModel):
    # reference: str
    answer: str
    processing_time: float

# Function to refresh the retriever and chain with FAISS
def refresh_retriever():
    global current_retriever, current_chain, last_refresh_time
    
    with retriever_lock:
        print("🔄 刷新檢索器和 RAG 鏈 (FAISS 版本)...")
        start_time = time.time()
        
        try:
            vectorstore = None
            faiss_actual_index_file = os.path.join(FAISS_INDEX_PATH, f"{FAISS_INDEX_NAME}.faiss")

            if os.path.exists(faiss_actual_index_file):
                try:
                    vectorstore = FAISS.load_local(
                        FAISS_INDEX_PATH,
                        text_embedding_3_large,
                        FAISS_INDEX_NAME,
                        allow_dangerous_deserialization=True 
                    )
                    print(f"[FAISS] Loaded existing FAISS index from {FAISS_INDEX_PATH}/{FAISS_INDEX_NAME}")
                    if hasattr(vectorstore.index, "ntotal"):
                         print(f"[FAISS] Index size: {vectorstore.index.ntotal} vectors")
                except Exception as load_err:
                    print(f"⚠️ [FAISS] Error loading index: {load_err}. Will attempt to create a new one.")
                    vectorstore = None # Ensure it's None so a new one is created

            if vectorstore is None: # If not loaded or load failed
                os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
                # Create a dummy FAISS index so that MultiVectorRetriever can be initialized
                initial_doc_texts = ["placeholder document for faiss initialization"]
                vectorstore = FAISS.from_texts(initial_doc_texts, text_embedding_3_large)
                vectorstore.save_local(FAISS_INDEX_PATH, FAISS_INDEX_NAME)
                print(f"[FAISS] Initialized and saved a placeholder FAISS index at {FAISS_INDEX_PATH}/{FAISS_INDEX_NAME}")
            

            if os.path.exists(DOCSTORE_MAPPING_PATH):
                with open(DOCSTORE_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    loaded_mapping = json.load(f)
            else:
                loaded_mapping = {}
            
            store = InMemoryStore()
            store.mset(list(loaded_mapping.items()))
            

            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=store,
                id_key="doc_id",
            )
            
            chain = multi_modal_rag_chain(retriever)
            
            current_retriever = retriever
            current_chain = chain
            last_refresh_time = time.time()
            
            refresh_time = time.time() - start_time
            print(f"✅ 檢索器刷新完成 (FAISS) (耗時: {refresh_time:.2f} 秒)")
            print(f"📊 文檔映射條目數: {len(loaded_mapping)}")
            
            return True
        except Exception as e:
            print(f"❌ 檢索器刷新失敗 (FAISS): {str(e)}")
            import traceback
            traceback.print_exc()
            return False

@app.on_event("startup")
async def startup_event():
    print("🚀 初始化 MM-RAG 系統 (FAISS 版本)...")
    
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True) # Use FAISS path
    os.makedirs(FIGURE_PATH, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    if not os.path.exists(DOCSTORE_MAPPING_PATH):
        print("🔍 向量數據庫映射文件不存在，創建空映射文件...")
        with open(DOCSTORE_MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    
    refresh_retriever()
    
    print("✅ MM-RAG 系統 (FAISS) 初始化完成，準備接收請求")

# Modified for FAISS
def add_file_to_vector_db(file_path: str) -> bool:
    from utils.vector_store import create_multi_vector_retriever # This util needs to be FAISS-aware
    from utils.summarize import generate_img_summaries, generate_text_summaries
    from utils.extract_file_utils import process_single_file
    
    print(f"\n🔄 處理上傳檔案 (FAISS): {file_path}")
    
    try:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext not in ['.pdf', '.pptx', '.ppt', '.docx']:
            print(f"❌ 不支援的檔案類型: {file_ext}")
            return False
        
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file_name)
        shutil.copy2(file_path, temp_file_path)
        
        result = process_single_file(file_name, temp_dir) # Assumes this util is generic
        
        if not result:
            print(f"❌ 檔案處理失敗: {file_name}")
            shutil.rmtree(temp_dir)
            return False
        
        texts = result['texts']
        tables = result['tables']
        page_summaries = result['page_summaries']
        page_identifiers = result['page_identifiers']
        
        joined_texts = " ".join(texts)
        text_splitter = NLTKTextSplitter()
        texts_4k_token = text_splitter.split_text(joined_texts) if joined_texts else []
        
        text_summaries, table_summaries = generate_text_summaries(texts_4k_token, tables, summarize_texts=True)
        image_summaries, img_filenames = generate_img_summaries(FIGURE_PATH) # Assumes FIGURE_PATH is managed correctly
        
        faiss_actual_index_file = os.path.join(FAISS_INDEX_PATH, f"{FAISS_INDEX_NAME}.faiss")
        is_new_db = not os.path.exists(faiss_actual_index_file)
        
        vectorstore = None
        if not is_new_db:
            try:
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    text_embedding_3_large, 
                    FAISS_INDEX_NAME,
                    allow_dangerous_deserialization=True
                )
                print(f"[FAISS] Loaded existing FAISS index for update.")
            except Exception as e:
                print(f"⚠️ [FAISS] Could not load existing index for update: {e}. Will create new.")
                is_new_db = True # Force creation
                vectorstore = None

        _ = create_multi_vector_retriever(
            vectorstore=vectorstore, # Pass the loaded store, or None if new
            text_summaries=text_summaries, 
            text_contents=texts_4k_token, 
            table_summaries=table_summaries, 
            table_contents=tables, 
            image_summaries=image_summaries,
            image_filenames=img_filenames, # Changed from img_filenames and added comma
            page_summaries=page_summaries,
            page_identifiers=page_identifiers,
            docstore_path=DOCSTORE_MAPPING_PATH,
            update_existing=not is_new_db,
            # FAISS specific parameters that create_multi_vector_retriever might need:
            vectorstore_type='faiss', # Explicitly tell the util
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_index_name=FAISS_INDEX_NAME,
            embedding_function=text_embedding_3_large
        )
        
        shutil.rmtree(temp_dir)
        
        print(f"✅ 檔案 {file_name} 成功添加到向量資料庫 (FAISS)")
        return True
        
    except Exception as e:
        print(f"❌ 處理檔案時發生錯誤 (FAISS): {str(e)}")
        import traceback
        traceback.print_exc()
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

# Helper functions (remain the same)
def update_document_status(document_id: str, status: str, message: str = "", filename: str = None):
    document_status[document_id] = {
        "status": status,
        "message": message,
        "filename": filename
    }

def get_document_status(document_id: str):
    return document_status.get(document_id, {
        "status": "unknown",
        "message": "Document not found",
        "filename": "unknown.pdf"
    })

# Subprocess and async handling (remain largely the same)
def process_file_in_subprocess(file_path: str) -> bool:
    try:
        return add_file_to_vector_db(file_path) # Calls the FAISS version
    except Exception as e:
        print(f"❌ processing file in subprocess: {str(e)}")
        return False

async def process_document_async(document_id: str, file_path: str, file_name: str):
    async with processing_semaphore:
        try:
            update_document_status(document_id, "processing", "Processing document...", file_name)
            print(f"🔄 file processing for {file_name} (ID: {document_id})")
            
            full_file_path = os.path.join(file_path, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if file_ext not in ['.pdf', '.pptx', '.ppt', '.docx', '.txt']:
                update_document_status(document_id, "failed", f"Unsupported file extension: {file_ext}", file_name)
                return
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                process_pool, 
                process_file_in_subprocess, 
                full_file_path
            )
            
            if success:
                update_document_status(document_id, "completed", "Document processed successfully", file_name)
                print(f"✅ file processed successfully {file_name} (ID: {document_id})")
                print("🔄 file processing completed, refreshing retriever")
                refresh_success = refresh_retriever()
                if refresh_success:
                    print("✅ retriever refreshed successfully")
                else:
                    print("⚠️ retriever refresh failed, please check logs")
            else:
                update_document_status(document_id, "failed", "Document processing failed", file_name)
                print(f"❌ document processing for {file_name} failed unexpectedly (ID: {document_id}) ")
        
        except Exception as e:
            print(f"❌ file processing error: {str(e)}")
            update_document_status(document_id, "failed", f"file processed failed: {str(e)}", file_name)
        
        finally:
            if document_id in active_processing_tasks:
                del active_processing_tasks[document_id]

# API Routes (structure remains the same, logic points to FAISS methods)
@app.get("/")
async def root():
    return {"message": "Welcome to MM-RAG API (FAISS Version)"}

@app.put("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    process_immediately: bool = Form(True)
):
    document_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        file_content = await file.read()
        with open(file_location, "wb") as buffer:
            buffer.write(file_content)
        update_document_status(document_id, "uploaded", "File uploaded successfully", file.filename)
        
        if process_immediately:
            active_processing_tasks[document_id] = True
            asyncio.create_task(
                process_document_async(
                    document_id=document_id,
                    file_path=UPLOAD_DIR,
                    file_name=file.filename
                )
            )
            status = "processing"
            message = "File uploaded and processing started in background."
        else:
            status = "uploaded"
            message = "File uploaded successfully (not yet processed)."
        
        return DocumentResponse(id=document_id, filename=file.filename, status=status, message=message)
    except Exception as e:
        print(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/processing-status")
async def get_processing_status():
    active_count = len(active_processing_tasks)
    all_status = {doc_id: get_document_status(doc_id) for doc_id in document_status}
    return {
        "active_tasks": active_count,
        "max_concurrent_tasks": processing_semaphore._value, # Actual current value
        "documents": all_status
    }

@app.post("/query", response_model=QueryResponse)
async def query(query_request: QueryRequest):
    global current_retriever, current_chain, last_refresh_time
    start_time = time.time()
    
    try:
        with retriever_lock:
            if current_retriever is None or current_chain is None:
                print("檢索器未初始化，正在刷新...")
                refresh_retriever()
            elif time.time() - last_refresh_time > refresh_interval:
                print(f"檢索器上次刷新時間超過 {refresh_interval} 秒，正在刷新...")
                refresh_retriever()
        
        print(f"🔍 處理查詢: {query_request.query}")
        
        with retriever_lock:
            if current_retriever and current_chain:
                # FAISS specific diagnostics
                if hasattr(current_retriever.vectorstore, 'index') and hasattr(current_retriever.vectorstore.index, 'ntotal'):
                    print(f"[FAISS Query] Index size: {current_retriever.vectorstore.index.ntotal} vectors.")
                else:
                    print("[FAISS Query] Vectorstore index details not available or not a standard FAISS index.")

                # Check docstore mapping
                if os.path.exists(DOCSTORE_MAPPING_PATH):
                    with open(DOCSTORE_MAPPING_PATH, 'r', encoding='utf-8') as f:
                        mapping_len = len(json.load(f))
                    print(f"[FAISS Query] Docstore mapping items: {mapping_len}")

                try:
                    # retrieve documents to see which docs are refereced
                    retrieved_docs = current_retriever.invoke(query_request.query)
                    print(f"📄 檢索到 {len(retrieved_docs)} 條相關文檔 (FAISS)")
                    # if retrieved_docs:
                    #     reference_parts = ["Retrieved document contents:"]
                    #     for i, doc in enumerate(retrieved_docs):
                    #         # 假設 doc.content 是文件的實際內容字串
                    #         # 您也可以考慮加入其他元數據，例如 doc.metadata.get('source', 'Unknown source')
                    #         reference_parts.append(f"  Document [{i+1}]: {doc['content']}") 
                    #     ref = "\n".join(reference_parts)
                    # else:
                    #     ref = "No documents were retrieved."

                    # retrieve documents and generate answer
                    answer = current_chain.invoke(query_request.query)
                    processing_time = time.time() - start_time
                    # return QueryResponse(answer=answer, processing_time=processing_time, reference=ref)
                    return QueryResponse(answer=answer, processing_time=processing_time)
                except Exception as retriever_error:
                    print(f"❌ 檢索器錯誤 (FAISS): {str(retriever_error)}")
                    return QueryResponse(
                        answer="很抱歉，FAISS 檢索系統遇到技術問題。請確保資料庫中有相關文檔，或嘗試重置系統後重新上傳文件。",
                        processing_time=time.time() - start_time
                    )
            else:
                raise HTTPException(status_code=500, detail="檢索器未初始化 (FAISS)，請稍後再試")
    except Exception as e:
        print(f"❌ 查詢處理錯誤 (FAISS): {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query (FAISS): {str(e)}")

@app.post("/reset")
async def reset_system():
    try:
        print("🔄 开始重置系统 (FAISS)...")
        with retriever_lock:
            # 1. Clear uploaded files
            try:
                upload_files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
                for file_name in upload_files:
                    os.remove(os.path.join(UPLOAD_DIR, file_name))
                print(f"✅ 已删除 {len(upload_files)} 个上传文件")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to clear uploaded files: {str(e)}")

            # 2. Clear figure directory
            try:
                if os.path.exists(FIGURE_PATH):
                    shutil.rmtree(FIGURE_PATH)
                    os.makedirs(FIGURE_PATH, exist_ok=True) # Recreate empty
                print(f"✅ 已清空图片文件夹")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to clear figure directory: {str(e)}")

            # 3. Delete FAISS index files
            try:
                if os.path.exists(FAISS_INDEX_PATH):
                    shutil.rmtree(FAISS_INDEX_PATH)
                    print(f"✅ 已删除 FAISS 索引目录: {FAISS_INDEX_PATH}")
                os.makedirs(FAISS_INDEX_PATH, exist_ok=True) # Recreate empty directory
                print(f"✅ 已重新创建空的 FAISS 索引目录")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to clear FAISS index: {str(e)}")

            # 4. Reset mapping file
            try:
                with open(DOCSTORE_MAPPING_PATH, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
                print(f"✅ 已重置映射文件")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to reset mapping file: {str(e)}")

            # 5. Clear status tracking
            document_status.clear()
            active_processing_tasks.clear()
            print(f"✅ 已清空状态追踪")

            # 6. Refresh retriever to initialize with empty/placeholder FAISS
            refresh_retriever()
            print(f"✅ 已完成检索器刷新 (FAISS)")
        
        print("✅ 系統重置完成 (FAISS)")
        return {
            "status": "success", 
            "message": "系统已重置 (FAISS)，FAISS 索引、映射文件和上传文件已清空",
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"❌ 系统重置过程中出错 (FAISS): {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"System reset failed (FAISS): {str(e)}")

# End of API routes

if __name__== "__main__":
    # Run the FAISS-based MM-RAG API
    uvicorn.run("app:app", host="0.0.0.0", port=1230, reload=True)

