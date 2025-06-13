# uvicorn app:app --host 0.0.0.0 --port 1230 --reload
import os

import dotenv

dotenv.load_dotenv()
from utils.logging_config import get_logger

logger = get_logger(__name__)
import asyncio
import json
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import uvicorn
from fastapi import Cookie, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import the RAG functionality from main.py
from main import multi_modal_rag_chain

# Import required utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import NLTKTextSplitter  # Ensure NLTK data is downloaded

from utils.LLM_Tool import text_embedding_3_large

# Create FastAPI app
app = FastAPI(
    title="MM-RAG API (FAISS Version)",
    description="Multimodal Retrieval Augmented Generation API using FAISS",
    version="0.1.1",  # Incremented version
    root_path="/mm_rag",  # Changed to match the new directory structure
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/web", StaticFiles(directory="frontend", html=True), name="frontend")

# Create necessary root directories
DATABASE_ROOT = "./database"
UPLOAD_ROOT = "./uploads"
FAISS_INDEX_NAME = "mm_rag_faiss_index"  # Name for FAISS index files


def get_user_paths(username: str):
    base_dir = os.path.join(DATABASE_ROOT, username)
    return {
        "database_dir": base_dir,
        "faiss_index_path": os.path.join(base_dir, "faiss_store"),
        "figure_path": os.path.join(base_dir, "figures"),
        "docstore_mapping_path": os.path.join(base_dir, "docstore_mapping.json"),
        "upload_dir": os.path.join(UPLOAD_ROOT, username),
    }


# User management
users = {}
user_sessions = {}

# Simple in-memory status tracking
document_status = {}

# Global variables for the retriever and refresh logic
retriever_lock = threading.RLock()
refresh_interval = 15
user_retrievers = {}

process_pool = ProcessPoolExecutor(max_workers=4)
processing_semaphore = asyncio.Semaphore(4)
active_processing_tasks = {}


# Pydantic models (remain the same)
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class LoginRequest(BaseModel):
    username: str
    password: str


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
def refresh_retriever(username: str):
    global user_retrievers
    user_paths = get_user_paths(username)

    with retriever_lock:
        logger.info("🔄 刷新檢索器和 RAG 鏈 (FAISS 版本)...")
        start_time = time.time()

        try:
            vectorstore = None
            faiss_actual_index_file = os.path.join(
                user_paths["faiss_index_path"], f"{FAISS_INDEX_NAME}.faiss"
            )

            if os.path.exists(faiss_actual_index_file):
                try:
                    vectorstore = FAISS.load_local(
                        user_paths["faiss_index_path"],
                        text_embedding_3_large,
                        FAISS_INDEX_NAME,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info(
                        f"[FAISS] Loaded existing FAISS index from {user_paths['faiss_index_path']}/{FAISS_INDEX_NAME}"
                    )
                    if hasattr(vectorstore.index, "ntotal"):
                        logger.info(
                            f"[FAISS] Index size: {vectorstore.index.ntotal} vectors"
                        )
                except Exception as load_err:
                    logger.warning(
                        f"⚠️ [FAISS] Error loading index: {load_err}. Will attempt to create a new one."
                    )
                    vectorstore = None  # Ensure it's None so a new one is created

            if vectorstore is None:  # If not loaded or load failed
                os.makedirs(user_paths["faiss_index_path"], exist_ok=True)
                # Create a dummy FAISS index so that MultiVectorRetriever can be initialized
                initial_doc_texts = ["placeholder document for faiss initialization"]
                vectorstore = FAISS.from_texts(
                    initial_doc_texts, text_embedding_3_large
                )
                vectorstore.save_local(user_paths["faiss_index_path"], FAISS_INDEX_NAME)
                logger.info(
                    f"[FAISS] Initialized and saved a placeholder FAISS index at {user_paths['faiss_index_path']}/{FAISS_INDEX_NAME}"
                )

            if os.path.exists(user_paths["docstore_mapping_path"]):
                with open(
                    user_paths["docstore_mapping_path"], "r", encoding="utf-8"
                ) as f:
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

            chain = multi_modal_rag_chain(retriever, user_paths["figure_path"])

            user_retrievers[username] = {
                "retriever": retriever,
                "chain": chain,
                "last_refresh": time.time(),
            }

            refresh_time = time.time() - start_time
            logger.info(f"✅ 檢索器刷新完成 (FAISS) (耗時: {refresh_time:.2f} 秒)")
            logger.info(f"📊 文檔映射條目數: {len(loaded_mapping)}")

            return True
        except Exception as e:
            logger.error(f"❌ 檢索器刷新失敗 (FAISS): {str(e)}")
            import traceback

            traceback.print_exc()
            return False


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 初始化 MM-RAG 系統 (FAISS 版本)...")

    os.makedirs(DATABASE_ROOT, exist_ok=True)
    os.makedirs(UPLOAD_ROOT, exist_ok=True)

    # Load users from users.json if exists
    users_path = os.path.join(DATABASE_ROOT, "users.json")
    if os.path.exists(users_path):
        try:
            with open(users_path, "r", encoding="utf-8") as f:
                users.update(json.load(f))
        except Exception as e:
            logger.warning(f"⚠️ 無法載入使用者檔案: {e}")

    logger.info("✅ MM-RAG 系統 (FAISS) 初始化完成，準備接收請求")


from fastapi import Response


@app.post("/login")
async def login(credentials: LoginRequest, response: Response):
    user_info = users.get(credentials.username)
    if not user_info or user_info.get("password") != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = str(uuid.uuid4())
    user_sessions[token] = credentials.username

    paths = get_user_paths(credentials.username)
    for p in [
        paths["database_dir"],
        paths["faiss_index_path"],
        paths["figure_path"],
        paths["upload_dir"],
    ]:
        os.makedirs(p, exist_ok=True)
    if not os.path.exists(paths["docstore_mapping_path"]):
        with open(paths["docstore_mapping_path"], "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    refresh_retriever(credentials.username)
    response.set_cookie(key="token", value=token, httponly=False, path="/")
    return {"token": token}


# Modified for FAISS
def add_file_to_vector_db(file_path: str, username: str) -> bool:
    from utils.processing import process_single_file
    from utils.summarize import generate_img_summaries, generate_text_summaries
    from utils.vector_store import (
        create_multi_vector_retriever,
    )  # This util needs to be FAISS-aware

    logger.info(f"\n🔄 處理上傳檔案 (FAISS): {file_path}")

    try:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext not in [".pdf", ".pptx", ".ppt", ".docx"]:
            logger.error(f"❌ 不支援的檔案類型: {file_ext}")
            return False

        import tempfile

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file_name)
        shutil.copy2(file_path, temp_file_path)

        user_paths = get_user_paths(username)
        result = process_single_file(
            file_name, temp_dir, database_dir=user_paths["database_dir"]
        )  # Assumes this util is generic

        if not result:
            logger.error(f"❌ 檔案處理失敗: {file_name}")
            shutil.rmtree(temp_dir)
            return False

        texts = result["texts"]
        tables = result["tables"]
        page_summaries = result["page_summaries"]
        page_identifiers = result["page_identifiers"]

        joined_texts = " ".join(texts)
        text_splitter = NLTKTextSplitter()
        texts_4k_token = text_splitter.split_text(joined_texts) if joined_texts else []

        text_summaries, table_summaries = generate_text_summaries(
            texts_4k_token, tables, summarize_texts=True
        )
        image_summaries, img_filenames = generate_img_summaries(
            user_paths["figure_path"]
        )  # Assumes figure path is managed correctly

        faiss_actual_index_file = os.path.join(
            user_paths["faiss_index_path"], f"{FAISS_INDEX_NAME}.faiss"
        )
        is_new_db = not os.path.exists(faiss_actual_index_file)

        vectorstore = None
        if not is_new_db:
            try:
                vectorstore = FAISS.load_local(
                    user_paths["faiss_index_path"],
                    text_embedding_3_large,
                    FAISS_INDEX_NAME,
                    allow_dangerous_deserialization=True,
                )
                logger.info("[FAISS] Loaded existing FAISS index for update.")
            except Exception as e:
                logger.warning(
                    f"⚠️ [FAISS] Could not load existing index for update: {e}. Will create new."
                )
                is_new_db = True  # Force creation
                vectorstore = None

        _ = create_multi_vector_retriever(
            vectorstore=vectorstore,  # Pass the loaded store, or None if new
            text_summaries=text_summaries,
            text_contents=texts_4k_token,
            table_summaries=table_summaries,
            table_contents=tables,
            image_summaries=image_summaries,
            image_filenames=img_filenames,  # Changed from img_filenames and added comma
            page_summaries=page_summaries,
            page_identifiers=page_identifiers,
            docstore_path=user_paths["docstore_mapping_path"],
            update_existing=not is_new_db,
            # FAISS specific parameters that create_multi_vector_retriever might need:
            vectorstore_type="faiss",  # Explicitly tell the util
            faiss_index_path=user_paths["faiss_index_path"],
            faiss_index_name=FAISS_INDEX_NAME,
            embedding_function=text_embedding_3_large,
        )

        shutil.rmtree(temp_dir)

        logger.info(f"✅ 檔案 {file_name} 成功添加到向量資料庫 (FAISS)")
        return True

    except Exception as e:
        logger.error(f"❌ 處理檔案時發生錯誤 (FAISS): {str(e)}")
        import traceback

        traceback.print_exc()
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


# Helper to list uploaded files for a user
def list_user_files(username: str) -> list[str]:
    """Return a list of filenames a user has uploaded."""
    user_paths = get_user_paths(username)
    if not os.path.exists(user_paths["upload_dir"]):
        return []
    return [
        f
        for f in os.listdir(user_paths["upload_dir"])
        if os.path.isfile(os.path.join(user_paths["upload_dir"], f))
    ]


# New helper to delete all vector and docstore entries for a given file
def delete_file_from_vector_db(file_name: str, username: str) -> bool:
    from utils.vector_store import FAISS

    user_paths = get_user_paths(username)
    try:
        doc_ids_to_remove = []
        if os.path.exists(user_paths["docstore_mapping_path"]):
            with open(user_paths["docstore_mapping_path"], "r", encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            mapping = {}

        for doc_id, info in list(mapping.items()):
            data_str = json.dumps(info)
            if file_name in data_str:
                doc_ids_to_remove.append(doc_id)
                mapping.pop(doc_id)

        with open(user_paths["docstore_mapping_path"], "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)

        faiss_file = os.path.join(
            user_paths["faiss_index_path"], f"{FAISS_INDEX_NAME}.faiss"
        )
        if os.path.exists(faiss_file):
            vs = FAISS.load_local(
                user_paths["faiss_index_path"],
                text_embedding_3_large,
                FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            if hasattr(vs, "delete"):
                try:
                    vs.delete(doc_ids_to_remove)
                except Exception:
                    try:
                        vs.delete(ids=doc_ids_to_remove)
                    except Exception as e:
                        logger.error(f"Delete failed: {e}")
            vs.save_local(user_paths["faiss_index_path"], FAISS_INDEX_NAME)

        fig_dir = os.path.join(
            user_paths["figure_path"], os.path.splitext(file_name)[0]
        )
        if os.path.exists(fig_dir):
            shutil.rmtree(fig_dir)

        upload_file = os.path.join(user_paths["upload_dir"], file_name)
        if os.path.exists(upload_file):
            os.remove(upload_file)

        return True
    except Exception as e:
        logger.error(f"Failed to delete {file_name}: {e}")
        return False


# Helper functions (remain the same)
def update_document_status(
    document_id: str,
    username: str,
    status: str,
    message: str = "",
    filename: str | None = None,
):
    document_status[document_id] = {
        "status": status,
        "message": message,
        "filename": filename,
        "user": username,
    }


def get_document_status(document_id: str):
    return document_status.get(
        document_id,
        {
            "status": "unknown",
            "message": "Document not found",
            "filename": "unknown.pdf",
        },
    )


# Subprocess and async handling (remain largely the same)
def process_file_in_subprocess(args) -> bool:
    file_path, username = args
    try:
        return add_file_to_vector_db(file_path, username)  # Calls the FAISS version
    except Exception as e:
        logger.error(f"❌ processing file in subprocess: {str(e)}")
        return False


async def process_document_async(
    document_id: str, file_path: str, file_name: str, username: str
):
    async with processing_semaphore:
        try:
            update_document_status(
                document_id, username, "processing", "Processing document...", file_name
            )
            logger.info(f"🔄 file processing for {file_name} (ID: {document_id})")

            full_file_path = os.path.join(file_path, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext not in [".pdf", ".pptx", ".ppt", ".docx", ".txt"]:
                update_document_status(
                    document_id,
                    username,
                    "failed",
                    f"Unsupported file extension: {file_ext}",
                    file_name,
                )
                return

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                process_pool, process_file_in_subprocess, (full_file_path, username)
            )

            if success:
                update_document_status(
                    document_id,
                    username,
                    "completed",
                    "Document processed successfully",
                    file_name,
                )
                logger.info(
                    f"✅ file processed successfully {file_name} (ID: {document_id})"
                )
                logger.info("🔄 file processing completed, refreshing retriever")
                refresh_success = refresh_retriever(username)
                if refresh_success:
                    logger.info("✅ retriever refreshed successfully")
                else:
                    logger.warning("⚠️ retriever refresh failed, please check logs")
            else:
                update_document_status(
                    document_id,
                    username,
                    "failed",
                    "Document processing failed",
                    file_name,
                )
                logger.error(
                    f"❌ document processing for {file_name} failed unexpectedly (ID: {document_id}) "
                )

        except Exception as e:
            logger.error(f"❌ file processing error: {str(e)}")
            update_document_status(
                document_id,
                username,
                "failed",
                f"file processed failed: {str(e)}",
                file_name,
            )

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
    process_immediately: bool = Form(True),
    authorization: str | None = Header(None),
    token: str | None = Cookie(None),
):
    if not authorization:
        authorization = token
    if not authorization or authorization not in user_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    username = user_sessions[authorization]
    user_paths = get_user_paths(username)

    document_id = str(uuid.uuid4())
    os.makedirs(user_paths["upload_dir"], exist_ok=True)
    file_location = os.path.join(user_paths["upload_dir"], file.filename)

    try:
        file_content = await file.read()
        with open(file_location, "wb") as buffer:
            buffer.write(file_content)
        update_document_status(
            document_id,
            username,
            "uploaded",
            "File uploaded successfully",
            file.filename,
        )

        if process_immediately:
            active_processing_tasks[document_id] = True
            asyncio.create_task(
                process_document_async(
                    document_id=document_id,
                    file_path=user_paths["upload_dir"],
                    file_name=file.filename,
                    username=username,
                )
            )
            status = "processing"
            message = "File uploaded and processing started in background."
        else:
            status = "uploaded"
            message = "File uploaded successfully (not yet processed)."

        return DocumentResponse(
            id=document_id, filename=file.filename, status=status, message=message
        )
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/processing-status")
async def get_processing_status(
    authorization: str | None = Header(None), token: str | None = Cookie(None)
):
    if not authorization:
        authorization = token
    if not authorization or authorization not in user_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    username = user_sessions[authorization]
    active_count = len(active_processing_tasks)
    all_status = {
        doc_id: get_document_status(doc_id)
        for doc_id, data in document_status.items()
        if data.get("user") == username
    }
    return {
        "active_tasks": active_count,
        "max_concurrent_tasks": processing_semaphore._value,  # Actual current value
        "documents": all_status,
    }


@app.post("/query", response_model=QueryResponse)
async def query(
    query_request: QueryRequest,
    authorization: str | None = Header(None),
    token: str | None = Cookie(None),
):
    if not authorization:
        authorization = token
    if not authorization or authorization not in user_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")

    username = user_sessions[authorization]
    start_time = time.time()

    try:
        with retriever_lock:
            user_data = user_retrievers.get(username)
            if (not user_data) or (
                time.time() - user_data["last_refresh"] > refresh_interval
            ):
                logger.info("檢索器未初始化或已過期，正在刷新...")
                refresh_retriever(username)
                user_data = user_retrievers.get(username)

        logger.info(f"🔍 處理查詢: {query_request.query}")

        with retriever_lock:
            if user_data:
                current_retriever = user_data["retriever"]
                current_chain = user_data["chain"]
                # FAISS specific diagnostics
                if hasattr(current_retriever.vectorstore, "index") and hasattr(
                    current_retriever.vectorstore.index, "ntotal"
                ):
                    logger.info(
                        f"[FAISS Query] Index size: {current_retriever.vectorstore.index.ntotal} vectors."
                    )
                else:
                    logger.info(
                        "[FAISS Query] Vectorstore index details not available or not a standard FAISS index."
                    )

                # Check docstore mapping
                user_paths = get_user_paths(username)
                if os.path.exists(user_paths["docstore_mapping_path"]):
                    with open(
                        user_paths["docstore_mapping_path"], "r", encoding="utf-8"
                    ) as f:
                        mapping_len = len(json.load(f))
                    logger.info(f"[FAISS Query] Docstore mapping items: {mapping_len}")

                try:
                    # retrieve documents to see which docs are refereced
                    retrieved_docs = current_retriever.invoke(query_request.query)
                    logger.info(f"📄 檢索到 {len(retrieved_docs)} 條相關文檔 (FAISS)")
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
                    logger.error(f"❌ 檢索器錯誤 (FAISS): {str(retriever_error)}")
                    return QueryResponse(
                        answer="很抱歉，FAISS 檢索系統遇到技術問題。請確保資料庫中有相關文檔，或嘗試重置系統後重新上傳文件。",
                        processing_time=time.time() - start_time,
                    )
            else:
                raise HTTPException(
                    status_code=500, detail="檢索器未初始化 (FAISS)，請稍後再試"
                )
    except Exception as e:
        logger.error(f"❌ 查詢處理錯誤 (FAISS): {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing query (FAISS): {str(e)}"
        )


@app.post("/reset")
async def reset_system(
    authorization: str | None = Header(None), token: str | None = Cookie(None)
):
    if not authorization:
        authorization = token
    if not authorization or authorization not in user_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    username = user_sessions[authorization]
    user_paths = get_user_paths(username)
    try:
        logger.info("🔄 开始重置系统 (FAISS)...")
        with retriever_lock:
            # 1. Clear uploaded files
            try:
                if os.path.exists(user_paths["upload_dir"]):
                    upload_files = [
                        f
                        for f in os.listdir(user_paths["upload_dir"])
                        if os.path.isfile(os.path.join(user_paths["upload_dir"], f))
                    ]
                    for file_name in upload_files:
                        os.remove(os.path.join(user_paths["upload_dir"], file_name))
                logger.info(
                    f"✅ 已删除 {len(upload_files) if 'upload_files' in locals() else 0} 个上传文件"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to clear uploaded files: {str(e)}"
                )

            # 2. Clear figure directory
            try:
                if os.path.exists(user_paths["figure_path"]):
                    shutil.rmtree(user_paths["figure_path"])
                os.makedirs(user_paths["figure_path"], exist_ok=True)
                logger.info("✅ 已清空图片文件夹")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to clear figure directory: {str(e)}",
                )

            # 3. Delete FAISS index files
            try:
                if os.path.exists(user_paths["faiss_index_path"]):
                    shutil.rmtree(user_paths["faiss_index_path"])
                    logger.info(
                        f"✅ 已删除 FAISS 索引目录: {user_paths['faiss_index_path']}"
                    )
                os.makedirs(user_paths["faiss_index_path"], exist_ok=True)
                logger.info("✅ 已重新创建空的 FAISS 索引目录")
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to clear FAISS index: {str(e)}"
                )

            # 4. Reset mapping file
            try:
                with open(
                    user_paths["docstore_mapping_path"], "w", encoding="utf-8"
                ) as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
                logger.info("✅ 已重置映射文件")
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to reset mapping file: {str(e)}"
                )

            # 5. Clear status tracking
            document_status.clear()
            active_processing_tasks.clear()
            logger.info("✅ 已清空状态追踪")

            # 6. Refresh retriever to initialize with empty/placeholder FAISS
            refresh_retriever(username)
            logger.info("✅ 已完成检索器刷新 (FAISS)")

        logger.info("✅ 系統重置完成 (FAISS)")
        return {
            "status": "success",
            "message": "系统已重置 (FAISS)，FAISS 索引、映射文件和上传文件已清空",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"❌ 系统重置过程中出错 (FAISS): {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"System reset failed (FAISS): {str(e)}"
        )


@app.get("/files")
async def get_uploaded_files(
    authorization: str | None = Header(None), token: str | None = Cookie(None)
):
    """Return a list of filenames uploaded by the current user."""
    if not authorization:
        authorization = token
    if not authorization or authorization not in user_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    username = user_sessions[authorization]
    return {"files": list_user_files(username)}


@app.post("/delete")
async def delete_document(
    file_name: str = Form(...),
    authorization: str | None = Header(None),
    token: str | None = Cookie(None),
):
    """Delete a single document from the user's vector DB."""
    if not authorization:
        authorization = token
    if not authorization or authorization not in user_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    username = user_sessions[authorization]
    with retriever_lock:
        success = delete_file_from_vector_db(file_name, username)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete file")
        refresh_retriever(username)
    return {"status": "success", "message": f"{file_name} deleted"}


# End of API routes

if __name__ == "__main__":
    # Run the FAISS-based MM-RAG API
    uvicorn.run("app:app", host="0.0.0.0", port=1230, reload=True)
