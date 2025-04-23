import uuid
import json # Added import
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
import io
import re
from IPython.display import HTML, display
from PIL import Image
import base64



def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, img_filenames, # Added img_filenames
    docstore_path="./docstore_mapping.json"
):
    """
    建立多向量檢索器，摘要存向量庫，原始內容與檔名存 docstore，並保存映射。 # Modified docstring
    vectorstore: 向量庫
    text_summaries: 文字摘要
    texts: 原始文字
    table_summaries: 表格摘要
    tables: 原始表格
    image_summaries: 圖片摘要
    img_filenames: 原始圖片檔名列表 # Added docstring
    docstore_path: 儲存 doc_id 到內容映射的檔案路徑
    回傳：retriever（多向量檢索器）
    """

    # 初始化文件存儲層，使用記憶體存儲
    store = InMemoryStore()
    # 定義文件唯一識別鍵名稱
    id_key = "doc_id"

    # 建立多向量檢索器，結合向量庫與文件存儲層
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 用來儲存所有 id -> content 的映射
    all_mappings = {}

    # 定義輔助函式，將摘要與原始內容加入向量庫與文件存儲
    def add_documents(retriever, doc_summaries, doc_contents, doc_filenames=None): # Added doc_filenames parameter
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
        retriever.vectorstore.add_documents(summary_docs)

        # Prepare content for docstore
        if doc_filenames: # If filenames are provided (for images)
            # Store a dictionary with content and filename
            store_contents = [{'filename': doc_contents[i], 'summary': doc_summaries[i]}
                              for i in range(len(doc_contents))]
        else: # For text and tables, store content directly
            store_contents = doc_contents

        # 將原始內容以 doc_id 為鍵，存入文件存儲層
        mappings = list(zip(doc_ids, store_contents))
        retriever.docstore.mset(mappings)
        # 更新總映射
        all_mappings.update(dict(mappings))


    if text_summaries and texts: # Ensure both exist
        add_documents(retriever, text_summaries, texts)

    if table_summaries and tables: # Ensure both exist
        add_documents(retriever, table_summaries, tables)

    if image_summaries and img_filenames: # Ensure all exist for images
        # Pass filenames when adding image documents
        add_documents(retriever, image_summaries, img_filenames, img_filenames) # Pass img_filenames two times to be classified as image file

    # --- Added: Save the mapping to a file ---
    try:
        with open(docstore_path, 'w', encoding='utf-8') as f:
            json.dump(all_mappings, f, ensure_ascii=False, indent=4)
        print(f"✅ Docstore mapping saved to {docstore_path}")
    except Exception as e:
        print(f"❌ Error saving docstore mapping: {e}")
    # --- End Added ---

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
    print("Wrong file execution. Please use the build_vector_db.py to build up the vector database. Then use the main.py to run the mm-RAG.")