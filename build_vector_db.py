# build_vector_db.py
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from utils.extract_file_utils import extract_pdf_elements, categorize_elements
from utils.summarize import generate_img_summaries, generate_text_summaries
from utils.vector_store import create_multi_vector_retriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

fpath = "files/"
FIGURES_PATH = "figures/"

if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

# 處理所有 PDF 檔案
all_texts = []
all_tables = []
for fname in os.listdir(fpath):
    if fname != ".gitkeep" and fname.lower().endswith(".pdf"):
        raw_pdf_elements = extract_pdf_elements(fpath, fname, FIGURES_PATH)
        texts, tables = categorize_elements(raw_pdf_elements)
        all_texts.extend(texts)
        all_tables.extend(tables)

joined_texts = " ".join(all_texts)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=0)
texts_4k_token = text_splitter.split_text(joined_texts)
text_summaries, table_summaries = generate_text_summaries(texts_4k_token, all_tables, summarize_texts=True)

# 圖片摘要
# Modified to capture filenames
image_summaries, img_filenames = generate_img_summaries(FIGURES_PATH)

# 建立 vector store 並儲存
vectorstore = Chroma(
    collection_name="mm_rag",
    embedding_function=OpenAIEmbeddings(api_key=api_key),
    persist_directory="./chroma_store"
)

# Modified to pass img_filenames and specify docstore path
_ = create_multi_vector_retriever(
    vectorstore,
    text_summaries, 
    texts_4k_token, 
    table_summaries, 
    all_tables,
    image_summaries,
    img_filenames, # Pass the filenames
    docstore_path="./docstore_mapping.json" # Specify path to save mapping
)

print("✅ Vector database 建立完成並儲存！")
