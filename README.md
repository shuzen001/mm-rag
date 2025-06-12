# 📄 多模態 RAG 系統 (MM-RAG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本專案是一個基於 FastAPI、LangChain 與 FAISS 的多模態檢索增強生成（MM-RAG）系統。支援 PDF、PPTX、DOCX 等文件自動抽取文字、表格、圖片，並以 GPT-4o 進行摘要，所有內容向量化後存入 FAISS，支援語意檢索與問答。

---

## ✨ 主要功能

- **多模態文件處理**：自動解析 PDF、PPTX、DOCX，抽取文字、表格、圖片。
- **智慧內容摘要**：利用 GPT-4o 為各元素生成精簡摘要。
- **多向量檢索**：摘要向量儲存於 FAISS，支援語意搜尋。
- **多模態問答生成**：組合圖片、表格、文字上下文，送入 GPT-4o 回答問題。
- **API 介面**：提供檔案上傳、查詢、重置、狀態查詢等 RESTful API。
- **模組化設計**：功能拆分為多個獨立 Python 檔案，方便維護與擴展。

---

## 📁 專案結構

```text

mm-rag/
├── app.py                  # FastAPI 主程式，處理檔案上傳與向量庫建構
├── main.py                 # 問答流程主程式（可單機測試）
├── requirements.txt        # 套件依賴清單
├── .env                    # 環境變數檔（需自行建立）
├── .gitignore              # Git 忽略規則設定
├── README.md               # 本文件
├── database/               # 向量庫、圖片、docstore
│   ├── faiss_store/        # FAISS 向量資料庫
│   ├── figures/            # 圖片存放
│   └── docstore_mapping.json # Docstore 映射
├── files/                  # 待處理的原始文件
├── uploads/                # 上傳暫存區
└── utils/                  # 輔助工具腳本
    ├── extract_file_utils.py
    ├── summarize.py
    └── vector_store.py
```

---

## ⚙️ 環境準備

1. 安裝 Python 3.10+
2. 安裝 `poppler-utils`、`tesseract-ocr` 與 `libreoffice`：
   ```bash
   # 以 Ubuntu 為例
   sudo apt-get install poppler-utils tesseract-ocr libreoffice
   ```
3. 建立虛擬環境（建議）：
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```
4. 安裝依賴：
    ```bash
    pip install -r requirements.txt
    ```
5. 設定 OpenAI API 金鑰：
    在專案根目錄下建立 `.env` 檔案，內容如下：
    ```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 🧪 單元測試

執行所有測試：

```bash
pytest -q
```

若要以真實 PDF 執行 `test_convert_file_to_markdown_real_pdf`，請先安裝 `docling`
相關依賴，並設定環境變數 `DOC_TEST_PDF` 為你的 PDF 路徑：

```bash
DOC_TEST_PDF=/path/to/your.pdf pytest -q tests/test_docling_markdown.py::test_convert_file_to_markdown_real_pdf
```

未設定此變數時，整合測試會自動被跳過。

---

## 🚀 啟動與使用流程

### 1️⃣ 啟動 API 服務

```bash
uvicorn app:app --host 0.0.0.0 --port 1230 --reload
```

### 2️⃣ 上傳文件

- 透過 `/mm_rag/upload` API 上傳 PDF、PPTX、DOCX 檔案，支援背景處理。
- 上傳後自動抽取內容、摘要並存入 FAISS。

### 3️⃣ 查詢問答

- 使用 `/mm_rag/query` API 提問，系統會檢索最相關內容並組合多模態上下文給 GPT-4o 回答。

### 4️⃣ 查詢處理狀態

- `/mm_rag/processing-status` 可查詢所有文件處理進度。

### 5️⃣ 重置系統

- `/mm_rag/reset` API 可一鍵清空所有上傳文件、FAISS 向量庫、docstore 映射。

### 6️⃣ 使用網頁介面

1. 啟動 API 後，前往 `http://localhost:1230/mm_rag/web/` 會看到以 `react-login-page` 風格打造的登入畫面。
2. 依照 `database/users.json` 中的帳號密碼登入。
3. 登入後可在頁面上傳文件、查看處理狀態，QA 區域會以 iframe 嵌入 `http://localhost/chatbot/zDAZ0GYT5OhdjSuD`。
4. 所有操作均會與該使用者的專屬資料庫同步。

---

## 🛠️ API 路由說明

### 1. 上傳文件
- `PUT /mm_rag/upload`
- 參數：`file` (UploadFile)，`process_immediately` (bool, 預設 True)
- 回傳：文件 ID、檔名、狀態、訊息

### 2. 查詢問答
- `POST /mm_rag/query`
- 參數：`query` (str)，`top_k` (int, 預設 5)
- 回傳：`answer` (str)，`processing_time` (float)

### 3. 查詢處理狀態
- `GET /mm_rag/processing-status`
- 回傳：所有文件的處理狀態、進度

### 4. 重置系統
- `POST /mm_rag/reset`
- 回傳：重置狀態、訊息、時間戳

---

## 📦 API 範例（以 Python requests 示意）

```python
import requests

# 上傳文件
with open('files/your.pdf', 'rb') as f:
    res = requests.put('http://localhost:1230/mm_rag/upload', files={'file': f})
    print(res.json())

# 查詢問答
payload = {"query": "請問本文件的重點？"}
res = requests.post('http://localhost:1230/mm_rag/query', json=payload)
print(res.json())

# 查詢處理狀態
res = requests.get('http://localhost:1230/mm_rag/processing-status')
print(res.json())

# 重置系統
res = requests.post('http://localhost:1230/mm_rag/reset')
print(res.json())
```

---

## 🔧 可擴展性

- **模型選擇**：可修改 `utils/LLM_Tool.py` 內的模型設定
- **提示詞調整**：可自定義 `summarize.py` 內摘要提示詞
- **嵌入模型替換**：可改為 HuggingFace、本地嵌入器
- **儲存策略**：可擴充 metadata、支援更多向量庫
- **API 路由**：可依需求擴充更多檢索/分析功能

---

## 📜 授權條款

本專案採用 [MIT License](https://opensource.org/licenses/MIT) 授權。