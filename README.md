# 📄 多模態 RAG 系統 (MM-RAG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

這是一個基於 LangChain 和 ChromaDB 的多模態檢索增強生成 (Multi-Modal Retrieval-Augmented Generation, MM-RAG) 系統。它能夠處理包含文字、表格和圖像的 PDF 文件，並根據這些文件的內容回答使用者的問題。

---

## ✨ 主要功能

- **多模態文件處理**：使用 `unstructured` 套件解析 PDF，提取文字、表格、圖像。
- **智慧內容摘要**：利用 GPT-4o 為各元素生成精簡且資訊密度高的摘要。
- **多向量檢索策略**：
  - 將摘要轉為向量儲存於 ChromaDB，用於語意搜尋。
  - 將原始內容與圖片參照（含檔名與摘要）儲存在 Docstore，並透過 ID 與向量連結。
- **多模態問答生成**：組合圖片、表格、文字上下文，送入 GPT-4o 回答問題。
- **環境變數管理**：使用 `.env` 檔案安全管理 OpenAI API 金鑰。
- **模組化設計**：功能拆分為多個獨立 Python 檔案，方便維護與擴展。

---

## 📁 專案結構

```text
mm-rag/
├── files/                  # 放置待處理的 PDF 文件
│   └── .gitkeep
├── utils/                  # 輔助工具腳本 (New Folder)
│   ├── extract_file_utils.py
│   ├── summarize.py
│   └── vector_store.py
├── .env                    # 環境變數檔（需自行建立）
├── .gitignore              # Git 忽略規則設定
├── requirements.txt        # 套件依賴清單
├── build_vector_db.py      # 建立摘要與向量資料庫
├── main.py                 # 問答流程主程式
└── README.md               # 本文件
```

---

## ⚙️ 環境準備

1. 安裝 Python 3.10+
2. 安裝 `poppler-utils`（Linux/macOS 解析 PDF 時可能需要）
3. 建立虛擬環境（建議）：
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```

---

## 🚀 安裝與設定

1. **安裝依賴套件**：
    ```bash
    pip install -r requirements.txt
    ```

2. **設定 API 金鑰**：
    在專案根目錄下建立 `.env` 檔案並填入以下內容：
    ```dotenv
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

3. **建議 .gitignore 內容**：
    ```gitignore
    .env
    venv/
    __pycache__/
    *.pyc
    .vscode/
    chroma_store/
    figures/
    docstore_mapping.json
    ```

---

## 📖 使用說明

### 1️⃣ 放入 PDF 檔案
將你的 PDF 放入 `files/` 資料夾。

### 2️⃣ 建立向量資料庫
執行以下指令：
```bash
python build_vector_db.py
```

此腳本會：
- 提取 PDF 中的文字、表格與圖片
- 使用 GPT-4o 為各元素產生摘要
- 儲存向量到 ChromaDB（`chroma_store/`）
- 儲存對應關係至 `docstore_mapping.json`

### 3️⃣ 啟動 RAG 問答流程
```bash
python main.py
```

你可以在 `main.py` 修改 `query = "你的問題"` 來進行不同查詢。

---

## 💡 工作流程詳解

1. **解析 PDF**
    - 使用 `unstructured` 解析出文字、表格與圖像。
    - 圖片會存到 `figures/` 資料夾。

2. **摘要生成**
    - `summarize.py` 會針對文字/表格/圖片產生摘要。
    - GPT-4o 處理圖片時使用 Base64 格式。

3. **向量嵌入與資料儲存**
    - 使用 `OpenAIEmbeddings` 將摘要轉換為向量。
    - 儲存於 ChromaDB，並用 `InMemoryStore` 建立 Docstore。

4. **問答流程**
    - `main.py` 讀取向量資料庫與 Docstore。
    - 使用 `MultiVectorRetriever` 檢索最相關內容。
    - 將文字、表格、圖片組合後送入 GPT-4o 回答。

---

## 🔧 可客製化項目

- **模型選擇**：可修改 `ChatOpenAI` 的 `model` 參數使用不同模型
- **提示詞調整**：可自定義 `img_prompt_func` 或摘要提示詞
- **嵌入模型替換**：可改為 HuggingFace 模型或本地嵌入器
- **圖像處理邏輯**：可調整圖像尺寸、Base64 編碼邏輯
- **儲存與檢索策略**：可更換向量資料庫、增加 metadata 儲存

---

## 📜 授權條款

本專案採用 [MIT License](https://opensource.org/licenses/MIT) 授權。