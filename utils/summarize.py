import base64
import os

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser




def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    產生文字與表格的摘要。
    texts: 文字列表
    tables: 表格列表
    summarize_texts: 是否對文字進行摘要
    回傳：text_summaries（文字摘要）、table_summaries（表格摘要）
    """

    # 提示
    prompt_text = """您是一名助手，負責摘要表格和文本。 \
    這些摘要將被作為embedding並用於檢索原始文本或表格元素。 \
    請提供針對檢索優化的詳細摘要。表格或文本：{element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # 文本摘要鏈
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # 初始化摘要列表
    text_summaries = []
    table_summaries = []

    # 如果提供文本且需要摘要，則應用於文本
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 10})
    elif texts:
        text_summaries = texts

    # 如果提供表格，則應用於表格
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 10})

    return text_summaries, table_summaries

def encode_image(image_path):
    """
    將圖片檔案轉為 base64 字串。
    image_path: 圖片路徑
    回傳：base64 字串
    方便將圖片內容傳遞給多模態 LLM。
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_path, prompt):
    """
    產生圖片摘要。
    img_path: 圖片位置
    prompt: 摘要提示詞
    回傳：圖片摘要文字
    將圖片內容濃縮為可檢索的語意摘要。
    """
    chat = ChatOpenAI(model="gpt-4o", max_tokens=4096)

    img_base64 = encode_image(img_path)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """
    產生資料夾內所有圖片的摘要與檔名。
    path: 圖片資料夾路徑
    回傳：image_summaries（摘要列表）、img_filenames（檔名列表）
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # 存儲圖像摘要
    image_summaries = []

    # 存儲圖像檔名
    img_filenames = []

    if not any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(path)):
        print("⚠️  資料夾中沒有可處理的圖片檔案。")
        return image_summaries, img_filenames

    # 提示
    prompt = """你是一個專門為向量檢索系統服務的圖片摘要生成器，請根據每張圖片的視覺內容生成精確、具資訊密度的文字描述。請注意以下規則：
	1.	你的目標是產生可以轉換成語意向量的摘要文字，供後續檢索使用。
	2.	僅根據圖片中可見資訊進行描述，嚴禁猜測或補充無法從畫面確認的細節。
	3.	不得產生與圖片無關的評論、判斷、推測或背景資訊。
	4.	輸出格式為單段文字摘要，不加入多餘的格式或解釋。"""

    # 應用於圖像
    for img_file in sorted(os.listdir(path)):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path, img_file)
            try:
                image_summaries.append(image_summarize(img_path, prompt))
                img_filenames.append(img_file)
            except Exception as e:
                print(f"❌ Error processing image {img_path}: {e}")

    return image_summaries, img_filenames


if __name__ == "__main__":
    print("Wrong file execution. Please use the build_vector_db.py to build up the vector database. Then use the main.py to run the mm-RAG.")