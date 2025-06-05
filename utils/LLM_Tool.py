import os 
from dotenv import load_dotenv
load_dotenv()
# API key will be loaded from .env file

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage

def gpt_4o():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        streaming=True
    )
    return llm

def gpt_4o_for_summary():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        streaming=True
    )
    return llm

text_embedding_3_large = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

def test_embedding() -> None:
    """測試 Azure Embedding 部署"""
    test_text = "Hello, Azure embeddings!"
    try:
        embedding = text_embedding_3_large.embed_query(test_text)
        print("\n[Embedding] text:", test_text)
        print("[Embedding] vector length:", len(embedding))
        print("[Embedding] sample (head):", embedding[:10], "...")
    except Exception as e:
        print(f"[Embedding] Error: {e}")

def test_chat() -> None:
    """測試 Azure GPT-4o Chat 模型"""
    llm = gpt_4o()
    test_prompt = [
        HumanMessage(content="你好！我正在測試 Azure GPT-4o，請簡短介紹一下你自己。")
    ]
    try:
        response = llm.invoke(test_prompt)
        print("\n[ChatGPT-4o] response:", response.content)
    except Exception as e:
        print(f"[ChatGPT-4o] Error: {e}")

if __name__ == "__main__":
    print("=== 開始測試 Azure Chat & Embedding ===")
    test_embedding()
    test_chat()
    print("=== 測試完畢 ===")
