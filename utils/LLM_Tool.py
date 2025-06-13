# import libs
import os

from dotenv import load_dotenv

from utils.logging_config import get_logger

load_dotenv()
# API key will be loaded from .env file

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = get_logger(__name__)


def gpt_4o():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)
    return llm


def gpt_4o_for_summary():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
    return llm


text_embedding_3_large = OpenAIEmbeddings(model="text-embedding-3-large")


def test_embedding() -> None:
    """測試 Azure Embedding 部署"""
    test_text = "Hello, Azure embeddings!"
    try:
        embedding = text_embedding_3_large.embed_query(test_text)
        logger.info("\n[Embedding] text: %s", test_text)
        logger.info("[Embedding] vector length: %s", len(embedding))
        logger.info("[Embedding] sample (head): %s ...", embedding[:10])
    except Exception as e:
        logger.error(f"[Embedding] Error: {e}")


def test_chat() -> None:
    """測試 Azure GPT-4o Chat 模型"""
    llm = gpt_4o()
    test_prompt = [
        HumanMessage(content="你好！我正在測試 Azure GPT-4o，請簡短介紹一下你自己。")
    ]
    try:
        response = llm.invoke(test_prompt)
        logger.info("\n[ChatGPT-4o] response: %s", response.content)
    except Exception as e:
        logger.error(f"[ChatGPT-4o] Error: {e}")


if __name__ == "__main__":
    logger.info("=== 開始測試 Azure Chat & Embedding ===")
    test_embedding()
    test_chat()
    logger.info("=== 測試完畢 ===")
