import importlib
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub external dependencies before importing the module under test
stub_converter_module = types.ModuleType("docling.document_converter")
stub_converter_module.DocumentConverter = lambda: None
sys.modules["docling.document_converter"] = stub_converter_module

stub_base_module = types.ModuleType("docling_core.types.doc.base")
stub_base_module.ImageRefMode = object
sys.modules["docling_core.types.doc.base"] = stub_base_module

# Additional stubs for langchain to avoid heavy deps
dummy_core = types.ModuleType("langchain_core")
dummy_core.messages = types.ModuleType("langchain_core.messages")
dummy_core.messages.HumanMessage = object
dummy_core.output_parsers = types.ModuleType("langchain_core.output_parsers")
dummy_core.output_parsers.StrOutputParser = object
dummy_core.prompts = types.ModuleType("langchain_core.prompts")
dummy_core.prompts.ChatPromptTemplate = object
sys.modules.setdefault("langchain_core", dummy_core)
sys.modules.setdefault("langchain_core.messages", dummy_core.messages)
sys.modules.setdefault("langchain_core.output_parsers", dummy_core.output_parsers)
sys.modules.setdefault("langchain_core.prompts", dummy_core.prompts)

dummy_openai = types.ModuleType("langchain_openai")
dummy_openai.chat_models = types.ModuleType("langchain_openai.chat_models")


class _Dummy:
    def __init__(self, *a, **k):
        pass


dummy_openai.chat_models.ChatOpenAI = _Dummy
dummy_openai.OpenAIEmbeddings = _Dummy
dummy_openai.ChatOpenAI = _Dummy
dummy_openai.OpenAIEmbeddings = _Dummy
sys.modules.setdefault("langchain_openai", dummy_openai)
sys.modules.setdefault("langchain_openai.chat_models", dummy_openai.chat_models)

dummy_callbacks = types.ModuleType("langchain.callbacks.streaming_stdout")
dummy_callbacks.StreamingStdOutCallbackHandler = object
sys.modules.setdefault("langchain.callbacks", types.ModuleType("langchain.callbacks"))
sys.modules.setdefault("langchain.callbacks.streaming_stdout", dummy_callbacks)

sys.modules.setdefault(
    "langchain.retrievers.multi_vector",
    types.ModuleType("langchain.retrievers.multi_vector"),
)
sys.modules["langchain.retrievers.multi_vector"].MultiVectorRetriever = object
sys.modules.setdefault("langchain.storage", types.ModuleType("langchain.storage"))
sys.modules["langchain.storage"].InMemoryStore = object
sys.modules.setdefault(
    "langchain_community.vectorstores",
    types.ModuleType("langchain_community.vectorstores"),
)
sys.modules["langchain_community.vectorstores"].FAISS = object
sys.modules.setdefault(
    "langchain_core.documents", types.ModuleType("langchain_core.documents")
)
sys.modules["langchain_core.documents"].Document = object
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
dummy_ipython = types.ModuleType("IPython")
dummy_ipython.display = types.ModuleType("IPython.display")
dummy_ipython.display.HTML = lambda *a, **k: None
dummy_ipython.display.display = lambda *a, **k: None
sys.modules.setdefault("IPython", dummy_ipython)
sys.modules.setdefault("IPython.display", dummy_ipython.display)

dummy_dotenv = types.ModuleType("dotenv")
dummy_dotenv.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dummy_dotenv)

import utils.docling_markdown as dm


class DummyDoc:
    def export_to_markdown(self, image_placeholder="<!-- image -->"):
        return f"Intro {image_placeholder} Outro {image_placeholder}"

    def _list_images_on_disk(self):
        return [Path("img1.png"), Path("img2.png")]


class DummyConverter:
    def convert(self, path):
        return types.SimpleNamespace(document=DummyDoc())


def test_convert_file_to_markdown_replaces_placeholders(monkeypatch):
    monkeypatch.setattr(dm, "DocumentConverter", lambda: DummyConverter())
    summaries = ["summary1", "summary2"]

    def fake_summarize_image(image_path):
        return summaries.pop(0)

    monkeypatch.setattr(dm, "_summarize_image", fake_summarize_image)
    markdown = dm.convert_file_to_markdown("dummy.pdf")
    assert markdown == "Intro summary1 Outro summary2"
