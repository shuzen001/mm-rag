import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub heavy dependencies before importing app
for name in [
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
    "langchain_openai",
    "langchain_openai.chat_models",
    "langchain.callbacks",
    "langchain.callbacks.streaming_stdout",
    "langchain.retrievers.multi_vector",
    "langchain.storage",
    "langchain_community.vectorstores",
    "langchain_core.documents",
    "langchain_text_splitters",
    "numpy",
    "IPython",
    "IPython.display",
    "dotenv",
    "main",
    "utils.LLM_Tool",
]:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)

sys.modules["langchain_core"].messages = ModuleType("langchain_core.messages")
sys.modules["langchain_core"].output_parsers = ModuleType(
    "langchain_core.output_parsers"
)
sys.modules["langchain_core"].prompts = ModuleType("langchain_core.prompts")
sys.modules["langchain_core"].messages.HumanMessage = object
sys.modules["langchain_core"].output_parsers.StrOutputParser = object
sys.modules["langchain_core"].prompts.ChatPromptTemplate = object
sys.modules["langchain_openai"].chat_models = ModuleType("langchain_openai.chat_models")


class _Dummy:
    def __init__(self, *a, **k):
        pass


sys.modules["langchain_openai"].chat_models.ChatOpenAI = _Dummy
sys.modules["langchain_openai"].OpenAIEmbeddings = _Dummy
sys.modules["langchain.retrievers.multi_vector"].MultiVectorRetriever = object
sys.modules["langchain.storage"].InMemoryStore = object
sys.modules["langchain_community.vectorstores"].FAISS = object
sys.modules["langchain_core.documents"].Document = object
sys.modules["langchain_text_splitters"].NLTKTextSplitter = object
sys.modules["IPython"].display = ModuleType("IPython.display")
sys.modules["IPython"].display.HTML = lambda *a, **k: None
sys.modules["IPython"].display.display = lambda *a, **k: None
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["main"].multi_modal_rag_chain = lambda *a, **k: None
sys.modules["utils.LLM_Tool"].text_embedding_3_large = None

import app


def test_list_user_files(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "DATABASE_ROOT", str(tmp_path / "db"))
    monkeypatch.setattr(app, "UPLOAD_ROOT", str(tmp_path / "uploads"))

    user_dir = Path(app.UPLOAD_ROOT) / "alice"
    user_dir.mkdir(parents=True)
    (user_dir / "a.pdf").write_text("x")
    (user_dir / "b.docx").write_text("y")

    files = app.list_user_files("alice")
    assert set(files) == {"a.pdf", "b.docx"}
