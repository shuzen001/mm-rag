import ast
import base64
import io
import os
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub minimal langchain modules like in test_utils
dummy_core = ModuleType("langchain_core")
dummy_core.messages = ModuleType("langchain_core.messages")
dummy_core.messages.HumanMessage = object
dummy_core.output_parsers = ModuleType("langchain_core.output_parsers")
dummy_core.output_parsers.StrOutputParser = object
dummy_core.prompts = ModuleType("langchain_core.prompts")
dummy_core.prompts.ChatPromptTemplate = object
sys.modules.setdefault("langchain_core", dummy_core)
sys.modules.setdefault("langchain_core.messages", dummy_core.messages)
sys.modules.setdefault("langchain_core.output_parsers", dummy_core.output_parsers)
sys.modules.setdefault("langchain_core.prompts", dummy_core.prompts)

dummy_openai = ModuleType("langchain_openai")
dummy_openai.chat_models = ModuleType("langchain_openai.chat_models")


class _Dummy:
    def __init__(self, *a, **k):
        pass


dummy_openai.chat_models.ChatOpenAI = _Dummy
dummy_openai.OpenAIEmbeddings = _Dummy
dummy_openai.ChatOpenAI = _Dummy
dummy_openai.OpenAIEmbeddings = _Dummy
sys.modules.setdefault("langchain_openai", dummy_openai)
sys.modules.setdefault("langchain_openai.chat_models", dummy_openai.chat_models)

dummy_callbacks = ModuleType("langchain.callbacks.streaming_stdout")
dummy_callbacks.StreamingStdOutCallbackHandler = object
sys.modules.setdefault("langchain.callbacks", ModuleType("langchain.callbacks"))
sys.modules.setdefault("langchain.callbacks.streaming_stdout", dummy_callbacks)

sys.modules.setdefault(
    "langchain.retrievers.multi_vector", ModuleType("langchain.retrievers.multi_vector")
)
sys.modules["langchain.retrievers.multi_vector"].MultiVectorRetriever = object
sys.modules.setdefault("langchain.storage", ModuleType("langchain.storage"))
sys.modules["langchain.storage"].InMemoryStore = object
sys.modules.setdefault(
    "langchain_community.vectorstores", ModuleType("langchain_community.vectorstores")
)
sys.modules["langchain_community.vectorstores"].FAISS = object
sys.modules.setdefault(
    "langchain_core.documents", ModuleType("langchain_core.documents")
)
sys.modules["langchain_core.documents"].Document = object
sys.modules.setdefault("numpy", ModuleType("numpy"))
dummy_ipython = ModuleType("IPython")
dummy_ipython.display = ModuleType("IPython.display")
dummy_ipython.display.HTML = lambda *a, **k: None
dummy_ipython.display.display = lambda *a, **k: None
sys.modules.setdefault("IPython", dummy_ipython)
sys.modules.setdefault("IPython.display", dummy_ipython.display)

dummy_dotenv = ModuleType("dotenv")
dummy_dotenv.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dummy_dotenv)

from PIL import Image

# helper to extract function without importing heavy dependencies


def load_split_function():
    root = Path(__file__).resolve().parents[1]
    src = (root / "main.py").read_text()
    module = ast.parse(src)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "split_image_text_types":
            func_code = ast.Module([node], [])
            ns = {
                "os": os,
                "encode_image": __import__(
                    "utils.summarize", fromlist=["encode_image"]
                ).encode_image,
                "resize_base64_image": __import__(
                    "utils.vector_store", fromlist=["resize_base64_image"]
                ).resize_base64_image,
                "Document": type("Document", (), {}),
                "logger": type("L", (), {"warning": lambda *a, **k: None})(),
            }
            exec(compile(func_code, "<ast>", "exec"), ns)
            return ns["split_image_text_types"]
    raise AssertionError("Function not found")


split_image_text_types = load_split_function()


def make_image(path, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (10, 10), color).save(path)


def test_split_image_text_types_with_different_user_dirs(tmp_path):
    user1 = tmp_path / "u1"
    user2 = tmp_path / "u2"
    fig1 = user1 / "figures" / "doc"
    fig2 = user2 / "figures" / "doc"
    img1 = fig1 / "img.png"
    img2 = fig2 / "img.png"
    make_image(img1, (255, 0, 0))
    make_image(img2, (0, 0, 255))

    doc = {"filename": "figures/doc/img.png", "type": "image"}

    res1 = split_image_text_types([doc], str(user1 / "figures"))
    res2 = split_image_text_types([doc], str(user2 / "figures"))

    assert res1["images"] and res2["images"]
    assert res1["images"][0] != res2["images"][0]

    data1 = base64.b64decode(res1["images"][0])
    data2 = base64.b64decode(res2["images"][0])
    with Image.open(io.BytesIO(data1)) as im1, Image.open(io.BytesIO(data2)) as im2:
        assert im1.getpixel((0, 0)) == (255, 0, 0)
        assert im2.getpixel((0, 0)) == (0, 0, 255)
