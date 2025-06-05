import base64
import io
import sys
from pathlib import Path


import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import types

# Provide dummy modules for optional dependencies used in vector_store.py
sys.modules.setdefault("langchain", types.ModuleType("langchain"))
sys.modules.setdefault("langchain.retrievers", types.ModuleType("langchain.retrievers"))
mv = types.ModuleType("langchain.retrievers.multi_vector")
mv.MultiVectorRetriever = object
sys.modules.setdefault("langchain.retrievers.multi_vector", mv)
st = types.ModuleType("langchain.storage")
st.InMemoryStore = object
sys.modules.setdefault("langchain.storage", st)
lc_core = types.ModuleType("langchain_core")
sys.modules.setdefault("langchain_core", lc_core)
docs = types.ModuleType("langchain_core.documents")
docs.Document = object
sys.modules.setdefault("langchain_core.documents", docs)
display_mod = types.ModuleType("IPython.display")
display_mod.HTML = lambda *a, **k: None
display_mod.display = lambda *a, **k: None
sys.modules.setdefault("IPython.display", display_mod)

# Minimal stub for PIL if not installed
if "PIL" not in sys.modules:
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    pil_stub.Image = image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub

from utils.vector_store import (
    looks_like_base64,
    is_image_data,
    resize_base64_image,
)

try:
    from PIL import Image
    PIL_AVAILABLE = hasattr(Image, "open")
except ImportError:
    PIL_AVAILABLE = False

# 1x1 red dot PNG
B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAAYOChgAAAAASUVORK5CYII="
)

def test_looks_like_base64_valid_and_invalid():
    assert looks_like_base64("SGVsbG8=")
    assert looks_like_base64(B64_PNG)
    assert not looks_like_base64("not_base64!")

def test_is_image_data():
    assert is_image_data(B64_PNG)
    text_b64 = base64.b64encode(b"hello").decode()
    assert not is_image_data(text_b64)

def test_resize_base64_image_changes_size():
    if not PIL_AVAILABLE:
        pytest.skip("Pillow not installed")
    original = base64.b64decode(B64_PNG)
    orig_img = Image.open(io.BytesIO(original))
    resized_b64 = resize_base64_image(B64_PNG, size=(2, 2))
    assert looks_like_base64(resized_b64)
    resized = base64.b64decode(resized_b64)
    resized_img = Image.open(io.BytesIO(resized))
    assert orig_img.size != resized_img.size
    assert resized_img.size == (2, 2)
