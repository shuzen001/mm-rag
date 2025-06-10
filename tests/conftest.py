import os
import sys
import types

# Provide dummy modules for optional dependencies used in utils.vector_store
sys.modules.setdefault("IPython", types.ModuleType("IPython"))
display_mod = types.ModuleType("display")
display_mod.display = lambda *a, **k: None
display_mod.HTML = lambda *a, **k: None
sys.modules.setdefault("IPython.display", display_mod)
sys.modules.setdefault("langchain_community", types.ModuleType("langchain_community"))
lc_vec = types.ModuleType("vectorstores")
lc_vec.FAISS = object
sys.modules.setdefault("langchain_community.vectorstores", lc_vec)

os.environ.setdefault("OPENAI_API_KEY", "dummy")

# Patch PIL.Image.open to work with empty temporary files in tests
import tempfile

import PIL.Image as PIL_Image

_orig_open = PIL_Image.open


def _patched_open(fp, mode="r", formats=None):
    if isinstance(fp, tempfile._TemporaryFileWrapper):

        class DummyImg:
            def __init__(self, f):
                self.file = f

        return DummyImg(fp)
    return _orig_open(fp, mode=mode, formats=formats)


PIL_Image.open = _patched_open

# Provide lightweight stand-ins for heavy functions used in tests
dummy_main = types.ModuleType("main")


def split_image_text_types(docs):
    images = [d["filename"] for d in docs if isinstance(d, dict) and "filename" in d]
    texts = [
        d["content"] if isinstance(d, dict) and "content" in d else d
        for d in docs
        if not (isinstance(d, dict) and "filename" in d)
    ]
    return {"images": images, "texts": texts}


def img_prompt_func(data_dict):
    return ["data:image/jpeg;base64,dummy"]


dummy_main.split_image_text_types = split_image_text_types
dummy_main.img_prompt_func = img_prompt_func
sys.modules.setdefault("main", dummy_main)

# Ensure project root is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
