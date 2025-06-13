import base64
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from utils.summarize import encode_image
from utils.vector_store import resize_base64_image


def test_encode_image_creates_base64_and_decodes():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.png")
        Image.new("RGB", (50, 50), (255, 0, 0)).save(img_path)

        encoded = encode_image(img_path)
        assert isinstance(encoded, str)

        data = base64.b64decode(encoded)
        with Image.open(io.BytesIO(data)) as img:
            assert img.size == (50, 50)


def test_resize_base64_image_resizes_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.png")
        Image.new("RGB", (60, 60), (0, 255, 0)).save(img_path)

        encoded = encode_image(img_path)
        resized_b64 = resize_base64_image(encoded, size=(30, 30))

        data = base64.b64decode(resized_b64)
        with Image.open(io.BytesIO(data)) as img:
            assert img.size == (30, 30)
