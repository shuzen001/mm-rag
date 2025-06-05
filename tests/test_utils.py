import base64
import os
import shutil
import tempfile

import pytest

pytest.importorskip("PIL.Image")
from PIL import Image

from utils.vector_store import (
    looks_like_base64,
    is_image_data,
    resize_base64_image,
)
from utils.summarize import encode_image
from main import split_image_text_types, img_prompt_func


def create_temp_image(size=(10, 10), color=(255, 0, 0)):
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    img = Image.new('RGB', size, color)
    img.save(path)
    return path


def test_encode_image_and_resize():
    img_path = create_temp_image()
    b64 = encode_image(img_path)
    assert looks_like_base64(b64)
    assert is_image_data(b64)

    resized_b64 = resize_base64_image(b64, size=(5, 5))
    assert looks_like_base64(resized_b64)
    assert is_image_data(resized_b64)

    data = base64.b64decode(resized_b64)
    img = Image.open(tempfile.NamedTemporaryFile(delete=False))
    img.file.write(data)
    img.file.seek(0)
    img = Image.open(img.file.name)
    assert img.size == (5, 5)
    img.close()

    os.remove(img_path)


def test_split_image_text_types_and_prompt(tmp_path):
    # create fake image
    img_path = tmp_path / "img.png"
    Image.new('RGB', (10, 10), (0, 255, 0)).save(img_path)

    docs = [
        {"filename": img_path.name},
        {"type": "text", "content": "hello"},
        "plain string",
    ]

    figures_dir = os.path.join('database', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    shutil_path = os.path.join(figures_dir, img_path.name)
    os.replace(str(img_path), shutil_path)

    result = split_image_text_types(docs)
    assert "images" in result and "texts" in result
    assert len(result["texts"]) == 2
    assert len(result["images"]) == 1

    prompt_messages = img_prompt_func({"context": result, "question": "test?"})
    assert isinstance(prompt_messages, list)
    assert any("data:image/jpeg;base64" in str(msg) for msg in prompt_messages)


