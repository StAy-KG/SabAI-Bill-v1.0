# v1_ocr_module.py

from paddleocr import PaddleOCR
from PIL import Image, UnidentifiedImageError
from functools import lru_cache
import numpy as np
import io


# Поддерживаемые языки OCR, привязанные к PaddleOCR
SUPPORTED_OCR_LANGS = {
    "ru",          # кириллица: RU/UA/KG и т.п.
    "en",          # чистый английский
    "latin",       # общий латинский (многие EU-языки)
    "th",          # тайский + англ.
    "ch",          # китайский упрощённый
    "chinese_cht", # китайский традиционный
    "japan",       # японский
    "korean",      # корейский
    "arabic",      # арабский
}


@lru_cache(maxsize=None)
def get_ocr(lang_code: str = "ru") -> PaddleOCR:
    """
    Кэшируем инстансы PaddleOCR по коду языка.
    Чтобы модель не грузилась заново при каждом запросе.
    """
    if lang_code not in SUPPORTED_OCR_LANGS:
        lang_code = "en"

    return PaddleOCR(
        lang=lang_code,
        #use_angle_cls=False,                  # убираем лишнюю голову для скорости
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        #use_textline_orientation=False,
        # use_gpu=True,  # если поставишь paddlepaddle-gpu и есть CUDA – можно включить
    )


def _to_ndarray(image_source):
    """
    Приводим вход к формату, который понимает PaddleOCR:
    - str: путь к файлу
    - streamlit UploadedFile / BytesIO / объект с .getvalue() или .read()
    - bytes / bytearray
    """
    # Уже путь к файлу
    if isinstance(image_source, str):
        return image_source

    # Streamlit UploadedFile или любой объект с .getvalue()
    if hasattr(image_source, "getvalue"):
        image_bytes = image_source.getvalue()
    elif isinstance(image_source, (bytes, bytearray)):
        image_bytes = image_source
    elif hasattr(image_source, "read"):
        # file-like объект
        try:
            if hasattr(image_source, "seek"):
                image_source.seek(0)
            image_bytes = image_source.read()
        finally:
            try:
                image_source.seek(0)
            except Exception:
                pass
    else:
        raise TypeError(
            f"Unsupported image_source type: {type(image_source)}. "
            "Expected str path, bytes or Streamlit UploadedFile."
        )

    if not image_bytes:
        raise ValueError("Empty image data: got 0 bytes from uploaded file")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        raise UnidentifiedImageError(
            "PIL не смог распознать изображение. "
            "Проверь, что это действительно JPEG/PNG и файл не битый."
        ) from e

    # 🔽 Даунскейлим ОЧЕНЬ большие картинки, чтобы ускорить OCR
    max_side = 1600
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size)

    return np.array(pil_img)


def extract_text(image_source, ocr_lang: str = "ru"):
    """
    OCR по изображению.
    :param image_source: путь/байты/UploadedFile
    :param ocr_lang: код языка для PaddleOCR (ru, en, latin, th, ...)
    :return: list[str] — строки текста
    """
    img_for_ocr = _to_ndarray(image_source)
    ocr = get_ocr(ocr_lang)

    result = ocr.ocr(img_for_ocr)

    # На всякий: генератор → список
    if not isinstance(result, (list, tuple)):
        result = list(result)

    lines: list[str] = []

    for res in result:
        # 1) Новый формат PaddleX: объект результата с .json
        if hasattr(res, "json"):
            data = res.json  # dict
            inner = data.get("res", data)
            rec_texts = inner.get("rec_texts", [])
            for t in rec_texts:
                if isinstance(t, str) and t.strip():
                    lines.append(t.strip())
            continue

        # 2) Просто dict: {'res': {..., 'rec_texts': [...]}} или сразу c 'rec_texts'
        if isinstance(res, dict):
            inner = res.get("res", res)
            rec_texts = inner.get("rec_texts", [])
            for t in rec_texts:
                if isinstance(t, str) and t.strip():
                    lines.append(t.strip())
            continue

        # 3) Старый формат: список [ [box, (text, score)], ... ]
        if isinstance(res, list):
            for line in res:
                if (
                    isinstance(line, (list, tuple))
                    and len(line) >= 2
                    and isinstance(line[1], (list, tuple))
                    and len(line[1]) > 0
                ):
                    text = line[1][0]
                    if isinstance(text, str) and text.strip():
                        lines.append(text.strip())
            continue

        # 4) На всякий случай: объект с атрибутом rec_texts
        if hasattr(res, "rec_texts"):
            rec_texts = getattr(res, "rec_texts", []) or []
            for t in rec_texts:
                if isinstance(t, str) and t.strip():
                    lines.append(t.strip())
            continue

    return lines
