# translator.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from functools import lru_cache
from langdetect import detect

# Используем дистиллированную версию для скорости (около 2.4 ГБ)
MODEL_NAME = "facebook/nllb-200-distilled-600M"

@lru_cache(maxsize=1)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

# Карта соответствия коротких кодов кодам NLLB
NLLB_LANG_MAP = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "th": "tha_Thai",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Kore",
    "tr": "tur_Latn",
    "ar": "ary_Arab"
}

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model = get_model()
    
    # Получаем полные коды языков для NLLB
    src_code = NLLB_LANG_MAP.get(src_lang, "eng_Latn")
    tgt_code = NLLB_LANG_MAP.get(tgt_lang, "rus_Cyrl")

    inputs = tokenizer(text, return_tensors="pt")
    
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code], 
        max_length=128
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


# ------------------ LANGUAGE DETECTION ------------------

LANG_DISPLAY = {
    "ru": "🇷🇺 Русский",
    "en": "🇬🇧 English",
    "th": "🇹🇭 ไทย (Thai)",
    "zh": "🇨🇳 中文",
    "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어",
    "vi": "🇻🇳 Tiếng Việt",
    "tr": "🇹🇷 Türkçe",
    "ar": "🇸🇦 العربية",
    "he": "🇮🇱 עברית",
    "id": "🇮🇩 Bahasa Indonesia",
    "ms": "🇲🇾 Bahasa Melayu",
    "fa": "🇮🇷 فارسی",
    "ka": "🇬🇪 ქართული",
    "hy": "🇦🇲 Հայերեն",
}


def detect_lang_code(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


def detect_lang_for_display(text: str) -> str:
    code = detect_lang_code(text)
    return LANG_DISPLAY.get(code, f"🌍 {code}")
