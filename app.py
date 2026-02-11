# app.py

import streamlit as st
from PIL import Image
import pandas as pd

from ocr_module import extract_text
from translator import translate_text, detect_lang_for_display, detect_lang_code
from parser import parse_receipt
from split_engine import split_bill
from category_module import categorize_item_en

# ------------------ UI STYLE ------------------

st.set_page_config(page_title="SabAI Bill", layout="wide")

st.markdown("""
    <style>
        body {background-color: #0D0F12;}
        .main {background-color: #0D0F12;}
        h1, h2, h3, h4, h5, h6, p, label {
            color: #D8F3FF !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00E5FF, #7A00FF);
            color: white;
            border-radius: 8px;
            padding: 0.6em 1em;
            border: none;
        }
        .stDataFrame {color: white;}
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------

st.markdown(
    "<h1 style='text-align:center; color:#00E5FF;'>SabAI Bill — Smart Receipt Scanner</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Fully offline OCR + Translation + Parsing + Split Bill</p>",
    unsafe_allow_html=True
)

# ------------------ CACHED OCR ------------------

@st.cache_data(show_spinner=False)
def run_ocr_cached(file_bytes: bytes, ocr_lang: str):
    """
    Кешируем результат OCR по (байты файла + язык).
    Чтобы при смене языка перевода / групп не пересчитывать OCR.
    """
    return extract_text(file_bytes, ocr_lang=ocr_lang)


@st.cache_data(show_spinner=False)
def translate_items_cached(items, src_lang: str, tgt_lang: str):
    """
    Кэш для перевода названий позиций.
    items — итерируемый объект с строками (названия).
    """
    from translator import translate_text  # локальный импорт, чтобы не было циклов
    results = []
    for name in items:
        name = str(name) if name is not None else ""
        if not name.strip():
            results.append(name)
            continue
        try:
            results.append(
                translate_text(
                    name,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
            )
        except Exception:
            results.append(name)
    return results

# ------------------ FILE UPLOAD ------------------

uploaded_file = st.file_uploader("Загрузите фото чека", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженный чек", width="stretch")

    # ------------------ OCR LANGUAGE SELECTION ------------------
    st.subheader("🌍 Язык / страна чека для OCR")

    OCR_LANG_CHOICES = {
        "🇷🇺 Россия / СНГ (кириллица)": "ru",
        "🇹🇭 Таиланд (тайский + англ.)": "th",
        "🌍 International (латиница, EU/US)": "latin",
        "🇺🇸 Только English": "en",
        "🇨🇳 Китай (упр.)": "ch",
        "🇭🇰 Китай (традиц.)": "chinese_cht",
        "🇯🇵 Япония": "japan",
        "🇰🇷 Корея": "korean",
        "🇸🇦 Арабские страны": "arabic",
    }

    ocr_label = st.selectbox(
        "Выберите, откуда чек (это влияет на качество распознавания текста):",
        list(OCR_LANG_CHOICES.keys()),
        index=1  # по умолчанию Таиланд
    )
    ocr_lang = OCR_LANG_CHOICES[ocr_label]

    # ------------------ OCR ------------------
    st.subheader("🔍 Распознавание текста (OCR)")

    with st.spinner("Извлечение текста..."):
        file_bytes = uploaded_file.getvalue()
        lines = run_ocr_cached(file_bytes, ocr_lang)

    # нормализация
    if isinstance(lines, str):
        lines = [ln.strip() for ln in lines.splitlines() if ln.strip()]
    else:
        lines = [str(ln).strip() for ln in lines if str(ln).strip()]

    st.caption(f"DEBUG: найдено строк: {len(lines)}")  # можно потом убрать

    if not lines:
        st.error("Текст не найден 😿")
        st.stop()

    raw_text = "\n".join(lines)
    st.text(raw_text)

    # авто-определение языка по всему тексту
    src_lang_code = detect_lang_code(raw_text)
    src_lang_display = detect_lang_for_display(raw_text)
    st.caption(f"Обнаруженный язык чека: {src_lang_display}")

    # ------------------ TRANSLATION ------------------
    st.subheader("🌐 Перевод позиций")

    LANG_CHOICES = {
        "🇷🇺 Русский": "ru",
        "🇬🇧 English": "en",
        "🇨🇳 简体中文": "zh",
        "🇹🇼 繁體中文": "zh-TW",
        "🇹🇭 ไทย": "th",
        "🇯🇵 日本語": "ja",
        "🇰🇷 한국어": "ko",
        "🇻🇳 Tiếng Việt": "vi",
        "🇹🇷 Türkçe": "tr",
        "🇸🇦 العربية": "ar",
        "🇮🇱 עברית": "he",
        "🇮🇩 Bahasa Indonesia": "id",
        "🇲🇾 Bahasa Melayu": "ms",
        "🇮🇷 فارسی": "fa",
        "🇬🇪 ქართული": "ka",
        "🇦🇲 Հայերեն": "hy",
    }

    target_label = st.selectbox("Выберите язык перевода", list(LANG_CHOICES.keys()))
    target_lang = LANG_CHOICES[target_label]

    # ------------------ PARSING (по оригинальным OCR-строкам) ------------------
    st.subheader("🧠 Структурирование чека")

    df = parse_receipt(lines)

    # --- перевод в EN для категоризации (один словарь regex на английском) ---
    items_en = translate_items_cached(
        tuple(df["item"]),
        src_lang_code,
        "en",
    )

    df["item_en"] = items_en
    df["category"] = df["item_en"].apply(categorize_item_en)

    # Переводим только названия позиций для отображения
    df_display = df.copy()

    translated_items = translate_items_cached(
        tuple(df["item"]),   # tuple для стабильного ключа кэша
        src_lang_code,
        target_lang,
    )
    df_display["item"] = translated_items

    st.dataframe(df_display, width="stretch")

    st.markdown("---")

    # ================== DS / ANALYTICS BLOCK ==================
    st.subheader("📊 Распределение трат по категориям (offline)")

    # если ты уже добавил df["category"] (через EN-правила)
    cat_sum = (
        df.groupby("category", as_index=False)["total"]
          .sum()
          .sort_values("total", ascending=False)
    )

    st.dataframe(cat_sum, width="stretch")
    st.bar_chart(cat_sum.set_index("category")["total"])
    # ===========================================================

    # ------------------ SPLIT BILL ------------------
    st.subheader("🧮 Разделение счёта по группам")

    assignments = {}
    groups = ["A", "B", "C", "D"]

    for idx, row in df_display.iterrows():
        label = f"{row.get('item', f'Позиция {idx}')}"
        amount = row.get("total", "")
        text = f"{label} — {amount}" if amount != "" else label

        selected = st.multiselect(
            text,
            groups,
            key=f"item_{idx}"
        )
        assignments[idx] = selected

    if st.button("Рассчитать итог"):
        totals = split_bill(df, assignments)  # считаем по исходным данным (до перевода)
        st.subheader("💰 Итог по группам:")
        st.write(totals)
        st.success("✔ Готово! Прошу высший балл у комиссии ;-)")
