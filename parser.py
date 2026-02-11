# parser.py

import re
import pandas as pd
from typing import Optional

PRICE_PATTERN = r'(\d+[.,]?\d*)$'            # цена в конце строки
QTY_PATTERN = r'^(?:x|\*)?(\d+)[\s\-]?'      # количество в начале строки

# Ключевые слова для "служебных" строк.
META_KEYWORDS = [
    "tax", "vat", "pos", "tel", "phone", "r#", "sum", "total",
    "итог", "сумма", "налог", "код", "касса", "чек", "оператор",
    # тайские: итог, нетто, скидка, сдача, сумма, "экономия"
    "รวม", "สุทธิ", "ส่วนลด", "เงินทอน", "ยอด",
    "ประหยัด", "ระหยัด",  # "экономия / save"
    "save",
]


def looks_like_metadata(line: str, price_value: Optional[float]) -> bool:
    """
    Фильтруем служебные строки: VAT-коды, телефоны, даты, итоговые суммы и т.п.
    """
    low = (line or "").lower()

    # Служебные слова
    if any(k in low for k in META_KEYWORDS):
        return True

    # Очень большие числа почти наверняка не отдельный товар
    if price_value is not None and price_value >= 150:
        return True

    # Телефоны вида 0-2826-7744
    if re.search(r"\d-\d{3,}", line):
        return True

    # Даты 23/11/68
    if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", line):
        return True

    return False


def parse_line_with_price(line: str):
    """
    Разбирает строку, где уже точно есть цена в конце.
    Возвращает (name, qty, price) или (None, None, None), если это мусор.
    """
    line = line.strip()
    price_match = re.search(PRICE_PATTERN, line)
    if not price_match:
        return None, None, None

    price_str = price_match.group(1).replace(",", ".")
    try:
        price = float(price_str)
    except ValueError:
        return None, None, None

    # Отсекаем явные метаданные
    if looks_like_metadata(line, price):
        return None, None, None

    # удаляем цену
    clean = re.sub(PRICE_PATTERN, "", line).strip()

    # Количество в начале
    qty_match = re.match(QTY_PATTERN, clean)
    if qty_match:
        qty = int(qty_match.group(1))
        clean = clean[qty_match.end():].strip()
    else:
        qty = 1

    name = clean

    # Слишком короткие/мусорные названия не берём
    if not name or len(name) < 2 or re.fullmatch(r"[\W_]+", name):
        return None, None, None

    # Очень мелкие цены (1–2 бат) часто пакеты/штампы/скидочные маркеры
    # Если нужно будет их учитывать — потом ослабим этот фильтр.
    if price < 3:
        return None, None, None

    return name, qty, price


def parse_receipt(lines):
    """
    Принимает список ОРИГИНАЛЬНЫХ строк OCR (НЕ перевода!)
    Возвращает DataFrame: item, qty, price, total.

    Паттерны:
      1) 'Название  x2  40.00'
      2) '40.00' на одной строке и 'Название' на следующей (часто в 7-Eleven TH).
    """
    items = []
    pending_name: Optional[str] = None
    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        line = (raw_line or "").strip()
        if not line:
            i += 1
            continue

        price_match = re.search(PRICE_PATTERN, line)
        has_letters = bool(re.search(r"[^\d.,]", line))

        # --- 1. Текст + цена в одной строке ---
        if price_match and has_letters:
            name, qty, price = parse_line_with_price(line)
            if name and price is not None:
                items.append({
                    "item": name,
                    "qty": qty,
                    "price": price,
                    "total": qty * price,
                })
            pending_name = None
            i += 1
            continue

        # --- 2. Строка только с ЦЕНОЙ ---
        if price_match and not has_letters:
            price_str = price_match.group(1).replace(",", ".")
            try:
                price = float(price_str)
            except ValueError:
                pending_name = None
                i += 1
                continue

            # Мелочь игнорируем
            if price < 3:
                pending_name = None
                i += 1
                continue

            # 2A. Тайский формат: цена → следующая строка = название
            j = i + 1
            while j < n and not (lines[j] or "").strip():
                j += 1

            if j < n:
                next_raw = lines[j]
                next_line = (next_raw or "").strip()
                next_has_letters = bool(re.search(r"[^\d.,]", next_line))
                next_price_match = re.search(PRICE_PATTERN, next_line)

                if next_line and next_has_letters and not next_price_match:
                    combined = f"{next_line} {line}"
                    if not looks_like_metadata(combined, price):
                        items.append({
                            "item": next_line,
                            "qty": 1,
                            "price": price,
                            "total": price,
                        })
                        pending_name = None
                        i = j + 1
                        continue

            # 2B. Fallback: "предыдущая строка = название"
            if pending_name:
                combined = f"{pending_name} {line}"
                if not looks_like_metadata(combined, price):
                    items.append({
                        "item": pending_name,
                        "qty": 1,
                        "price": price,
                        "total": price,
                    })

            pending_name = None
            i += 1
            continue

        # --- 3. Строка с буквами, но без цены: кандидаты на название ---
        if has_letters and not price_match:
            pending_name = line
            i += 1
            continue

        # Остальное (голые цифры) игнорируем
        i += 1

    df = pd.DataFrame(items)
    return df
