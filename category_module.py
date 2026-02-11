# category_module.py
import re

# Минимум категорий, которые понятны комиссии и полезны:
EN_RULES = {
    "Food": [
        r"\bchicken\b", r"\bpork\b", r"\bbeef\b", r"\brice\b", r"\bnoodle\b", r"\bsoup\b",
        r"\bhot\s*pot\b", r"\bmeatball\b", r"\bshrimp\b", r"\bcrab\b", r"\bvegetable\b",
    ],
    "Drinks": [
        r"\bwater\b", r"\btea\b", r"\bcoffee\b", r"\bjuice\b", r"\bbeer\b", r"\bsoda\b", r"\bcola\b",
    ],
    "Household": [
        r"\bsoap\b", r"\bdetergent\b", r"\btissue\b", r"\bshampoo\b", r"\bclean(er)?\b", r"\bbleach\b",
    ],
    "Personal Care": [
        r"\btooth(paste)?\b", r"\bcream\b", r"\blotion\b", r"\bdeodorant\b", r"\brazor\b",
    ],
    "Service & Fees": [
        r"\bvat\b", r"\btax\b", r"\bservice\b", r"\bcharge\b", r"\bfee\b", r"\btip\b",
        r"\bbag\b", r"\bpack(aging)?\b",
    ],
    "Other": []
}

def categorize_item_en(name_en: str) -> str:
    s = (name_en or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return "Other"

    for cat, patterns in EN_RULES.items():
        if cat == "Other":
            continue
        for p in patterns:
            if re.search(p, s, flags=re.IGNORECASE):
                return cat
    return "Other"
