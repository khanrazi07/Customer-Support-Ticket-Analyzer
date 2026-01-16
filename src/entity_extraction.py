import re
from typing import Dict, List


COMPLAINT_KEYWORDS = [
    "broken", "late", "delay", "error", "failed",
    "missing", "refund", "incorrect", "damaged"
]


def extract_products(text: str, product_list: List[str]) -> List[str]:
    text_lower = text.lower()
    return [
        product for product in product_list
        if product.lower() in text_lower
    ]


def extract_dates(text: str) -> List[str]:
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b"
    ]

    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text.lower()))
    return dates


def extract_complaints(text: str) -> List[str]:
    text_lower = text.lower()
    return [
        word for word in COMPLAINT_KEYWORDS
        if word in text_lower
    ]


def extract_entities(
    text: str,
    product_list: List[str]
) -> Dict[str, List[str]]:

    return {
        "products": extract_products(text, product_list),
        "dates": extract_dates(text),
        "complaints": extract_complaints(text)
    }
