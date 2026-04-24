# -*- coding: utf-8 -*-
import re
import uuid
import logging

logger = logging.getLogger(__name__)

def remove_diacritics(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so sánh Lexical chính xác."""
    if not text: return ""
    import unicodedata
    # Normalize to NFD (Decomposed form)
    nfkd_form = unicodedata.normalize('NFKD', text)
    # Filter out non-spacing marks (diacritics)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Thay thế chữ Đ/đ đặc biệt
    only_ascii = only_ascii.replace('đ', 'd').replace('Đ', 'D')
    return only_ascii

_RAC_PATTERNS = [
    r"^==\s*Liên kết ngoài\s*==.*$",   # một số heading (tùy)
]

def lam_sach_trang(page: dict) -> dict:
    """
    Làm sạch text ở mức trang (bỏ dòng trống, ký tự lạ…)
    """
    text = (page.get("text") or "").replace("\r", "\n")
    # bỏ dòng rất dài vô nghĩa (hiếm)
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if len(ln) > 2000:
            ln = ln[:2000]
        lines.append(ln)
    text = "\n".join(lines)

    # loại một số pattern rác (tùy chọn)
    for pat in _RAC_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    page["text"] = text.strip()
    return page

def _tach_cau_doan(text: str):
    # Tách theo xuống dòng trước, giữ đoạn ngắn vừa phải
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts

def tach_cau(text: str):
    """Chia nhỏ văn bản thành danh sách các câu (sentence-level)"""
    if not text: return []
    # Regex split by terminal punctuation . ! ? followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def chia_doan(pages, do_dai_min=100, do_dai_max=600):
    """
    Chia nhỏ các trang Wikipedia thành các đoạn văn bản (Passages).
    Hệ thống mới ưu tiên Sentence-level slicing để tăng độ chính xác trích xuất Fact.
    """
    passages = []
    p_id_counter = 0
    
    for page in pages:
        full_text = page.get("text", "")
        url = page.get("url", "")
        title = page.get("title", "Wikipedia")
        
        # Bước 1: Chia theo đoạn văn bản thô (\n\n)
        raw_chunks = full_text.split("\n\n")
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk) < 20: continue
            
            # Bước 2: Với các đoạn dài, tiến hành tách câu (Sentence-level Precision)
            if len(chunk) > do_dai_max:
                sentences = tach_cau(chunk)
                current_subchunk = ""
                for s in sentences:
                    if len(current_subchunk) + len(s) < do_dai_max:
                        current_subchunk += " " + s
                    else:
                        if len(current_subchunk) >= do_dai_min:
                            passages.append({
                                "id": p_id_counter,
                                "pid": p_id_counter,
                                "text": current_subchunk.strip(),
                                "url": url,
                                "title": title
                            })
                            p_id_counter += 1
                        current_subchunk = s
                # Add nốt phần còn lại
                if len(current_subchunk) >= 10:
                    passages.append({
                        "id": p_id_counter,
                        "pid": p_id_counter,
                        "text": current_subchunk.strip(),
                        "url": url,
                        "title": title
                    })
            else:
                # Đoạn vừa tầm -> Giữ nguyên làm 1 chunk
                passages.append({
                    "id": p_id_counter,
                    "pid": p_id_counter,
                    "text": chunk,
                    "url": url,
                    "title": title
                })
                p_id_counter += 1
                
    return passages
