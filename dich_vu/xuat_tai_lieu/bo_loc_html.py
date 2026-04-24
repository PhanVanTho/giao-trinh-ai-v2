# -*- coding: utf-8 -*-
import html
import re
from bs4 import BeautifulSoup

def _strip_control_chars(text):
    """Loại bỏ NULL bytes và control characters không hợp lệ trong XML/PDF."""
    if not text:
        return text
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def clean_for_reportlab(html_text):
    """
    Enterprise-grade HTML sanitizer for ReportLab (V21.5).
    Handles nesting, unsupported tags, block-level alignment, and security.
    """
    if not html_text:
        return ""

    # 1. Khởi tạo BeautifulSoup để normalize HTML
    soup = BeautifulSoup(str(html_text), "html.parser")

    # 2. Security: Loại bỏ hoàn toàn script và style
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # 3. Block-level handling: Tránh dính chữ khi unwrap các thẻ khối
    BLOCK_TAG_LIST = ["div", "p", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
    for block_tag in soup.find_all(BLOCK_TAG_LIST):
        # Thêm dấu xuống dòng trước thẻ khối để text không bị dính vào nhau
        block_tag.insert_before("\n")
        # Chúng ta không unwrap luôn ở đây để tránh làm hỏng cấu trúc cây khi đang lặp
    
    # 4. Whitelist tags & Attribute cleaning
    # ReportLab support: b, i, u, sup, sub, a, br, span (limited), font (limited)
    ALLOWED_TAGS = ["b", "i", "u", "sup", "sub", "a", "br"]
    
    for tag in soup.find_all(True):
        if tag.name not in ALLOWED_TAGS:
            tag.unwrap()
        else:
            # Làm sạch thuộc tính (Attributes)
            if tag.name == "a":
                href = tag.get("href", "").strip()
                if href:
                    # Chốt chặn href: Chỉ giữ lại link, xóa title, target, rel...
                    tag.attrs = {"href": href}
                else:
                    # Nếu <a> không có link thì gỡ thẻ luôn
                    tag.unwrap()
            else:
                # b, i, u, sup, sub, br không được có thuộc tính (đặc biệt là class)
                tag.attrs = {}

    # 5. Kết xuất lại chuỗi sạch
    clean_html = str(soup)

    # 6. Unicode & Entity Fix (Enterprise Final Touch)
    # Giải mã &nbsp;, &quot;...
    clean_html = html.unescape(clean_html)
    # Thay thế \xa0 (non-breaking space) bằng dấu cách thường để ReportLab không lỗi font
    clean_html = clean_html.replace("\xa0", " ")
    
    # Dọn dẹp khoảng trắng thừa do unwrap tags
    clean_html = re.sub(r' +', ' ', clean_html)
    
    # Loại bỏ control characters không hợp lệ
    clean_html = _strip_control_chars(clean_html)
    
    return clean_html.strip()

def clean_for_docx(html_text):
    """
    Sanitizer cho DOCX: Giữ cấu trúc văn bản nhưng gỡ hết HTML.
    """
    if not html_text:
        return ""
        
    soup = BeautifulSoup(str(html_text), "html.parser")
    
    # Giữ xuống dòng cho thẻ <br>
    for br in soup.find_all("br"):
        br.replace_with("\n")
        
    # Giữ xuống dòng cho các thẻ khối
    for block in soup.find_all(["div", "p", "li"]):
        block.insert_before("\n")
        
    # Lấy văn bản thuần
    text = soup.get_text()
    
    # Unescape entities
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    
    # Loại bỏ control characters không hợp lệ
    text = _strip_control_chars(text)
    
    return text.strip()
