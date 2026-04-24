# -*- coding: utf-8 -*-
"""
kiem_tra_cuc_bo.py
==================
Local rule-based validator — không cần gọi AI, chạy milliseconds.

Nhiệm vụ: Kiểm tra nhanh chương vừa được OpenAI tạo ra để quyết định:
  1. Có cần gọi Gemini Supervisor không?
  2. Có cần yêu cầu OpenAI viết lại ngay không?

Nguyên tắc:
  - 80% trường hợp KHÔNG cần gọi Gemini nhờ bộ lọc này.
  - Chi phí: 0 API call, ~0ms.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

# --- Ngưỡng kiểm tra ---
NGUONG_DO_DAI_TOI_THIEU = 150        # chars mỗi section content
NGUONG_DO_DAI_QUA_NGAN = 80          # chars — cảnh báo nghiêm trọng
NGUONG_SECTION_TOI_THIEU = 1         # số section tối thiểu
NGUONG_TI_LE_CITATION = 0.0          # ít nhất 0% đoạn có citation (kiểm tra sự tồn tại)


def _kiem_tra_json_hop_le(chapter: dict) -> tuple[bool, str]:
    """Kiểm tra cấu trúc JSON cơ bản."""
    if not isinstance(chapter, dict):
        return False, "Chapter không phải dict"
    if "title" not in chapter:
        return False, "Thiếu trường 'title'"
    if "sections" not in chapter or not isinstance(chapter["sections"], list):
        return False, "Thiếu hoặc sai định dạng 'sections'"
    if len(chapter["sections"]) < NGUONG_SECTION_TOI_THIEU:
        return False, f"Số section quá ít: {len(chapter['sections'])}"
    return True, ""


def _kiem_tra_co_citation(chapter: dict) -> tuple[bool, str]:
    """Kiểm tra có tồn tại ít nhất 1 citation trong toàn bộ chương."""
    for sec in chapter.get("sections", []):
        content = sec.get("content", "")
        citations = sec.get("citations", [])
        
        # Kiểm tra citation trong content text [1], [2]...
        if re.search(r'\[\d+\]', content):
            return True, ""
        # Kiểm tra citations array
        if citations and len(citations) > 0:
            return True, ""
    
    return False, "Không tìm thấy citation nào trong chương"


def _kiem_tra_do_dai(chapter: dict, quy_mo: str = "tieu_chuan") -> tuple[bool, str, list]:
    """
    Kiểm tra độ dài content của từng section.
    Trả về: (ok, issue_message, danh_sach_section_qua_ngan)
    """
    nguong = {
        "can_ban": 80,
        "tieu_chuan": 150,
        "chuyen_sau": 400,
    }.get(quy_mo, 150)
    
    sections_too_short = []
    for sec in chapter.get("sections", []):
        content = sec.get("content", "")
        if len(content) < nguong:
            sections_too_short.append({
                "title": sec.get("title", "?"),
                "length": len(content),
                "required": nguong
            })
    
    if sections_too_short:
        names = [s["title"] for s in sections_too_short]
        return False, f"Các section quá ngắn: {names}", sections_too_short
    
    return True, "", []


def _kiem_tra_cau_truc_markdown(chapter: dict) -> tuple[bool, str]:
    """Kiểm tra có sử dụng Markdown heading không (## hoặc ###)."""
    total_sections = 0
    sections_with_heading = 0
    
    for sec in chapter.get("sections", []):
        content = sec.get("content", "")
        total_sections += 1
        if re.search(r'^#{2,3}\s', content, re.MULTILINE):
            sections_with_heading += 1
    
    if total_sections == 0:
        return False, "Không có section nào"
    
    # Chỉ yêu cầu ít nhất 30% section có heading (không quá strict)
    rate = sections_with_heading / total_sections
    if rate < 0.3 and total_sections > 1:
        return False, f"Thiếu Markdown heading: chỉ {sections_with_heading}/{total_sections} section"
    
    return True, ""


def _kiem_tra_khong_co_loi_ro_rang(chapter: dict) -> tuple[bool, str]:
    """Phát hiện các dấu hiệu lỗi rõ ràng trong content."""
    phat_hien_loi = []
    
    for sec in chapter.get("sections", []):
        content = sec.get("content", "").lower()
        
        # Phát hiện nội dung lỗi hệ thống
        if "lỗi hệ thống" in content or "error" == content.strip():
            phat_hien_loi.append(f"Section '{sec.get('title')}' chứa nội dung lỗi")
        
        # Phát hiện content trống
        if not content.strip():
            phat_hien_loi.append(f"Section '{sec.get('title')}' trống")
        
        # Phát hiện content chỉ là placeholder
        if content.strip() in ["...", "n/a", "tbd", "todo"]:
            phat_hien_loi.append(f"Section '{sec.get('title')}' chứa placeholder")
    
    if phat_hien_loi:
        return False, " | ".join(phat_hien_loi)
    return True, ""


def kiem_tra_nhanh(chapter: dict, quy_mo: str = "tieu_chuan") -> dict:
    """
    Kiểm tra toàn diện nhanh không cần AI.
    
    Returns:
        {
            "hop_le": bool,        # True = pass tất cả
            "can_openai_rewrite": bool,  # True = lỗi nghiêm trọng, cần rewrite ngay
            "can_gemini_check": bool,    # True = có nghi vấn, nên để Gemini xem
            "issues": [str],       # Danh sách vấn đề phát hiện
            "chi_tiet": dict       # Chi tiết từng bước kiểm tra
        }
    """
    issues = []
    chi_tiet = {}
    
    # 1. Kiểm tra JSON
    json_ok, json_msg = _kiem_tra_json_hop_le(chapter)
    chi_tiet["json"] = {"ok": json_ok, "msg": json_msg}
    if not json_ok:
        issues.append(f"[JSON] {json_msg}")
    
    # 2. Kiểm tra lỗi rõ ràng
    loi_ok, loi_msg = _kiem_tra_khong_co_loi_ro_rang(chapter)
    chi_tiet["loi_ro_rang"] = {"ok": loi_ok, "msg": loi_msg}
    if not loi_ok:
        issues.append(f"[LỖI] {loi_msg}")
    
    # 3. Kiểm tra độ dài
    do_dai_ok, do_dai_msg, _ = _kiem_tra_do_dai(chapter, quy_mo)
    chi_tiet["do_dai"] = {"ok": do_dai_ok, "msg": do_dai_msg}
    if not do_dai_ok:
        issues.append(f"[ĐỘ DÀI] {do_dai_msg}")
    
    # 4. Kiểm tra citation
    cite_ok, cite_msg = _kiem_tra_co_citation(chapter)
    chi_tiet["citation"] = {"ok": cite_ok, "msg": cite_msg}
    if not cite_ok:
        issues.append(f"[CITATION] {cite_msg}")
    
    # 5. Kiểm tra markdown (ít nghiêm trọng hơn)
    md_ok, md_msg = _kiem_tra_cau_truc_markdown(chapter)
    chi_tiet["markdown"] = {"ok": md_ok, "msg": md_msg}
    if not md_ok:
        issues.append(f"[MARKDOWN] {md_msg}")
    
    # --- Quyết định hành động ---
    # Lỗi nghiêm trọng: cần OpenAI rewrite ngay (không cần Gemini trước)
    loi_nghiem_trong = not json_ok or not loi_ok
    
    # Lỗi trung bình: nên để Gemini kiểm tra
    loi_trung_binh = not do_dai_ok or not cite_ok or not md_ok
    
    hop_le = len(issues) == 0
    can_openai_rewrite = loi_nghiem_trong
    can_gemini_check = loi_trung_binh and not loi_nghiem_trong
    
    result = {
        "hop_le": hop_le,
        "can_openai_rewrite": can_openai_rewrite,
        "can_gemini_check": can_gemini_check,
        "issues": issues,
        "chi_tiet": chi_tiet
    }
    
    if issues:
        logger.debug(f"[LocalCheck] Chapter '{chapter.get('title')}': {issues}")
    
    return result


def nen_goi_gemini(
    ket_qua_kiem_tra: dict,
    chapter_index: int,
    total_chapters: int,
    ti_le_ngau_nhien: float = 0.3
) -> bool:
    """
    Quyết định thông minh có nên gọi Gemini Supervisor không.
    
    Logic:
    1. Local check phát hiện vấn đề → True
    2. Chương đầu và chương cuối (critical) → True  
    3. 30% random sampling → True
    4. Còn lại → False (tiết kiệm quota)
    
    Args:
        ket_qua_kiem_tra: Kết quả từ kiem_tra_nhanh()
        chapter_index: 0-indexed
        total_chapters: Tổng số chương
        ti_le_ngau_nhien: Tỉ lệ random sampling (mặc định 0.3 = 30%)
    """
    import random
    
    # 1. Local check phát hiện vấn đề trung bình → nhờ Gemini xem
    if ket_qua_kiem_tra.get("can_gemini_check"):
        logger.info(f"[Gemini Decision] Chapter {chapter_index+1}: Gọi Gemini vì local check phát hiện vấn đề")
        return True
    
    # 2. Chương đầu (intro) và chương cuối (kết luận) — critical
    is_first = chapter_index == 0
    is_last = chapter_index == total_chapters - 1
    if is_first or is_last:
        logger.info(f"[Gemini Decision] Chapter {chapter_index+1}: Gọi Gemini vì chương critical (đầu/cuối)")
        return True
    
    # 3. Random sampling 30%
    if random.random() < ti_le_ngau_nhien:
        logger.info(f"[Gemini Decision] Chapter {chapter_index+1}: Gọi Gemini (random 30% sampling)")
        return True
    
    logger.info(f"[Gemini Decision] Chapter {chapter_index+1}: SKIP Gemini (pass local check + không sampling)")
    return False
