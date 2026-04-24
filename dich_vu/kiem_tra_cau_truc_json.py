# -*- coding: utf-8 -*-
import json
import re
import logging

logger = logging.getLogger(__name__)

def tach_json(text: str) -> str:
    """
    Trích xuất JSON từ phản hồi của mô hình. 
    Hỗ trợ cả markdown fences ```json ... ``` và text rác bao quanh.
    """
    if not text:
        return ""

    # 1. Loại bỏ markdown code fences
    text2 = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text2 = re.sub(r"\s*```$", "", text2.strip())
    text2 = text2.strip()

    # 2. Nếu đã là JSON chuẩn (bắt đầu { [ và kết thúc } ])
    if (text2.startswith("{") and text2.endswith("}")) or \
       (text2.startswith("[") and text2.endswith("]")):
        return text2

    # 3. Tìm index của object/array đầu tiên và cuối cùng
    start_obj = text2.find("{")
    start_arr = text2.find("[")
    
    start_idx = -1
    if start_obj != -1 and start_arr != -1:
        start_idx = min(start_obj, start_arr)
    else:
        start_idx = max(start_obj, start_arr)
        
    if start_idx != -1:
        char_start = text2[start_idx]
        char_end = "}" if char_start == "{" else "]"
        end_idx = text2.rfind(char_end)
        if end_idx != -1 and end_idx > start_idx:
            # Lấy đúng đoạn JSON, bỏ qua rác phía sau (như dấu chấm hoặc text giải thích)
            return text2[start_idx:end_idx+1]

    return text2

def clean_title_numbering(t: str) -> str:
    """
    Xóa các số thứ tự thừa ở đầu tiêu đề (8.3. , Chapter 1: , 1.1 ...)
    để tránh bị trùng lặp với giao diện (Hotfix V5.4)
    """
    if not t: return ""
    # 1. Xóa "Chương X: " hoặc "Chapter X: "
    t = re.sub(r'^(?:Chương|Chapter)\s*\d+[:\.\s\-]*', '', t, flags=re.IGNORECASE)
    # 2. Xóa các số thứ tự kiểu 8.3.1 hoặc 1.1 hoặc 8.
    t = re.sub(r'^[\d\.\s\-a-zA-Z]+\s*[:\.]\s*', '', t)
    return t.strip()

def safe_parse_json(text: str) -> dict | list | None:
    """
    Parse JSON an toàn, không raise exception để Orchestrator xử lý.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # Thử lại bằng cách tách JSON kỹ hơn nếu lần đầu fail
        try:
            cleaned = tach_json(text)
            return json.loads(cleaned)
        except Exception:
            return None

def safe_json_fix(data, expected_structure):
    """
    DETERMINISTIC GUARD (PRO): Đảm bảo JSON đầu ra luôn khớp với cấu trúc mong muốn.
    - Xử lý Auto-normalization: Chuyển đổi sections từ chuỗi (string) sang đối tượng (dict).
    - Xây dựng sections_text chuẩn: Giữ trọn vẹn context cho tầng Audit.
    """
    if not isinstance(data, dict): 
        logger.warning(f"[AI FORMAT ERROR] Expected dict, got {type(data)}. Raw: {str(data)[:200]}")
        data = {}
    
    fixed_data = {
        "title": clean_title_numbering(data.get("title") or expected_structure.get("title") or "Chương không tiêu đề"),
        "sections": [],
        "status": data.get("status", "success")
    }
    
    raw_sections = data.get("sections", [])
    if not isinstance(raw_sections, list):
        logger.warning(f"[AI FORMAT ERROR] 'sections' should be list, got {type(raw_sections)}")
        raw_sections = []

    # 1. Thủ tục Normalize: string -> dict
    clean_sections = []
    for s in raw_sections:
        if isinstance(s, dict):
            # Clean title ngay lập tức
            if "title" in s: s["title"] = clean_title_numbering(s["title"])
            
            # Đảm bảo content là string (Hotfix V5.5)
            content = s.get("content", "")
            if isinstance(content, list):
                content = "\n".join([str(item) for item in content])
            elif not isinstance(content, str):
                content = str(content)
            s["content"] = content.strip()
            
            clean_sections.append(s)
        elif isinstance(s, str):
            logger.warning(f"[AI FORMAT ERROR] Section returned as string in '{fixed_data['title']}'")
            clean_sections.append({
                "title": "Nội dung bài học",
                "content": s.strip(),
                "citations": []
            })
            
    # 2. Thủ tục Map & Gap-Fill với Title Normalization (Hotfix V5.2)
    def normalize_title(t):
        if not t: return ""
        # Sử dụng clean_title_numbering trước khi normalize sâu (Hotfix V5.4)
        t = clean_title_numbering(t)
        # Xóa các ký tự đặc biệt và lowercase
        t = re.sub(r'[^\w\s]', '', t).lower().strip()
        return t

    # Tạo map với key đã normalize và lưu index gốc để tránh trùng (Hotfix V5.3)
    actual_sections_data = [] # List of (norm_title, section_dict)
    for s in clean_sections:
        if isinstance(s, dict) and s.get("title"):
            actual_sections_data.append((normalize_title(s.get("title")), s))

    used_indices = set()
    expected_sections = expected_structure.get("sections", [])
    for exp_sec in expected_sections:
        orig_title = exp_sec.get("title")
        norm_exp_t = normalize_title(orig_title)
        
        found_match = False
        # 1. Thử khớp chính xác (Ưu tiên hàng đầu)
        for i, (act_norm_t, act_sec) in enumerate(actual_sections_data):
            if i not in used_indices and norm_exp_t == act_norm_t:
                fixed_data["sections"].append(act_sec)
                used_indices.add(i)
                found_match = True
                break
        
        if found_match: continue

        # 2. Thử khớp mờ (Partial Match)
        for i, (act_norm_t, act_sec) in enumerate(actual_sections_data):
            if i not in used_indices and (norm_exp_t in act_norm_t or act_norm_t in norm_exp_t):
                fixed_data["sections"].append(act_sec)
                used_indices.add(i)
                found_match = True
                break
            
        if not found_match:
            logger.warning(f"[Mapping Fail] Could not find match for section '{orig_title}' in Chapter '{fixed_data['title']}'")
            fixed_data["sections"].append({
                "title": clean_title_numbering(orig_title),
                "content": "[Nội dung đang được hệ thống xử lý bổ sung từ nguồn...]",
                "citations": []
            })
            
    # 3. Xây dựng sections_text PRO (Giữ Context cho Audit)
    fixed_data["sections_text"] = "\n".join(
        f"{s.get('title','')}: {s.get('content','')}" 
        for s in fixed_data["sections"]
    )
    
    return fixed_data

def fallback_raw_facts(chap_info, relevant_passages):
    """
    TẦNG DỰ PHÒNG CUỐI (PRO): Nối dữ liệu thô nếu AI hoàn toàn thất bại.
    Đảm bảo 100% không bao giờ crash downstream.
    """
    raw_text = "\n\n[HỆ THỐNG]: Cả OpenAI và Gemini cứu hộ đều gặp lỗi trình bày mục này. Dưới đây là tóm lược dữ liệu thô từ nguồn tham khảo:\n" + \
               "\n".join([f"- {p.get('text','')}" for p in relevant_passages[:5]])
    
    fixed_data = {
        "title": chap_info.get("title", "Chương dự phòng"),
        "sections": [],
        "status": "fallback",
        "is_fallback": True,
        "sections_text": raw_text
    }
    
    for sec in chap_info.get("sections", []):
        fixed_data["sections"].append({
            "title": sec.get("title"),
            "content": raw_text,
            "citations": [],
            "is_fallback": True
        })
    return fixed_data

def detect_hallucination_simple(text, facts_text):
    """
    HEURISTIC GUARD: Kiểm tra từng câu chống ảo giác.
    Nếu câu chứa số/thực thể mà không tìm thấy trong bộ Facts thô -> Nghi vấn.
    """
    if not text or not facts_text: return []
    
    import re
    sentences = re.split(r'[.!?\n]', text)
    facts_lower = facts_text.lower()
    suspicious = []
    
    for s in sentences:
        s = s.strip()
        if len(s) < 20: continue
        
        # 1. Trích xuất số và thực thể (từ viết hoa)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', s)
        entities = re.findall(r'\b[A-ZÀ-Ỹ][a-zà-ỹ]+\b', s)
        
        # Loại bỏ các thực thể quá ngắn hoặc phổ biến đầu câu
        threats = [t for t in numbers + entities if len(t) > 2]
        
        if not threats: continue
        
        # 2. Kiểm tra xem các threat này có trong facts không
        missing = [t for t in threats if t.lower() not in facts_lower]
        
        # Nếu > 40% thực thể trong câu là lạ -> Đánh dấu câu đó nghi vấn
        if len(missing) / len(threats) > 0.4:
            suspicious.append(s)
            
    return suspicious

def kiem_tra_json(obj: dict):
    """
    Validation tối thiểu để export không lỗi.
    Nếu thiếu -> raise ValueError.
    """
    if not isinstance(obj, dict):
        raise ValueError("Output không phải dict JSON")

    for k in ["topic", "terms", "outline", "book_vi", "references"]:
        if k not in obj:
            raise ValueError(f"Thiếu trường '{k}' trong JSON")
    return True

def safe_section_fix(raw_section: dict, expected_title: str):
    """
    Chuẩn hóa dữ liệu của một MỤC duy nhất (Micro-Fix). (V5.8/V18.3)
    Đảm bảo tiêu đề sạch, nội dung là chuỗi, và auto-filter các fact low-confidence.
    """
    if not isinstance(raw_section, dict):
        return {
            "title": clean_title_numbering(expected_title),
            "content": str(raw_section),
            "citations": []
        }
    
    # 1. Clean Title
    title = raw_section.get("title") or expected_title
    title = clean_title_numbering(title)
    
    # 2. Normalize Content (Hotfix V5.5 sync)
    content = raw_section.get("content", "")
    if isinstance(content, list):
        content = "\n".join([str(item) for item in content])
    elif not isinstance(content, str):
        content = str(content)

    # 3. Fact-First Validation (V18.3/V18.9)
    fact_mappings = raw_section.get("fact_mappings", [])
    if isinstance(fact_mappings, list):
        for mapping in fact_mappings:
            if isinstance(mapping, dict):
                sid = mapping.get("source_id")
                conf = mapping.get("confidence", 1.0)
                # Auto-filter: gỡ nhãn ID nếu confidence < 0.8
                if conf < 0.8 and sid:
                    logger.warning(f"[Auto-Filter] Loại bỏ source_id [{sid}] ở mục '{title}' do confidence ({conf}) < 0.8")
                    content = re.sub(rf'\[{sid}\]', '', content)

    # 4. Citations extraction (V5.5 sync)
    citations = raw_section.get("citations", [])
    if not citations and fact_mappings:
        extracted = set(re.findall(r'\[(\w+)\]', content))
        for fid in extracted:
            citations.append({"id": fid, "url": ""})
            
    # Dọn dẹp khoảng trắng dư thừa do replace
    content = re.sub(r'\s+([\.!,:;?])', r'\1', content)
    gen_mode = raw_section.get("generation_mode", "normal")
            
    return {
        "title": title,
        "content": content.strip(),
        "citations": citations,
        "generation_mode": gen_mode,
        "summary": raw_section.get("summary", ""),
        "fact_mappings": fact_mappings
    }

def convert_fact_tags_to_html(content: str, global_map: dict) -> str:
    """
    BACKEND CITATION ENGINE (V21.2): 
    Biến đổi [factN] thành <sup class="citation"><a href="..." title="...">[N]</a></sup>.
    - Deterministic: 100% chính xác dựa trên global_map.
    - UX: Hover hiện tiêu đề nguồn Wikipedia.
    """
    if not content or not global_map: return content
    
    # Tìm tất cả các thẻ [factN]
    pattern = r"\[fact(\d+)\]"
    matches = re.finditer(pattern, content)
    
    # Sắp xếp các cụm khớp từ cuối lên đầu để không làm lệch index khi replace
    sorted_matches = sorted(list(matches), key=lambda x: x.start(), reverse=True)
    
    for m in sorted_matches:
        full_tag = m.group(0)
        fid = m.group(1)
        
        # Mapping: fact_id trong content tương ứng với doc index trong global_map/passages
        # Lưu ý: OpenAI thường gán d1 -> fact1, d2 -> fact2...
        doc_key = f"fact{fid}"
        
        if doc_key in global_map:
            doc_info = global_map[doc_key]
            url = doc_info.get("url", "#")
            title = doc_info.get("title", "Nguồn tham khảo")
            
            # HTML chuyên nghiệp: Hover thấy title, click sang tab mới
            replacement = (
                f'<sup class="citation">'
                f'<a href="{url}" target="_blank" title="{title}">[{fid}]</a>'
                f'</sup>'
            )
            content = content[:m.start()] + replacement + content[m.end():]
        else:
            # Nếu không có mapping, xóa tag ẩn để tránh rác (Sanitize)
            content = content[:m.start()] + "" + content[m.end():]
            
    return content
