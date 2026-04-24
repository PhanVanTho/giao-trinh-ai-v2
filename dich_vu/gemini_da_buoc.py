# -*- coding: utf-8 -*-
import json
import re
import time
import os
import logging
from google import genai
from google.genai import types
from cau_hinh import CauHinh

logger = logging.getLogger(__name__)

def _get_gemini_response(prompt: str, api_keys: list, model_name: str = CauHinh.SUPERVISOR_MODEL_LITE):
    """
    ENTERPRISE GEMINI CALLER:
    Xử lý xoay vòng API Key, quản lý Quota (429) và Retry tự động.
    """
    max_retries = 3
    last_error = None
    
    # Tạo bản sao của api_keys để trộn (không ảnh hưởng list gốc)
    shuffled_keys = list(api_keys)
    
    for attempt in range(max_retries):
        import random
        random.shuffle(shuffled_keys) # Smart Key Shuffle: Phân tán tải trọng cho các luồng song song
        
        for current_key in shuffled_keys:
            client = genai.Client(api_key=current_key)
            try:
                # Micro-Jitter: Tránh việc 3 luồng bắn API đúng cùng 1 mili-giây
                time.sleep(random.uniform(0.1, 0.8))
                
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                    ),
                )
                return resp.text or ""
            except Exception as e:
                last_error = e
                error_str = str(e)
                if any(x in error_str for x in ["429", "RESOURCE_EXHAUSTED", "quota"]):
                    logger.warning(f"[Gemini] Quota exceeded key ...{current_key[-4:]}. Switching...")
                    continue
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    logger.warning(f"[Gemini] 503 OVERLOAD on key ...{current_key[-4:]}. Backing off... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(random.uniform(2.0, 4.0) + attempt * 2) # Jittered Backoff
                    continue
                else:
                    logger.error(f"[Gemini] Error: {e}")
                    time.sleep(1)
                    break
        if attempt < max_retries - 1:
            time.sleep(random.uniform(4.0, 6.0) + attempt * 5)
    raise last_error

def _tach_json(text: str) -> str:
    """
    Trích xuất JSON từ phản hồi của mô hình.
    Hỗ trợ cả JSON Object {} và JSON Array [].
    """
    if not text:
        raise ValueError("Phản hồi rỗng")

    # bỏ code fences
    text2 = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text2 = re.sub(r"\s*```$", "", text2.strip())
    text2 = text2.strip()

    # nếu đã là JSON
    if (text2.startswith("{") and text2.endswith("}")) or \
       (text2.startswith("[") and text2.endswith("]")):
        return text2

    # Tìm index đầu tiên của { hoặc [
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
            return text2[start_idx:end_idx+1]

    raise ValueError("Không tìm thấy JSON trong phản hồi")

def tao_dan_y(
    chu_de: str,
    corpus_passages: list,
    model: str = CauHinh.GEMINI_MODEL,
    api_key: str | list = None,
    so_chuong_min: int = 8,
    so_chuong_max: int = 12,
    che_do: str = "auto",
    danh_sach_chuong: str = "",
    quy_mo: str = "tieu_chuan",
):
    """
    Bước 1: Tạo dàn ý (chapters, sections) và danh sách thuật ngữ bẳng Gemini.
    """
    # Điều chỉnh số chương dựa trên quy mô nếu đang dùng mặc định (8-12)
    if so_chuong_min == 8 and so_chuong_max == 12:
        mapping = {
            "can_ban": (4, 6),
            "chuyen_sau": (13, 18),
            "tieu_chuan": (8, 12)
        }
        so_chuong_min, so_chuong_max = mapping.get(quy_mo, (8, 12))

    api_keys = api_key if isinstance(api_key, list) else [api_key]
    if not api_keys or not api_keys[0]:
        raise RuntimeError("Thiếu GEMINI_API_KEY")


    packed = []
    for p in corpus_passages:
        packed.append({
            "pid": p.get("pid"),
            "title": p.get("title"),
            "text": p.get("text")[:500] + "...", # Chỉ lấy đoạn đầu để tiết kiệm token cho outline
            "url": p.get("url")
        })
    
    # Xử lý chế độ tạo
    cau_truc_prompt = ""
    if che_do == "custom_so_chuong":
        cau_truc_prompt = f"""
   - Dựa trên danh sách thuật ngữ, hãy gom nhóm chúng thành các chủ đề lớn.
   - BẮT BUỘC: Giáo trình phải có CHÍNH XÁC {so_chuong_max} chương. (Lưu ý: số lượng chương bạn tạo ra PHẢI ĐÚNG LÀ {so_chuong_max}).
   - Đảm bảo MỌI thuật ngữ quan trọng đều được phân bổ vào các chương này một cách hợp lý."""
    elif che_do == "custom_danh_sach" and danh_sach_chuong:
        cau_truc_prompt = f"""
   - NGƯỜI DÙNG ĐÃ YÊU CẦU DANH SÁCH CHƯƠNG CỐ ĐỊNH NHƯ SAU:
     {danh_sach_chuong}
   - BẮT BUỘC: Bạn PHẢI TẠO CẤU TRÚC ĐÚNG THEO DANH SÁCH CHƯƠNG NÀY (Đúng số lượng chương, đúng tên chương).
   - Hãy phân bổ thông minh các thuật ngữ đã lọc vào các chương đã cố định ở trên."""
    else: # auto
        if quy_mo == "can_ban":
            cau_truc_prompt = f"""
   - Dựa trên thuật ngữ đã lọc, gom thành các CHỦ ĐỀ LỚN.
   - GIÁO TRÌNH CĂN BẢN: Tối đa {so_chuong_max} chương, nội dung NGẮN GỌN, tập trung vào ý chính."""
        elif quy_mo == "chuyen_sau":
            cau_truc_prompt = f"""
   - Dựa trên 100 thuật ngữ đã lọc, hãy gom nhóm thành các CHỦ ĐỀ LỚN và ĐÀO SÂU hoàn toàn.
   - GIÁO TRÌNH CHUYÊN SÂU: Tạo TỐI THIỂU {so_chuong_min} chương, phủ rộng MỌI khía cạnh học thuật.
   - Mỗi chương phải có nhiều mục con chi tiết (4-6 mục/chương).
   - Đảm bảo MỌI thuật ngữ đều được phân bổ vào ít nhất một chương."""
        else:  # tieu_chuan
            cau_truc_prompt = """
   - Dựa trên 100 thuật ngữ đã lọc, hãy gom nhóm chúng thành các CHỦ ĐỀ LỚN (Clusters).
   - Mỗi chủ đề lớn sẽ trở thành một CHƯƠNG.
   - Số lượng chương = số lượng cụm chủ đề tìm được (không bắt buộc phải là 5, 8 hay 10).
     + Nếu chủ đề rộng => nhiều chương.
     + Nếu chủ đề hẹp => ít chương.
     + Ưu tiên sự đầy đủ và logic học thuật hơn là số lượng chương cố định.
   - Đảm bảo MỌI thuật ngữ quan trọng đều được phân bổ vào ít nhất một chương."""

    prompt = f"""
Bạn là chuyên gia biên soạn giáo trình đại học.
Chủ đề: {chu_de}

Dữ liệu tham khảo (CORPUS_SUMMARY):
{json.dumps(packed, ensure_ascii=False)}

NHIỆM VỤ:
1. PHÂN TÍCH THUẬT NGỮ (Term Analysis):
   - Trích xuất 100 thuật ngữ chuyên ngành quan trọng nhất và ĐƯỢC LẶP LẠI NHIỀU NHẤT trong dữ liệu.
   - Sắp xếp chúng theo thứ tự: (Mức độ liên quan x Tần suất xuất hiện) giảm dần.
   - CHỈ sử dụng các thuật ngữ CÓ trong dữ liệu tham khảo. TUYỆT ĐỐI KHÔNG BỊA RA THUẬT NGỮ.

2. XÂY DỰNG CẤU TRÚC (Structure Building):{cau_truc_prompt}
   - QUY TẮC ĐẶT TÊN CHƯƠNG/MỤC: TUYỆT ĐỐI KHÔNG ĐƯỢC thêm số thứ tự, prefix (ví dụ: "1.", "1.1", "Chương 1", "Mục 2",...) vào tên. CHỈ trả về tiêu đề thuần văn bản (Ví dụ xuất: "Khái niệm AI" thay vì "1.1 Khái niệm AI").

3. CRITICAL JSON FORMATTING RULES:
   - Return valid JSON only. Ensure proper escaping.

ĐẦU RA (JSON format):
{{
  "topic": "{chu_de}",
  "terms": [
    {{"term": "Tên thuật ngữ", "meaning": "Giải thích ngắn gọn"}}
  ],
  "outline": [
    {{
      "chapter_index": 1,
      "title": "Tên chương (Tương ứng cụm thuật ngữ 1)",
      "sections": [
        {{"title": "Tên mục", "recommended_pids": ["pid1", "pid2"]}}
      ]
    }}
  ]
}}
Tuyệt đối chỉ trả về JSON BẮT BUỘC ĐÚNG CẤU TRÚC TRÊN (phải có key 'outline').
"""

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        for current_key in api_keys:
            client = genai.Client(api_key=current_key)
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text or ""
                return json.loads(_tach_json(text))
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"[WARN] Gemini Step 1: Quota exceeded with key ...{current_key[-4:]}. Switching key...")
                    continue
                else:
                    print(f"[WARN] Gemini Step 1 error: {e}")
                    time.sleep(2)
                    break
        else:
            if attempt < max_retries - 1:
                print(f"[WARN] All Gemini keys exhausted for Step 1. Sleeping 60s...")
                time.sleep(60)
            else:
                raise last_error
    raise last_error

def viet_noi_dung_nhom_chuong(
    chu_de: str,
    danh_sach_chuong: list,
    terms: list,
    relevant_passages: list,
    model: str = CauHinh.GEMINI_MODEL,
    api_key: str | list = None,
    quy_mo: str = "tieu_chuan",
    failure_memory: str = None,
):
    """
    Bước 2: Viết nội dung cho MỘT NHÓM CHƯƠNG cụ thể bằng Gemini.
    """
    api_keys = api_key if isinstance(api_key, list) else [api_key]
    if not api_keys or not api_keys[0]:
        raise RuntimeError("Thiếu GEMINI_API_KEY")


    # Chỉ lấy Top 30 facts liên quan nhất
    selected_passages = relevant_passages[:30]
    corpus_text = "FACTS:\n"
    for i, p in enumerate(selected_passages):
        p_id = p.get("id", i + 1)
        p_text = p.get("text", "")
        corpus_text += f"[{p_id}] {p_text}\n"

    # Ràng buộc độ dài viết theo quy mô
    if quy_mo == "can_ban":
        r = "REGIME: BASIC (Short). Max 200 words/section."
    elif quy_mo == "chuyen_sau":
        r = "REGIME: ADVANCED (Deep). Min 1000 words/section."
    else:  # tieu_chuan
        r = "REGIME: STANDARD. 400-700 words/section."

    memory_prompt = ""
    if failure_memory:
        memory_prompt = f"\n⚠️ PREVIOUS ATTEMPT FAILED. ISSUES TO FIX:\n{failure_memory}\n"

    prompt = f"""You are a professional ACADEMIC AUTHOR for a university textbook on "{chu_de}".
{r}
{memory_prompt}

{corpus_text}

STRICT RULES:
1. GROUNDING: You MUST use at least 70% of the provided FACTS. Coverage is your primary KPI.
2. CITATION (MANDATORY): Every factual claim MUST include a citation [number]. 
3. DENSITY: Aim for 2-3 citations [ID] per paragraph.
4. NO HALLUCINATION: Use ONLY the provided FACTS. Do not invent names, dates, or technical details.
5. ACADEMIC TONE: Formal Vietnamese. Permanent modality preservation.
6. FORMAT: Output valid JSON. Each section MUST have "content" and "citations" (list of IDs used).

FAIL CONDITIONS:
- Coverage < 70%.
- Missing citations.
- Hallucinated content.

SECTIONS TO FILL: {json.dumps([{"title": c["title"], "sections": [s["title"] for s in c["sections"]]} for c in danh_sach_chuong], ensure_ascii=False)}"""

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        for current_key in api_keys:
            client = genai.Client(api_key=current_key)
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text or "{}"
                return {
                    "status": "success",
                    "raw_text": text,
                    "data": None # Sẽ được parse ở Orchestrator
                }
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"[Worker Gemini] Quota hit (Key ...{current_key[-4:]}). Switching...")
                    continue
                else:
                    logger.error(f"[Worker Gemini] Error: {e}")
                    break
        time.sleep(2)

    return {
        "status": "error",
        "error_type": "all_keys_failed",
        "message": str(last_error),
        "raw_text": ""
    }
    raise last_error


def kiem_tra_ao_giac_nhom_chuong(
    chu_de: str,
    danh_sach_chapter_content: list,
    relevant_passages: list,
    model: str = CauHinh.GEMINI_MODEL,
    api_key: str | list = None
):
    """
    Bước 3 (Optional): Kiểm tra ảo giác bằng Gemini cho MỘT NHÓM CHƯƠNG.
    """
    api_keys = api_key if isinstance(api_key, list) else [api_key]
    if not api_keys or not api_keys[0]:
        raise RuntimeError("Thiếu GEMINI_API_KEY")


    packed_corpus = []
    for p in relevant_passages:
        packed_corpus.append({
            "url": p.get("url"),
            "text": p.get("text")
        })

    prompt = f"""
Bạn là một CHUYÊN GIA KIỂM ĐỊNH CHẤT LƯỢNG HỌC THUẬT (Fact-checker).
Nhiệm vụ của bạn là kiểm tra xem nội dung văn bản dưới đây có bị "ảo giác" (hallucination - bịa đặt thông tin không có trong nguồn) hay không.

CHỦ ĐỀ: "{chu_de}"
SỐ LƯỢNG CHƯƠNG CẦN KIỂM: {len(danh_sach_chapter_content)}

NỘI DUNG CẦN KIỂM TRA (Do AI cấp dưới vừa viết):
{json.dumps([{"title": c.get('title'), "sections": c.get('sections', [])} for c in danh_sach_chapter_content], ensure_ascii=False)}

NGUỒN DỮ LIỆU GỐC (CORPUS CHÍNH XÁC):
{json.dumps(packed_corpus, ensure_ascii=False)}

YÊU CẦU KIỂM TRA & CHỈNH SỬA:
1. Đọc kỹ từng câu trong [NỘI DUNG CẦN KIỂM TRA].
2. Đối chiếu TỪNG TỪ với [NGUỒN DỮ LIỆU GỐC].
3. XÓA BỎ HOÀN TOÀN các câu bịa đặt không có trong nguồn.
4. Giữ nguyên văn phong học thuật, cách chia đoạn, nhưng ĐẢM BẢO CHÍNH XÁC 100% so với nguồn.
5. Vẫn phải giữ nguyên định dạng JSON đầu ra (là mảng gồm các chương).
   
6. CRITICAL JSON FORMATTING RULES:
   - Return valid JSON only. Ensure proper escaping.

ĐẦU RA BẮT BUỘC (JSON format array):
[
  {{
    "title": "Tên Chương 1",
    "sections": [
      {{
        "title": "Tên mục (giữ nguyên)",
        "content": "Nội dung ĐÃ ĐƯỢC KIỂM TRA VÀ SỬA LỖI ẢO GIÁC...",
        "citations": ["url1", "url2"] 
      }}
    ]
  }},
  {{
    "title": "Tên Chương 2",
    "sections": [...]
  }}
]

Tuyệt đối chỉ trả về JSON array, không giải thích gì thêm.
"""

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        for current_key in api_keys:
            client = genai.Client(api_key=current_key)
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2, # Temperature thấp để fact-checking chính xác
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text or ""
                return json.loads(_tach_json(text))
            except Exception as e:
                 chapters_titles = [c.get("title") for c in danh_sach_chapter_content]
                 last_error = e
                 error_str = str(e)
                 if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                     print(f"[WARN] Gemini Fact-check: Quota exceeded with key ...{current_key[-4:]} for chapters {chapters_titles}. Switching key...")
                     continue
                 else:
                     print(f"[WARN] Gemini Fact-check error (Chapters {chapters_titles}): {e}")
                     time.sleep(2)
                     break
        else:
             if attempt < max_retries - 1:
                print(f"[WARN] All Gemini keys exhausted for Fact-check. Sleeping 60s...")
                time.sleep(60)
             else:
                raise last_error
    raise last_error


def gemini_fix_json(bad_text, api_keys=None):
    """
    DETERMINISTIC JSON FIXER (GEMINI FLASH): 
    Sử dụng phản hồi của AI để cứu vãn các cấu trúc JSON bị hỏng nặng.
    """
    if not bad_text: return "{}"
    from .kiem_tra_cau_truc_json import tach_json
    
    prompt = f"""Bạn là một chuyên gia sửa lỗi JSON. 
Dưới đây là một đoạn văn bản chứa dữ liệu JSON bị hỏng hoặc có văn bản thừa:
---
{bad_text}
---
NHIỆM VỤ: Hãy trích xuất và sửa lỗi cú pháp để trả về một chuỗi JSON hợp lệ 100%. 
CHỈ TRẢ VỀ JSON, không giải thích, không thêm text ngoài.
"""
    # Dùng Flash cho rẻ và nhanh
    res = goi_gemini_da_buoc(prompt, api_keys=api_keys, model_name=CauHinh.GEMINI_MODEL, max_tokens=2048)
    if res["status"] == "success":
        return tach_json(res["raw_text"])
    return bad_text

def goi_gemini_da_buoc(prompt: str, api_keys: list = None, model_name: str = CauHinh.SUPERVISOR_MODEL_LITE, max_tokens: int = 2048):
    """
    Hàm helper trung tâm để gọi Gemini với cơ chế retry và xoay vòng key.
    """
    if not api_keys:
        api_keys = CauHinh.GEMINI_API_KEYS
    
    try:
        raw_text = _get_gemini_response(prompt, api_keys, model_name=model_name)
        return {"status": "success", "raw_text": raw_text}
    except Exception as e:
        logger.error(f"[Gemini Helper] Error: {e}")
        return {"status": "error", "message": str(e)}

def generate_related_topics_gemini(chu_de: str, current_results_titles: list, quy_mo: str, api_keys: list = None) -> list:
    """
    Kéo thêm từ khóa bài viết liên quan khi dữ liệu thiếu (V21.3 Diamond).
    Hỗ trợ Expansion Level 1 & 2 với bộ lọc Anti-Drift chuyên sâu.
    """
    if not api_keys: api_keys = CauHinh.GEMINI_API_KEYS
    
    logger.info(f"[EKRE-Expansion] AI mining related topics for: {chu_de}")
    
    # Prompt tinh chỉnh để tránh trôi sang chủ đề "Giáo dục/Sư phạm"
    prompt = f"""Dựa trên chủ đề chính: "{chu_de}".
Chúng tôi đang xây dựng giáo trình chuyên sâu cấp độ "{quy_mo}" và đã có các bài viết: {json.dumps(current_results_titles[:15], ensure_ascii=False)}

NHIỆM VỤ: Liệt kê 5-8 tiêu đề bài viết khác trên Wikipedia (Tiếng Việt hoặc Tiếng Anh chuyển sang Việt) liên quan CỐT LÕI đến lý thuyết, cơ chế hoặc ứng dụng của "{chu_de}".

STRICT RULES:
1. CHỈ sinh từ khóa cùng lĩnh vực học thuật chuyên ngành của "{chu_de}".
2. TUYỆT ĐỐI KHÔNG mở rộng sang lĩnh vực giáo dục, phương pháp giảng dạy, chương trình học (pedagogy, education, teaching methods).
3. KHÔNG sử dụng các từ khóa chung chung.
4. Trả về danh sách tiêu đề cách nhau bởi dấu phẩy.

Ví dụ cho "Trí tuệ nhân tạo": Mạng nơ-ron nhân tạo, Học sâu, Logic mờ, Xử lý ngôn ngữ tự nhiên.
"""
    
    res = goi_gemini_da_buoc(prompt, api_keys=api_keys, model_name=CauHinh.GEMINI_MODEL, max_tokens=300)
    
    if res["status"] == "success":
        # Parse và Normalize
        raw_topics = [t.strip() for t in res["raw_text"].split(",") if len(t.strip()) > 2]
        
        # 1. Loại bỏ các từ khóa dính líu đến giáo dục (Hard filter)
        forbidden = ["giáo dục", "sư phạm", "giảng dạy", "học tập", "trường học", "curriculum", "pedagogy", "education", "teaching"]
        filtered_topics = [t for t in raw_topics if not any(f in t.lower() for f in forbidden)]
        
        # 2. Loại bỏ trùng lặp với danh sách hiện tại
        current_lower = {t.lower() for t in current_results_titles}
        final_topics = [t for t in filtered_topics if t.lower() not in current_lower]
        
        logger.info(f"[Expansion] Found {len(final_topics)} new focused topics.")
        return final_topics[:8]
        
    return []

def viet_noi_dung_chuong(
    chu_de: str,
    chapter_info: dict,
    terms: list,
    relevant_passages: list,
    model: str = CauHinh.GEMINI_MODEL_LITE,
    api_key: str | list = None,
    quy_mo: str = "tieu_chuan",
    failure_memory: str = None,
    section_title: str = None,
):
    """
    Phiên bản viết 1 chương đơn lẻ của Gemini (Tương thích với Supervisor).
    """
    # Gemini dùng hàm viet_noi_dung_nhom_chuong bên trong nhưng giới hạn 1 chương
    res = viet_noi_dung_nhom_chuong(
        chu_de=chu_de,
        danh_sach_chuong=[chapter_info],
        terms=terms,
        relevant_passages=relevant_passages,
        model=model,
        api_key=api_key,
        quy_mo=quy_mo,
        failure_memory=failure_memory
    )
    if res and len(res) > 0:
        # Nếu có failure_memory, chúng ta có thể thực hiện một pass sửa lỗi nhỏ ở đây
        # Hỗ trợ cấu trúc trả về đúng của Supervisor
        return res[0]
    return {"title": chapter_info.get("title"), "sections": []}
def viet_noi_dung_muc_gemini(
    chu_de: str,
    chapter_title: str,
    section_title: str,
    relevant_passages: list,
    api_keys: list = None,
    prev_section_summary: str = None,
    **kwargs
):
    """
    GEMINI RESCUE WRITER:
    Dùng làm phương án dự phòng cuối cùng khi OpenAI gặp lỗi trình bày (JSON Error).
    Gemini 3.1 Flash Lite rất ít khi lỗi format JSON.
    """
    if not api_keys: raise RuntimeError("Missing GEMINI_API_KEYS for Rescue")

    fact_ids = []
    facts_text = "SOURCE FACTS:\n"
    for i, p in enumerate(relevant_passages[:10]):
        p_id = str(p.get("id", i + 1))
        p_text = p.get("text", "")
        facts_text += f"[{p_id}] {p_text}\n"
        fact_ids.append(p_id)

    prompt = f"""You are a university professor writing ONE SECTION for "{chu_de}".
Chapter: "{chapter_title}"
Section: "{section_title}"

RESCUE MISSION: OpenAI failed to format this section. You must produce a clean, structured JSON response.

STRICT RULES:
1. CITATIONS: You MUST include [ID] tags for every factual claim.
2. NO TITLES: Do not include the Chapter or Section title in the content.
3. CONTENT ONLY: Start directly with the narrative.
4. JSON: Return ONLY valid JSON in the format below.

{facts_text}

MANDATORY FACT IDs: {", ".join(fact_ids)}

RESPONSE FORMAT (JSON ONLY):
{{
  "title": "{section_title}",
  "content": "Paragraph 1 with [ID]... \\n\\nParagraph 2...",
  "summary": "Brief summary"
}}"""

    try:
        # Sử dụng Flash Lite 3.1 để có độ ổn định và tốc độ cao nhất
        from google import genai
        from google.genai import types
        
        raw_text = _get_gemini_response(prompt, api_keys, model_name=CauHinh.SUPERVISOR_MODEL_LITE)
        if not raw_text: return {"status": "error", "message": "Empty response"}
        
        return {"status": "success", "raw_text": raw_text}
    except Exception as e:
        logger.error(f"[Gemini Rescue] Error: {e}")
        return {"status": "error", "message": str(e)}

def gemini_critic_agent(topic: str, doc_title: str, doc_text: str, api_keys: list = None) -> dict:
    """
    AGENT 2: THE CRITIC (Thẩm định viên LLM)
    Đọc văn bản và chấm điểm độ chính xác học thuật dựa trên chủ đề.
    """
    if not api_keys: api_keys = CauHinh.GEMINI_API_KEYS
    if not doc_text: return {"is_approved": False, "confidence_score": 0, "reason": "Văn bản rỗng."}
    
    # Giới hạn độ dài text (khoảng 2500 ký tự đầu để tiết kiệm token và tăng tốc)
    text_to_eval = doc_text[:2500]
    
    prompt = f"""You are an ACADEMIC CRITIC AGENT (Agent 2 in an automated pipeline).
Your job is to read an excerpt from a crawled article and strictly evaluate whether it contains high-quality, relevant academic information about the topic.

TOPIC OF THE TEXTBOOK: "{topic}"
ARTICLE TITLE: "{doc_title}"
EXCERPT TO EVALUATE:
---
{text_to_eval}
---

EVALUATION RUBRIC:
1. Relevance: Is the excerpt factually related to the core subject of "{topic}"? It does NOT need to cover the complex academic angle perfectly, as long as it provides foundational facts, history, or concepts about the subject.
2. Quality: Is the tone informative, encyclopedic, and factual?
3. Noise: Does it contain too much irrelevant information, spam, or disambiguation (trang định hướng)?

YOUR RESPONSE FORMAT MUST BE PURE JSON:
{{
  "is_approved": true or false,
  "confidence_score": <int from 0 to 100>,
  "reason": "<short 1-sentence explanation in Vietnamese of why it was approved or rejected>"
}}
"""
    try:
        from google import genai
        from google.genai import types
        
        # Dùng model nhẹ cho tác vụ Judge
        raw_text = _get_gemini_response(prompt, api_keys, model_name=CauHinh.SUPERVISOR_MODEL_LITE)
        if not raw_text: return {"is_approved": False, "confidence_score": 0, "reason": "Lỗi API rỗng."}
        
        parsed = json.loads(_tach_json(raw_text))
        return {
            "is_approved": bool(parsed.get("is_approved", False)),
            "confidence_score": int(parsed.get("confidence_score", 0)),
            "reason": str(parsed.get("reason", "Không có lý do."))
        }
    except Exception as e:
        logger.error(f"[Agent Critic] Error: {e}")
        return {"is_approved": False, "confidence_score": 0, "reason": f"Lỗi hệ thống Critic."}

def gemini_reviewer_agent(topic: str, section_title: str, draft_content: str, required_citations: list, api_keys: list = None) -> dict:
    """
    AGENT 2: THE REVIEWER (Kiểm toán viên LLM)
    Đọc bản nháp của Writer (OpenAI) và chấm điểm định dạng, số lượng trích dẫn, chất lượng.
    """
    if not api_keys: api_keys = CauHinh.GEMINI_API_KEYS
    
    prompt = f"""You are an ACADEMIC REVIEWER AGENT in a Multi-Agent textbook generation pipeline.
Your task is to review a section draft written by the Writer Agent.

TEXTBOOK TOPIC: {topic}
SECTION TITLE: {section_title}

DRAFT CONTENT TO REVIEW:
---
{draft_content}
---

AVAILABLE CITATION IDs (The Writer was supposed to use some of these):
{required_citations}

EVALUATION RUBRIC:
1. Format: Is it formatted in clean Markdown?
2. Citations: Does it contain inline citations in the format [id] (e.g. [1], [a2])? Are there at least a few citations?
3. Quality: Is the text informative, well-written, and free of obvious hallucinations?

If the draft is excellent, return status "PASS".
If the draft has missing citations, broken formatting, or is too short/poor quality, return status "NEEDS_REVISION" and provide clear, actionable feedback for the Editor Agent.

YOUR RESPONSE MUST BE PURE JSON:
{{
  "status": "PASS" or "NEEDS_REVISION",
  "feedback": "<Clear instructions for the Editor Agent on what to fix. Leave empty if PASS. Write in Vietnamese.>",
  "missing_citations": true or false
}}
"""
    try:
        from google import genai
        
        raw_text = _get_gemini_response(prompt, api_keys, model_name=CauHinh.SUPERVISOR_MODEL_LITE)
        if not raw_text: return {"status": "NEEDS_REVISION", "feedback": "API Reviewer trả về rỗng.", "missing_citations": False}
        
        parsed = json.loads(_tach_json(raw_text))
        status = parsed.get("status", "NEEDS_REVISION")
        if status not in ["PASS", "NEEDS_REVISION"]: status = "NEEDS_REVISION"
        
        return {
            "status": status,
            "feedback": str(parsed.get("feedback", "Không rõ lỗi.")),
            "missing_citations": bool(parsed.get("missing_citations", False))
        }
    except Exception as e:
        logger.error(f"[Agent Reviewer] Error: {e}")
        return {"status": "NEEDS_REVISION", "feedback": f"Lỗi hệ thống Reviewer: {str(e)}", "missing_citations": False}
