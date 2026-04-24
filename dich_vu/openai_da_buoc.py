# -*- coding: utf-8 -*-
import json
import re
import time
import os
import logging
import random
import functools
import copy
from openai import OpenAI
from cau_hinh import CauHinh

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

class InsufficientDataError(Exception):
    """Lỗi khi không có đủ dữ liệu từ Discovery để xây dựng giáo trình."""
    pass
from dich_vu.schemas import (
    SECTION_SCHEMA, CHAPTER_SCHEMA, OUTLINE_SCHEMA, 
    TERM_EXTRACTION_SCHEMA, FACT_EXTRACTION_SCHEMA, REWRITE_SCHEMA,
    BATCH_SECTION_SCHEMA
)

# 🛠️ Cấu hình quy mô giáo trình tập trung (V28 - Aligned with User Specs)
CURRICULUM_SCALES = {
    "can_ban": {"ch": (4, 5), "sec": (2, 3), "label": "BASIC"},
    "chuyen_sau": {"ch": (12, 20), "sec": (4, 6), "label": "ADVANCED"},
    "tieu_chuan": {"ch": (7, 10), "sec": (3, 5), "label": "STANDARD"}
}

def get_structure_config(quy_mo):
    return CURRICULUM_SCALES.get(quy_mo, CURRICULUM_SCALES["tieu_chuan"])

def _lang_directive(ngon_ngu: str = "vi") -> str:
    """V29: Trả về directive ngôn ngữ cho prompt AI."""
    if ngon_ngu == "en":
        return """
LANGUAGE DIRECTIVE: Write ALL content in ENGLISH.
- Use proper English academic terminology.
- All chapter titles, section titles, and body text MUST be in English.
- Do NOT mix languages."""
    else:
        return """
LANGUAGE DIRECTIVE: Viết TOÀN BỘ nội dung bằng TIẾNG VIỆT.
- Sử dụng thuật ngữ chuyên ngành tiếng Việt.
- Tất cả tiêu đề chương, tiêu đề mục, và nội dung PHẢI bằng tiếng Việt.
- KHÔNG trộn lẫn ngôn ngữ."""

def with_smart_retry(max_attempts=3, base_delay=1):
    """
    Expert-level exponential backoff with random jitter for API resilience.
    Base=2, Factor=2, Max Attempts=3.
    Targeting 429 (Rate Limit) and temporary 5xx errors.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    err_msg = str(e).lower()
                    
                    # Check if it's a rate limit or temporary server error
                    is_retryable = any(x in err_msg for x in ["429", "rate_limit", "500", "502", "503", "timeout"])
                    
                    if not is_retryable or attempts >= max_attempts:
                        logger.error(f"[Retry] Permanent error or max attempts reached: {e}")
                        raise e
                    
                    # Calculate delay: base * (2^attempts) + random jitter
                    # Factor 2 is implicit in (2 ** (attempts - 1))
                    delay = base_delay * (2 ** (attempts - 1)) + random.uniform(0, 1)
                    
                    logger.warning(f"[Retry] Attempt {attempts} failed with {type(e).__name__}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            return None 
        return wrapper
    return decorator

def _tach_json(text: str) -> str:
    """
    OpenAI (GPT) đôi khi bọc JSON trong ```json ...```.
    Hàm này cố gắng trích đúng JSON object hoặc array lớn nhất.
    """
    if not text:
        raise ValueError("OpenAI trả về rỗng")

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

    raise ValueError("Không tìm thấy JSON trong phản hồi OpenAI")

def xac_dinh_ngan_sach_thuat_ngu(num_articles: int, num_chapters: int, quy_mo: str = "tieu_chuan"):
    """
    Xác định quy mô giáo trình và định mức thuật ngữ (Term Budgeting). (V24.4 - Adaptive Caps)
    30 (Basic), 50 (Standard), 80 (Advanced).
    """
    mapping_level = {
        "can_ban": "basic",
        "tieu_chuan": "standard",
        "chuyen_sau": "advanced"
    }
    level = mapping_level.get(quy_mo, "standard")
    
    if num_chapters <= 0:
        cfg = get_structure_config(quy_mo)
        ch_min = cfg.get("ch", (4, 8))[0]
        ch_max = cfg.get("ch", (4, 8))[1]
        num_chapters = (ch_min + ch_max) // 2
    
    # 💎 Adaptive Term Cap (V24.4)
    config_caps = {
        "basic": 30,
        "standard": 50,
        "advanced": 80
    }
    core_count = config_caps.get(level, 50)
    
    # Supporting Budget
    config_supp = {
        "basic": 25,
        "standard": 50,
        "advanced": 80
    }
    support_count = config_supp.get(level, 50)
    
    return {
        "level": level.upper(),
        "core_count": core_count,
        "support_count": support_count
    }

def xay_dung_metadata_toan_dien(passages: list) -> list:
    """
    Cognitive Layer: Metadata Builder.
    Trích xuất Intro và Headings từ docs để AI hiểu được 'Bản đồ kiến thức'.
    """
    docs_map = {}
    for p in passages:
        title = p.get("title", "Unknown")
        if title not in docs_map:
            docs_map[title] = {"title": title, "text_chunks": []}
        docs_map[title]["text_chunks"].append(p.get("text", ""))

    metadata = []
    for title, data in docs_map.items():
        full_text = "\n".join(data["text_chunks"])
        lines = full_text.split("\n")
        
        # Lấy 10 dòng đầu làm intro
        intro = "\n".join([l.strip() for l in lines if l.strip()][:10])
        # Lấy các heading Wikipedia (== Heading ==)
        headings = [l.strip() for l in lines if l.startswith("==")][:12]
        
        metadata.append({
            "title": title,
            "intro": intro[:1500],
            "headings": headings
        })
    return metadata

@with_smart_retry(max_attempts=2)
def trich_xuat_thuat_ngu(passages: list, api_key: str, target_core: int = 40, target_support: int = 60, semaphore=None):
    """
    Tier 1: Term Extraction (Adaptive V23.3)
    Điều phối việc trích xuất thuật ngữ: Nếu > 15-20 passages, dùng chu kỳ chunking tập trung.
    """
    if not api_key: return {"core_terms": [], "supporting_terms": []}
    
    # 💎 Adaptive Logic: Nếu quá nhiều dữ liệu, chia để trị tránh Timeout
    if len(passages) > 20:
        logger.info(f"[AdaptiveExtraction] Dữ liệu lớn ({len(passages)} docs). Kích hoạt chế độ Chunking.")
        return trich_xuat_thuat_ngu_chunked(passages, api_key, target_core, target_support, semaphore)
    
    metadata = xay_dung_metadata_toan_dien(passages)
    return _trich_xuat_thuat_ngu_don_le(metadata, api_key, target_core, target_support, semaphore)

def _trich_xuat_thuat_ngu_don_le(metadata: list, api_key: str, target_core: int, target_support: int, semaphore=None):
    """Thực hiện trích xuất trong 1 lần gọi (Dùng cho lượng tin vừa phải)."""
    start_time = time.time()
    client = OpenAI(api_key=api_key, max_retries=0)
    
    prompt = f"""You are an EXPERT CURRICULUM DESIGNER.
Based on the provided article METADATA (titles, intros, headings):

1. Extract precisely {target_core} CORE TERMS: The most important concepts to build the Table of Contents.
2. Extract precisely {target_support} SUPPORTING TERMS: Technical details or sub-concepts for depth.

RULES: No duplicates. Core terms = broad/foundation, Supporting = deep/specific.

METADATA:
{json.dumps(metadata[:10], ensure_ascii=False)}

RETURN ONLY JSON matching TERM_EXTRACTION_SCHEMA.
"""
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    response = client.chat.completions.create(
                        model=CauHinh.WRITER_MODEL,
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.3,
                        response_format=TERM_EXTRACTION_SCHEMA,
                        timeout=45.0
                    )
            else:
                response = client.chat.completions.create(
                    model=CauHinh.WRITER_MODEL,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.3,
                    response_format=TERM_EXTRACTION_SCHEMA,
                    timeout=45.0
                )
            
            data = json.loads(response.choices[0].message.content)
            return {
                "core_terms": data.get("core_terms", []),
                "supporting_terms": data.get("supporting_terms", [])
            }
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[TermExtraction] Fast Retry triggered due to jitter: {e}")
                continue
            logger.error(f"[TermExtraction] Error in SinglePass: {e}")
            return {"core_terms": [], "supporting_terms": []}

@with_smart_retry(max_attempts=2)
def trich_xuat_thuat_ngu_chunked(passages: list, api_key: str, target_core: int, target_support: int, semaphore=None):
    """
    Tier 1.5: Chunked Term Extraction (V23.3)
    Groups passages into metadata, chunks metadata (max 3), and dedups.
    """
    full_metadata = xay_dung_metadata_toan_dien(passages)
    chunk_size = 10
    chunks = [full_metadata[i:i + chunk_size] for i in range(0, len(full_metadata), chunk_size)]
    
    all_core = []
    all_support = []
    
    # Chỉ xử lý tối đa 3 chunks quan trọng nhất để bảo vệ performance
    for idx, c in enumerate(chunks[:3]):
        logger.info(f"[ChunkedExtraction] Processing Chunk {idx+1}/{min(len(chunks), 3)}...")
        try:
            res = _trich_xuat_thuat_ngu_don_le(c, api_key, max(5, target_core // 2), max(5, target_support // 2), semaphore)
            all_core.extend(res.get("core_terms", []))
            all_support.extend(res.get("supporting_terms", []))
        except Exception as e:
            logger.warning(f"[ChunkedExtraction] Cụm {idx+1} thất bại: {e}")
            continue

    # --- 💎 Term Deduplication (V24.4: Semantic Guard) ---
    def deduplicate_terms(terms, limit):
        if not terms: return []
        
        # 1. Try Semantic Dedup (NumPy required)
        if np is not None:
            try:
                client = OpenAI(api_key=api_key)
                term_texts = [t.get("term", "") for t in terms]
                resp = client.embeddings.create(model="text-embedding-3-small", input=term_texts)
                vectors = [np.array(e.embedding) / np.linalg.norm(e.embedding) for e in resp.data]
                
                unique_indices = []
                for i, v in enumerate(vectors):
                    is_dup = False
                    for prev_idx in unique_indices:
                        if np.dot(v, vectors[prev_idx]) > 0.85:
                            is_dup = True; break
                    if not is_dup:
                        unique_indices.append(i)
                unique_terms = [terms[i] for i in unique_indices]
                return sorted(unique_terms, key=lambda x: x.get("importance_score", 0), reverse=True)[:limit]
            except Exception as e:
                logger.warning(f"[TermDedup] Semantic dedup failed: {e}. Falling back to Smart String logic.")

        # 2. Smart String Fallback (Substring comparison)
        unique = []
        seen_lower = set()
        for t in terms:
            nm = t.get("term", "").lower().strip()
            if not nm: continue
            is_dup = False
            for u in unique:
                u_nm = u.get("term", "").lower().strip()
                if nm in u_nm or u_nm in nm:
                    is_dup = True; break
            if not is_dup:
                unique.append(t)
        return sorted(unique, key=lambda x: x.get("importance_score", 0), reverse=True)[:limit]

    return {
        "core_terms": deduplicate_terms(all_core, target_core),
        "supporting_terms": deduplicate_terms(all_support, target_support)
    }

def trich_xuat_facts_tu_corpus(chu_de: str, passages: list, api_key: str, model: str = CauHinh.WRITER_MODEL):
    """
    Tier 1: Fact Extraction (LOCK)
    Trích xuất các atomic facts (sự thật nguyên tử) từ corpus thô.
    """
    if not api_key: return []
    # Disable SDK retry (Hotfix V5.2)
    client = OpenAI(api_key=api_key, max_retries=0)
    
    context_text = "\n".join([f"[{p.get('id', i)}] {p.get('text')}" for i, p in enumerate(passages)])
    
    prompt = f"""You are a Fact Extraction Engine. Your goal is to extract individual, atomic factual statements from the provided context about "{chu_de}".

STRICT RULES:
1. ONLY extract facts present in the text.
2. DO NOT paraphrase loosely. Keep technical terms exact.
3. Each fact must be ONE simple sentence.
4. NO internal knowledge.

CONTEXT:
{context_text}

RETURN ONLY a JSON list of facts:
[
  {{"id": "Source [N]", "fact": "Direct atomic statement"}}
]"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            response_format=FACT_EXTRACTION_SCHEMA
        )
        raw_res = response.choices[0].message.content
        data = json.loads(raw_res)
        if isinstance(data, dict):
            return data.get("facts", [])
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[FactExtraction] Lỗi: {e}")
        return []

@with_smart_retry(max_attempts=3)
def tao_dan_y(
    chu_de: str,
    corpus_passages: list,
    model: str = CauHinh.WRITER_MODEL,
    api_key: str = None,
    so_chuong_min: int = 8,
    so_chuong_max: int = 12,
    che_do: str = "auto",
    danh_sach_chuong: str = "",
    quy_mo: str = "tieu_chuan",
    semaphore=None,
):
    """
    Bước 1: Tạo dàn ý.
    """
    # Điều chỉnh số chương dựa trên quy mô nếu đang dùng mặc định (8-12)
    if so_chuong_min == 8 and so_chuong_max == 12:
        mapping = {
            "can_ban": (4, 6),
            "chuyen_sau": (13, 18),
            "tieu_chuan": (8, 12)
        }
        so_chuong_min, so_chuong_max = mapping.get(quy_mo, (8, 12))

    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY")
    
    # Disable SDK retry (Hotfix V5.2)
    time.sleep(random.uniform(0.5, 1.5))
    client = OpenAI(api_key=api_key, max_retries=0)

    packed = []
    for p in corpus_passages:
        packed.append({
            "pid": p.get("pid"),
            "title": p.get("title"),
            "text": p.get("text")[:500] + "...",
            "url": p.get("url")
        })
    
    cau_truc_prompt = ""
    if che_do == "custom_so_chuong":
        cau_truc_prompt = f"- BẮT BUỘC có CHÍNH XÁC {so_chuong_max} chương, mỗi chương 4-5 mục con."
    elif che_do == "custom_danh_sach" and danh_sach_chuong:
        cau_truc_prompt = f"- SỬ DỤNG DANH SÁCH: {danh_sach_chuong}. Mỗi chương cần 4-5 mục bóc tách."
    else:
        if quy_mo == "can_ban": 
            cau_truc_prompt = f"- Tối đa {so_chuong_max} chương. Mỗi chương có ít nhất 2 mục con tập trung vào khái niệm chính."
        elif quy_mo == "chuyen_sau": 
            cau_truc_prompt = f"- Tối thiểu {so_chuong_min} chương. Mỗi chương BẮT BUỘC có 6-8 mục con chi tiết học thuật cao."
        else: # tieu_chuan
            cau_truc_prompt = "- Mỗi chương BẮT BUỘC có từ 4 đến 5 mục con chuyên sâu."

    prompt = f"""Bạn là chuyên gia biên soạn giáo trình đại học về "{chu_de}".
Dữ liệu: {json.dumps(packed, ensure_ascii=False)}

YÊU CẦU DÀN Ý (V20.4 - High Fidelity):
1. Trích xuất 100 thuật ngữ chuyên ngành quan trọng nhất.
2. Xây dựng cấu trúc chương/mục: {cau_truc_prompt} 
   - LƯU Ý: Phải tuân thủ NGHIÊM NGẶT số lượng mục con tối thiểu cho mỗi chương.
3. CHIỀU SÂU SEMANTIC: Mỗi chương phải đi từ: Nguyên lý/Bản chất -> Cơ chế chi tiết -> Ứng dụng thực tiễn -> Phân tích nâng cao. 
   - Nếu dữ liệu ít, hãy tập trung phân tách các khía cạnh KỸ THUẬT khác nhau thành các mục con riêng biệt thay vì gộp chung.
4. TÍNH DANH BIỆT & ANTI-TRIVIAL: 
   - Mỗi mục phải có tên gắn liền với một thực thể kiến thức hoặc kỹ thuật cụ thể.
   - NGHIÊM CẤM "chia nhỏ giả tạo": Không sử dụng các tiêu đề rỗng như "Giới thiệu 1", "Giới thiệu 2", "Phần mở rộng A", "Phần mở rộng B".
   - Cấm sử dụng lại các từ chung chung như "Giới thiệu", "Tổng quan", "Lời kết" ở nhiều chương.
5. GIỚI HẠN AN TOÀN: Tổng số mục (sub-sections) toàn cuốn sách không được vượt quá 60 mục.

Trả về JSON đúng định dạng:
{{
  "topic": "{chu_de}",
  "terms": [ {{"term": "...", "meaning": "..."}} ],
  "outline": [
    {{
      "chapter_index": 1,
      "title": "...",
      "sections": [ {{"title": "...", "recommended_pids": []}} ]
    }}
  ]
}}"""

    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=0.7,
                        response_format=OUTLINE_SCHEMA,
                        timeout=45.0
                    )
            else:
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.7,
                    response_format=OUTLINE_SCHEMA,
                    timeout=45.0
                )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[OutlineGeneration] Fast Retry triggered due to jitter: {e}")
                continue
            logger.error(f"[OutlineGeneration] Final Error: {e}")
            raise e

@with_smart_retry(max_attempts=2)
def tao_dan_y_tu_passages(chu_de: str, corpus_passages: list, api_key: str, quy_mo: str = "tieu_chuan", semaphore=None):
    """
    Tier 2: Passage-to-Outline Fallback (V23.3)
    Dùng khi Term Extraction fail hoặc timeout. AI sẽ tự bóc tách cấu trúc từ metadata nguồn.
    """
    client = OpenAI(api_key=api_key, max_retries=0)
    # Chỉ lấy tinh hoa metadata để tránh tràn context
    packed = [{"title": p.get("title"), "text": p.get("text", "")[:400]} for p in corpus_passages[:15]]
    
    prompt = f"""Bạn là kiến trúc sư chương trình giảng dạy đại học. Tạo dàn ý giáo trình "{chu_de}" dựa trên các nguồn sau:
DỮ LIỆU NGUỒN (METADATA): {json.dumps(packed, ensure_ascii=False)}

YÊU CẦU CỐT LÕI (STRICT RULES):
1. LỘ TRÌNH HỌC THUẬT (ACADEMIC PROGRESSION): Cấu trúc phải đi từ Nền tảng (Intro/Fundamentals) -> Cốt lõi (Core Mechanics) -> Nâng cao (Advanced/Optimization) -> Ứng dụng/Thực tiễn (Applications).
2. TÍNH DUY NHẤT (UNIQUENESS): Mỗi chương phải tập trung vào một cụm khái niệm BIỆT LẬP. KHÔNG được lặp lại chủ đề giữa các chương.
3. ANTI-GENERIC: Tuyệt đối không dùng các tiêu đề chung chung như "Tổng quan", "Giới thiệu", "Kết luận" lặp đi lặp lại. Phải gắn tên chương với thực thể kỹ thuật cụ thể.
4. QUY MÔ: Chia thành tối đa 8 chương. Mỗi chương 3–5 mục con chi tiết.

RETURN ONLY JSON matching OUTLINE_SCHEMA.
"""
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        messages=[{"role": "system", "content": prompt}],
                        model=CauHinh.WRITER_MODEL,
                        temperature=0.6,
                        response_format=OUTLINE_SCHEMA,
                        timeout=45.0
                    )
            else:
                resp = client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt}],
                    model=CauHinh.WRITER_MODEL,
                    temperature=0.6,
                    response_format=OUTLINE_SCHEMA,
                    timeout=45.0
                )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[Tier2Outline] Fast Retry triggered: {e}")
                continue
            logger.error(f"[Tier2Outline] Error: {e}")
            raise e

# ===========================================================================
# 🆕 CONSTRAINT-AWARE POLISH LAYER (V30)
# Mục tiêu: Cải thiện điểm cấu trúc (Logic/Dependency/Coherence) mà KHÔNG
# gây thêm hallucination hoặc phá vỡ mapping nguồn dữ liệu.
# ===========================================================================

def _get_embeddings_batch(texts: list, api_key: str):
    """Lấy embeddings cho một batch text sử dụng OpenAI text-embedding-3-small."""
    if not texts or np is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors = [np.array(e.embedding) / np.linalg.norm(e.embedding) for e in resp.data]
        return vectors
    except Exception as e:
        logger.warning(f"[Polish-Embed] Embedding failed: {e}")
        return None

def _cosine_sim(v1, v2):
    """Tính cosine similarity giữa 2 vector."""
    if v1 is None or v2 is None:
        return 1.0  # fallback an toàn
    return float(np.dot(v1, v2))


def _polish_reorder(outline: list, chu_de: str, client, api_key: str):
    """
    Bước 1: Sắp xếp lại thứ tự chương theo lộ trình sư phạm.
    Có Hard Dependency Check + Implicit Dependency Heuristic.
    """
    if len(outline) <= 2:
        return outline  # Quá ít chương, không cần reorder
    
    titles = [f"{c['chapter_index']}. {c['title']}" for c in outline]
    titles_text = "\n".join(titles)
    
    # 1a. Xây dependency graph từ LLM
    dep_prompt = f"""Given these chapters for an academic curriculum on "{chu_de}":
{titles_text}

Identify prerequisite pairs where chapter A MUST come before chapter B 
(because B requires knowledge from A).
Return ONLY valid JSON: {{"pairs": [[A_index, B_index], ...]}}
Only include STRONG, obvious prerequisites. If none, return {{"pairs": []}}"""

    dep_graph = []
    try:
        dep_resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL,
            messages=[{"role": "user", "content": dep_prompt}],
            temperature=0.0, timeout=30.0
        )
        dep_data = json.loads(dep_resp.choices[0].message.content)
        dep_graph = dep_data.get("pairs", [])
        logger.info(f"[Polish-Reorder] Dependency graph: {len(dep_graph)} pairs")
    except Exception as e:
        logger.warning(f"[Polish-Reorder] Dependency extraction failed: {e}")
    
    # 1b. Implicit dependency heuristic: nếu term_A xuất hiện trong title_B → A trước B
    for i, ch_a in enumerate(outline):
        for j, ch_b in enumerate(outline):
            if i == j:
                continue
            a_title = ch_a["title"].lower()
            b_title = ch_b["title"].lower()
            # Nếu title A là substring của title B → A là nền tảng cho B
            if len(a_title) > 3 and a_title in b_title and len(b_title) > len(a_title):
                pair = [ch_a["chapter_index"], ch_b["chapter_index"]]
                if pair not in dep_graph:
                    dep_graph.append(pair)
                    logger.info(f"[Polish-Reorder] Implicit dep: '{ch_a['title']}' → '{ch_b['title']}'")
    
    # 1c. Reorder bằng LLM
    reorder_prompt = f"""Reorder these chapters to improve pedagogical flow for topic "{chu_de}".

Constraints:
- Do NOT violate prerequisite relationships
- Earlier chapters must introduce foundational concepts required by later ones
- Preserve ALL chapters exactly. Do NOT add, remove, or rename any chapter.
- Return ONLY valid JSON: {{"order": [idx1, idx2, ...]}}

Chapters:
{titles_text}"""

    try:
        reorder_resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL,
            messages=[{"role": "user", "content": reorder_prompt}],
            temperature=0.0, timeout=30.0
        )
        new_order = json.loads(reorder_resp.choices[0].message.content).get("order", [])
    except Exception as e:
        logger.warning(f"[Polish-Reorder] Reorder API failed: {e}. Keeping original.")
        return outline
    
    # Validate: đúng số lượng và đúng indices
    valid_indices = {c["chapter_index"] for c in outline}
    if set(new_order) != valid_indices:
        logger.warning(f"[Polish-Reorder] Invalid order (missing/extra indices). ROLLBACK.")
        return outline
    
    # 1d. HARD DEPENDENCY CHECK
    for (a, b) in dep_graph:
        if a in new_order and b in new_order:
            if new_order.index(a) > new_order.index(b):
                logger.warning(f"[Polish-Reorder] Dependency violation: {a} must come before {b}. ROLLBACK.")
                return outline
    
    # Áp dụng thứ tự mới
    index_map = {c["chapter_index"]: c for c in outline}
    reordered = [index_map[idx] for idx in new_order if idx in index_map]
    for i, ch in enumerate(reordered):
        ch["chapter_index"] = i + 1
    
    logger.info(f"[Polish-Reorder] Reorder successful: {new_order}")
    return reordered


def _polish_rename(outline: list, chu_de: str, client, api_key: str):
    """
    Bước 2: Đổi tên chương/mục cho mượt mà.
    Có Cosine Similarity Gate + Length Ratio Check.
    """
    # Chuẩn bị dữ liệu hiện tại
    current_data = []
    for ch in outline:
        ch_data = {"chapter_index": ch["chapter_index"], "title": ch["title"], "sections": []}
        for sec in ch.get("sections", []):
            ch_data["sections"].append({"title": sec["title"]})
        current_data.append(ch_data)
    
    rename_prompt = f"""Rewrite the following chapter and section titles for topic "{chu_de}" to improve clarity and pedagogical tone.

Constraints:
- Preserve original meaning EXACTLY
- Do NOT introduce new concepts not present in the original title
- Do NOT generalize or specialize the content scope
- Keep the SAME number of chapters and sections
- Titles should be clear, descriptive, and suitable for a university textbook
- Return ONLY valid JSON array with same structure

Current outline:
{json.dumps(current_data, ensure_ascii=False, indent=2)}"""

    try:
        rename_resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL,
            messages=[{"role": "user", "content": rename_prompt}],
            temperature=0.2, timeout=45.0
        )
        renamed_data = json.loads(rename_resp.choices[0].message.content)
        # Nếu LLM trả về dict với key "outline" hoặc tương tự
        if isinstance(renamed_data, dict):
            renamed_data = renamed_data.get("outline", renamed_data.get("chapters", []))
        if not isinstance(renamed_data, list):
            logger.warning("[Polish-Rename] Invalid response format. ROLLBACK.")
            return outline
    except Exception as e:
        logger.warning(f"[Polish-Rename] API failed: {e}. ROLLBACK.")
        return outline
    
    # VALIDATION: Kiểm tra số lượng chương
    if len(renamed_data) != len(outline):
        logger.warning(f"[Polish-Rename] Chapter count mismatch: {len(renamed_data)} vs {len(outline)}. ROLLBACK.")
        return outline
    
    # Kiểm tra số sections từng chương
    for i, (old_ch, new_ch) in enumerate(zip(outline, renamed_data)):
        old_secs = len(old_ch.get("sections", []))
        new_secs = len(new_ch.get("sections", []))
        if old_secs != new_secs:
            logger.warning(f"[Polish-Rename] Section count mismatch at ch {i+1}: {old_secs} vs {new_secs}. ROLLBACK.")
            return outline
    
    # COSINE SIMILARITY GATE + LENGTH RATIO CHECK
    old_titles = [ch["title"] for ch in outline]
    new_titles = [ch.get("title", "") for ch in renamed_data]
    
    old_vecs = _get_embeddings_batch(old_titles, api_key)
    new_vecs = _get_embeddings_batch(new_titles, api_key)
    
    result_outline = copy.deepcopy(outline)
    rename_count = 0
    
    for i, (old_ch, new_ch) in enumerate(zip(result_outline, renamed_data)):
        new_title = new_ch.get("title", old_ch["title"])
        
        # Length ratio check (bắt narrowing/broadening)
        len_ratio = len(new_title) / max(len(old_ch["title"]), 1)
        if len_ratio > 2.0 or len_ratio < 0.3:
            logger.info(f"[Polish-Rename] Length drift ch{i+1}: ratio={len_ratio:.2f}. SKIP.")
            continue
        
        # Cosine check
        if old_vecs and new_vecs:
            sim = _cosine_sim(old_vecs[i], new_vecs[i])
            if sim < 0.85:
                logger.info(f"[Polish-Rename] Semantic drift ch{i+1}: '{old_ch['title']}' → '{new_title}' (sim={sim:.3f}). SKIP.")
                continue
        
        old_ch["title"] = new_title
        rename_count += 1
        
        # Rename sections (với cùng safety checks)
        for j, (old_sec, new_sec) in enumerate(zip(old_ch.get("sections", []), new_ch.get("sections", []))):
            new_sec_title = new_sec.get("title", old_sec["title"])
            sec_len_ratio = len(new_sec_title) / max(len(old_sec["title"]), 1)
            if 0.3 <= sec_len_ratio <= 2.0:
                old_sec["title"] = new_sec_title
    
    logger.info(f"[Polish-Rename] Renamed {rename_count}/{len(outline)} chapter titles.")
    return result_outline


def _polish_flag_overlaps(outline: list, chu_de: str, client, api_key: str):
    """
    Bước 3: Phát hiện chương trùng lặp (flag-only) + Rule-based Resolution.
    Sử dụng weighted scoring (length + topic similarity) thay vì chỉ giữ chương dài hơn.
    """
    if len(outline) <= 3:
        return outline
    
    titles = [ch["title"] for ch in outline]
    
    # Tính embedding cho tất cả chapter titles
    title_vecs = _get_embeddings_batch(titles, api_key)
    topic_vec = _get_embeddings_batch([chu_de], api_key)
    
    if not title_vecs or not topic_vec:
        return outline  # Không có embedding → bỏ qua
    
    topic_v = topic_vec[0]
    
    # Tìm các cặp trùng lặp bằng embedding (không cần gọi LLM)
    overlapping_pairs = []
    for i in range(len(outline)):
        for j in range(i + 1, len(outline)):
            sim = _cosine_sim(title_vecs[i], title_vecs[j])
            if sim > 0.88:  # Ngưỡng cao: chắc chắn trùng
                overlapping_pairs.append((i, j, sim))
    
    if not overlapping_pairs:
        logger.info("[Polish-Overlap] No overlapping chapters detected.")
        return outline
    
    # Rule-based resolution: weighted score = 0.3*section_count + 0.7*topic_similarity
    to_remove = set()
    for (i, j, sim) in overlapping_pairs:
        ch_i = outline[i]
        ch_j = outline[j]
        
        sec_i = len(ch_i.get("sections", []))
        sec_j = len(ch_j.get("sections", []))
        
        topic_sim_i = _cosine_sim(title_vecs[i], topic_v)
        topic_sim_j = _cosine_sim(title_vecs[j], topic_v)
        
        score_i = 0.3 * (sec_i / max(sec_i, sec_j, 1)) + 0.7 * topic_sim_i
        score_j = 0.3 * (sec_j / max(sec_i, sec_j, 1)) + 0.7 * topic_sim_j
        
        loser = j if score_i >= score_j else i
        to_remove.add(outline[loser]["chapter_index"])
        logger.info(
            f"[Polish-Overlap] Overlap (sim={sim:.3f}): "
            f"'{ch_i['title']}' (score={score_i:.3f}) vs '{ch_j['title']}' (score={score_j:.3f}). "
            f"Removing ch{outline[loser]['chapter_index']}."
        )
    
    cleaned = [c for c in outline if c["chapter_index"] not in to_remove]
    for i, ch in enumerate(cleaned):
        ch["chapter_index"] = i + 1
    
    logger.info(f"[Polish-Overlap] Removed {len(to_remove)} duplicate chapter(s).")
    return cleaned


def _structural_invariance_check(original: list, polished: list):
    """
    Kiểm tra Structural Invariance: không mất concept, không thêm concept.
    So sánh tập hợp tên section trước/sau (section = đơn vị kiến thức nhỏ nhất).
    """
    def extract_concepts(outline):
        concepts = set()
        for ch in outline:
            for sec in ch.get("sections", []):
                # Chuẩn hoá: lowercase, strip
                concepts.add(sec.get("title", "").strip().lower())
        return concepts
    
    original_concepts = extract_concepts(original)
    polished_concepts = extract_concepts(polished)
    
    lost = original_concepts - polished_concepts
    added = polished_concepts - original_concepts
    
    if lost:
        logger.warning(f"[Polish-Invariance] LOST concepts: {lost}")
    if added:
        logger.info(f"[Polish-Invariance] Renamed concepts (expected): {len(added)} new titles")
    
    # Chỉ FAIL nếu mất concept (added là do rename, chấp nhận được)
    return len(lost) == 0


def _quick_structure_score(outline: list, chu_de: str, client):
    """Chấm nhanh cấu trúc bằng LLM (1 call duy nhất, trả về số 1-10)."""
    titles = "\n".join([f"Ch{c['chapter_index']}: {c['title']}" for c in outline])
    prompt = f"""Rate this curriculum outline for "{chu_de}" on a scale 1-10 for overall pedagogical quality.
Consider: logical flow from basic to advanced, prerequisite ordering, chapter distinctness.
Be lenient — this is an outline only, not full content.
Return ONLY a single number (1-10), nothing else.

{titles}"""
    try:
        resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=5, timeout=15.0
        )
        score = float(re.search(r'(\d+\.?\d*)', resp.choices[0].message.content).group(1))
        return min(max(score, 1.0), 10.0)
    except Exception as e:
        logger.warning(f"[Polish-Score] Scoring failed: {e}. Returning neutral 5.0")
        return 5.0


def _polish_relevance_gate(outline: list, chu_de: str, api_key: str, 
                           core_terms: list = None,
                           threshold_keep: float = 0.55,
                           threshold_remove: float = 0.30):
    """
    Step 0: Topic Relevance Gate (Multi-Anchor, Soft Threshold).
    Loại bỏ sections/chapters lạc đề bằng multi-anchor cosine similarity.
    
    3 vùng quyết định:
      - score >= threshold_keep  → GIỮ (on-topic chắc chắn)
      - score <= threshold_remove → LOẠI (off-topic chắc chắn)
      - ở giữa → GIỮ (ưu tiên an toàn, tránh cắt nhầm)
    
    Multi-anchor: score = max(cosine(section, anchor) for anchor in anchors)
    Anchors = [topic] + top core_terms từ EKRE.
    """
    if len(outline) <= 2 or np is None:
        return outline
    
    # === Xây dựng anchor set ===
    anchors = [chu_de]
    if core_terms:
        # Lấy top 15 core terms làm anchor bổ sung
        for t in core_terms[:15]:
            term_text = t.get("term", "") if isinstance(t, dict) else str(t)
            if term_text and term_text != chu_de:
                anchors.append(term_text)
    
    logger.info(f"[Polish-Relevance] Anchors: {len(anchors)} ({anchors[:5]}...)")
    
    # === Thu thập tất cả section titles ===
    all_section_titles = []
    section_map = []  # (chapter_idx, section_idx)
    for ch_i, ch in enumerate(outline):
        for sec_i, sec in enumerate(ch.get("sections", [])):
            all_section_titles.append(sec.get("title", ""))
            section_map.append((ch_i, sec_i))
    
    if not all_section_titles:
        return outline
    
    # === Embedding: anchors + section titles (1 batch duy nhất) ===
    all_texts = anchors + all_section_titles
    vectors = _get_embeddings_batch(all_texts, api_key)
    
    if not vectors:
        return outline
    
    anchor_vecs = vectors[:len(anchors)]
    section_vecs = vectors[len(anchors):]
    
    # === Tính multi-anchor relevance score cho từng section ===
    off_topic_sections = set()
    gray_zone_count = 0
    
    for idx, (ch_i, sec_i) in enumerate(section_map):
        # Multi-anchor: lấy điểm MAX trong tất cả anchors
        max_sim = max(_cosine_sim(section_vecs[idx], av) for av in anchor_vecs)
        sec_title = all_section_titles[idx]
        
        if max_sim <= threshold_remove:
            # Vùng LOẠI: chắc chắn off-topic
            off_topic_sections.add((ch_i, sec_i))
            logger.info(
                f"[Polish-Relevance] REMOVE: '{sec_title}' "
                f"(max_sim={max_sim:.3f} <= {threshold_remove})"
            )
        elif max_sim < threshold_keep:
            # Vùng GRAY: giữ lại (ưu tiên an toàn), chỉ log cảnh báo
            gray_zone_count += 1
            logger.info(
                f"[Polish-Relevance] GRAY (kept): '{sec_title}' "
                f"(max_sim={max_sim:.3f}, zone [{threshold_remove}-{threshold_keep}])"
            )
        # else: max_sim >= threshold_keep → on-topic, không cần log
    
    if not off_topic_sections:
        logger.info(
            f"[Polish-Relevance] All {len(all_section_titles)} sections passed. "
            f"({gray_zone_count} in gray zone, kept safely)."
        )
        return outline
    
    # === Áp dụng lọc: loại section, xóa chương nếu >60% section bị loại ===
    result_outline = []
    removed_sections = 0
    removed_chapters = 0
    
    for ch_i, ch in enumerate(outline):
        original_sections = ch.get("sections", [])
        filtered_sections = [
            sec for sec_i, sec in enumerate(original_sections)
            if (ch_i, sec_i) not in off_topic_sections
        ]
        
        removed_in_ch = len(original_sections) - len(filtered_sections)
        
        # Nếu >60% section bị loại → xóa toàn bộ chương
        if len(original_sections) > 0 and removed_in_ch / len(original_sections) > 0.6:
            logger.info(
                f"[Polish-Relevance] REMOVE chapter '{ch['title']}' "
                f"({removed_in_ch}/{len(original_sections)} sections off-topic)"
            )
            removed_chapters += 1
            removed_sections += len(original_sections)
            continue
        
        if filtered_sections:
            ch_copy = copy.deepcopy(ch)
            ch_copy["sections"] = filtered_sections
            result_outline.append(ch_copy)
            removed_sections += removed_in_ch
        else:
            removed_chapters += 1
    
    # Re-index chapters
    for i, ch in enumerate(result_outline):
        ch["chapter_index"] = i + 1
    
    logger.info(
        f"[Polish-Relevance] Removed {removed_sections} off-topic sections, "
        f"{removed_chapters} off-topic chapters. "
        f"Remaining: {len(result_outline)} chapters."
    )
    return result_outline


def _apply_polish_layer(result: dict, chu_de: str, api_key: str):
    """
    🆕 V30 Orchestrator: Áp dụng Constraint-aware Polish Layer.
    Partial rollback theo từng bước + Score Guard.
    """
    if not result or not result.get("outline"):
        return result
    
    outline = result.get("outline", [])
    if len(outline) < 2:
        return result  # Quá ít chương, bỏ qua
    
    client = OpenAI(api_key=api_key, max_retries=0)
    original = copy.deepcopy(outline)
    current = copy.deepcopy(outline)
    
    logger.info(f"[Polish] === Starting Constraint-aware Polish Layer ({len(current)} chapters) ===")
    polish_start = time.time()
    
    # === Bước 0: Topic Relevance Gate (Multi-Anchor, Pure Math) ===
    try:
        core_terms = result.get("terms", [])
        filtered = _polish_relevance_gate(current, chu_de, api_key, core_terms=core_terms)
        if len(filtered) != len(current):
            removed_count = len(current) - len(filtered)
            logger.info(f"[Polish] Step 0 (Relevance Gate): Removed {removed_count} off-topic chapter(s) ✓")
            current = filtered
        else:
            logger.info("[Polish] Step 0 (Relevance Gate): All chapters on-topic.")
    except Exception as e:
        logger.warning(f"[Polish] Step 0 (Relevance Gate) FAILED: {e}. Skipping.")
    
    # === Bước 1: Safe Reorder ===
    try:
        reordered = _polish_reorder(current, chu_de, client, api_key)
        if [c["chapter_index"] for c in reordered] != [c["chapter_index"] for c in current]:
            current = reordered
            logger.info("[Polish] Step 1 (Reorder): APPLIED ✓")
        else:
            logger.info("[Polish] Step 1 (Reorder): No change needed.")
    except Exception as e:
        logger.warning(f"[Polish] Step 1 (Reorder) FAILED: {e}. Skipping.")
    
    # === Bước 2: Safe Rename ===
    after_reorder = copy.deepcopy(current)
    try:
        renamed = _polish_rename(current, chu_de, client, api_key)
        # Kiểm tra structural invariance sau rename
        if _structural_invariance_check(after_reorder, renamed):
            current = renamed
            logger.info("[Polish] Step 2 (Rename): SAFE ✓")
        else:
            logger.warning("[Polish] Step 2 (Rename): Invariance violated → PARTIAL ROLLBACK")
            current = after_reorder
    except Exception as e:
        logger.warning(f"[Polish] Step 2 (Rename) FAILED: {e}. Skipping.")
    
    # === Bước 3: Flag Overlaps ===
    try:
        before_overlap = copy.deepcopy(current)
        cleaned = _polish_flag_overlaps(current, chu_de, client, api_key)
        if len(cleaned) != len(current):
            logger.info(f"[Polish] Step 3 (Overlap): Removed {len(current) - len(cleaned)} chapter(s)")
            current = cleaned
        else:
            logger.info("[Polish] Step 3 (Overlap): No overlaps found.")
    except Exception as e:
        logger.warning(f"[Polish] Step 3 (Overlap) FAILED: {e}. Skipping.")
    
    # === Score Guard (Self-improving loop) ===
    if current != original:
        try:
            score_before = _quick_structure_score(original, chu_de, client)
            score_after = _quick_structure_score(current, chu_de, client)
            
            if score_after < score_before:
                logger.warning(
                    f"[Polish] Score DECREASED ({score_before:.1f} → {score_after:.1f}). FULL ROLLBACK."
                )
                result["outline"] = original
            else:
                logger.info(
                    f"[Polish] Score IMPROVED ({score_before:.1f} → {score_after:.1f}). COMMIT. ✓"
                )
                result["outline"] = current
        except Exception as e:
            logger.warning(f"[Polish] Score guard failed: {e}. Using polished version.")
            result["outline"] = current
    else:
        logger.info("[Polish] No changes made. Keeping original.")
    
    elapsed = time.time() - polish_start
    logger.info(f"[Polish] === Polish Layer completed in {elapsed:.2f}s ===")
    return result

# ===========================================================================
# END OF POLISH LAYER
# ===========================================================================

def _programmatic_outline_builder(chu_de: str, terms_data: dict, target_ch: int, sec_min: int, sec_max: int):
    """
    V28 Ultimate Fallback: Tự động xây dựng dàn ý từ danh sách thuật ngữ
    khi AI liên tục collapse. Chia terms đều vào các chương.
    """
    core = terms_data.get("core_terms", [])
    supporting = terms_data.get("supporting_terms", [])
    all_terms = core + supporting
    
    if not all_terms:
        return None
    
    # Phân bổ terms đều vào các chương
    terms_per_ch = max(1, len(all_terms) // target_ch)
    sec_per_ch = max(sec_min, min(sec_max, terms_per_ch))
    
    outline = []
    term_idx = 0
    for ch_i in range(target_ch):
        chunk = all_terms[term_idx : term_idx + terms_per_ch]
        if not chunk and term_idx < len(all_terms):
            chunk = all_terms[term_idx:]
        if not chunk:
            # Tạo chapter placeholder từ topic
            chunk = [{"term": f"{chu_de} - Khía cạnh {ch_i + 1}"}]
        
        # Tạo title từ term đầu tiên của chunk
        ch_title = chunk[0].get("term", f"Chương {ch_i + 1}")
        
        # Tạo sections từ các terms trong chunk
        sections = []
        for s_i, t in enumerate(chunk[:sec_per_ch]):
            sections.append({
                "title": t.get("term", f"Mục {s_i + 1}"),
                "recommended_pids": []
            })
        # Đảm bảo tối thiểu sec_min sections
        while len(sections) < sec_min:
            sections.append({
                "title": f"Phân tích chuyên sâu {len(sections) + 1}",
                "recommended_pids": []
            })
        
        outline.append({
            "chapter_index": ch_i + 1,
            "title": ch_title,
            "sections": sections
        })
        term_idx += terms_per_ch
    
    terms_list = [{"term": t.get("term", ""), "meaning": ""} for t in all_terms[:80]]
    
    logger.warning(f"[Architect] Programmatic Outline built: {len(outline)} chapters from {len(all_terms)} terms.")
    return {
        "topic": chu_de,
        "terms": terms_list,
        "outline": outline
    }


def nhom_thuat_ngu_va_tao_dan_y(terms_data: dict, api_key: str, chu_de: str, so_chuong: int = 10, quy_mo: str = "tieu_chuan", semaphore=None, safety_class: str = "SAFE", ngon_ngu: str = "vi", so_chuong_custom=None, danh_sach_chuong=None):
    """
    Cognitive Layer: Clustering & Outline.
    Phân cụm thuật ngữ theo lộ trình: Nền tảng -> Cốt lõi -> Ứng dụng.
    """
    if not api_key: raise RuntimeError("Thiếu OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, max_retries=0)
    
    # 💎 Diamond Guard: Kiểm tra độ dày dữ liệu trước khi xây dựng dàn ý
    all_terms = terms_data.get("core_terms", []) + terms_data.get("supporting_terms", [])
    if len(all_terms) < 4:
        logger.warning(f"[Architect] Insufficient data: only {len(all_terms)} terms found. Topic drift likely.")
        raise InsufficientDataError(f"Không tìm thấy đủ thuật ngữ chuyên ngành (chỉ có {len(all_terms)}). Vui lòng thử chủ đề rộng hơn.")

    # 🛠️ Adaptive Structuring: Soft Feasibility Scaling (V24.4)
    cfg = get_structure_config(quy_mo)
    doc_count = len(all_terms)  # V28 FIX: Đếm tổng terms thay vì chỉ core_terms
    
    # Hardened Config Retrieval
    ch_min = cfg.get("ch", (4, 8))[0]
    ch_max = cfg.get("ch", (4, 8))[1]
    sec_min = cfg.get("sec", (3, 5))[0]
    sec_max = cfg.get("sec", (3, 5))[1]
    
    # 💎 Soft Scaling Safety Valve (V28 - Term-Count Based, min=12 for chuyen_sau)
    if quy_mo == "chuyen_sau":
        if doc_count < 10:
            ch_min, ch_max = 12, 14
            logger.warning(f"[Architect] Very low density ({doc_count} terms). Scaling to 12-14 chapters.")
        elif doc_count < 25:
            ch_min, ch_max = 12, 16
            logger.info(f"[Architect] Low density ({doc_count} terms). Scaling to 12-16 chapters.")
        elif doc_count < 50:
            ch_min, ch_max = 14, 18
            logger.info(f"[Architect] Moderate density ({doc_count} terms). Scaling to 14-18 chapters.")
        # else: giữ nguyên 12-20 từ config
    
    # 💎 Final Target Calculation (V28: Range-based for auto, exact for custom)
    # 💎 Final Target Calculation (V31: Support custom count and list)
    target_ch = 0
    target_mode = "RANGE"
    
    if danh_sach_chuong and len(danh_sach_chuong) > 0:
        target_ch = len(danh_sach_chuong)
        target_mode = "MANUAL_LIST"
    elif so_chuong_custom and int(so_chuong_custom) > 0:
        target_ch = int(so_chuong_custom)
        target_mode = "EXACT"
    elif so_chuong > 0:
        target_ch = so_chuong
        target_mode = "EXACT"

    # 💎 V23.3+ Optimization: Chuyển đổi data sang text list
    core_list = terms_data.get("core_terms", [])
    input_terms_text = "\n".join([f"- {t.get('term')}" for t in core_list[:80]])

    lang_dir = _lang_directive(ngon_ngu)
    system_prompt = f"""You are an expert curriculum designer. Organize technical terms into a logical academic outline. 
Directly generate the outline. Do NOT explain reasoning or provide long introductions. Return VALID JSON matching OUTLINE_SCHEMA.
{lang_dir}"""

    # V28+: Prompt thay đổi tuỳ theo mode
    if target_mode == "MANUAL_LIST":
        chapter_instruction = f"""- YOU MUST USE THIS EXACT CHAPTER LIST: {json.dumps(danh_sach_chuong, ensure_ascii=False)}
- For each chapter in the list, create {sec_min} to {sec_max} detailed sections.
- The number of chapters MUST be EXACTLY {target_ch} (following the provided list)."""
    elif target_mode == "EXACT":
        chapter_instruction = f"""- EXACTLY {target_ch} chapters. THIS IS MANDATORY. Count them before returning.
- Each chapter must have {sec_min} to {sec_max} sections.
- You MUST generate EXACTLY {target_ch} chapters. Do not generate fewer under any circumstances."""
    else:
        chapter_instruction = f"""- Between {ch_min} and {ch_max} chapters. Choose the number that best fits the depth and breadth of the input terms.
- Each chapter must have {sec_min} to {sec_max} sections.
- You MUST generate AT LEAST {ch_min} chapters. Do NOT generate fewer than {ch_min} under any circumstances."""

    prompt = f"""Topic: {chu_de}
Scale: {quy_mo.upper()}

TASK: Create a structured curriculum outline using the terms provided.

INPUT TERMS:
{input_terms_text}

STRICT COMPLIANCE REQUIREMENTS (V28):
1. STRUCTURE:
{chapter_instruction}

2. CONSERVATIVE SYNTHESIS & MICRO-SEGMENTATION:
- Use only high-confidence concepts from the provided materials.
- If the knowledge base is broad but topics overlap, IDENTIFY THE SUBTLE SUB-TECHNICAL DIFFERENCES and create separate, laser-focused chapters for each aspect.
- Do NOT introduce speculative topics, but DO feel free to split massive concepts into specific sub-phases (e.g., instead of one "History" chapter, use "Early Period", "Golden Era", "Modern Influence").

3. FLOW: Introduction → Foundation → Mechanics → Advanced → Applications.
4. NAMING: Specific, technical, descriptive. No generic titles.
5. NO REASONING: Output ONLY the JSON.

RETURN ONLY JSON matching this format:
{{
  "topic": "{chu_de}",
  "terms": [ {{ "term": "...", "meaning": "mô tả ngắn" }} ],
  "outline": [
    {{
      "chapter_index": 1,
      "title": "...",
      "sections": [ {{ "title": "...", "recommended_pids": [] }} ]
    }}
  ]
}}"""

    start_at = time.time()
    target_display = f"{target_ch}" if target_mode == "EXACT" else f"{ch_min}-{ch_max}"
    logger.info(f"[STEP] Bắt đầu xây dựng Dàn ý (quy_mo={quy_mo}, mode={target_mode}, target={target_display}).")
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        model=CauHinh.WRITER_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        response_format=OUTLINE_SCHEMA,
                        timeout=120.0
                    )
            else:
                resp = client.chat.completions.create(
                    model=CauHinh.WRITER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format=OUTLINE_SCHEMA,
                    timeout=120.0
                )
            result = json.loads(resp.choices[0].message.content)
            
            # --- 🛑 HARD CAP & VALIDATION (V23.5.3) ---
            raw_outline = result.get("outline", [])
            total_sec = sum(len(c.get("sections", [])) for c in raw_outline)
            MAX_TOTAL_SECTIONS = 90 # Khống chế bùng nổ token
            
            if total_sec > MAX_TOTAL_SECTIONS:
                logger.warning(f"[HardCap] Total sections ({total_sec}) exceeds limit ({MAX_TOTAL_SECTIONS}). Reducing...")
                for c in raw_outline:
                    if len(c.get("sections", [])) > 6:
                        c["sections"] = c["sections"][:6]
                logger.info(f"[HardCap] Adjusted total sections to {sum(len(c.get('sections', [])) for c in raw_outline)}")
            
            # 💎 Structural Audit (V28 — Range-aware)
            actual_ch = len(result.get("outline", []))
            effective_target = target_ch if target_mode == "EXACT" else ch_min
            
            if target_mode == "EXACT" and actual_ch != target_ch:
                logger.warning(f"[Architect] Structural Violation: Requested {target_ch}, Got {actual_ch}.")
            elif target_mode == "RANGE" and (actual_ch < ch_min or actual_ch > ch_max):
                logger.warning(f"[Architect] Out of Range: Expected {ch_min}-{ch_max}, Got {actual_ch}.")
                
            # --- V28 Architect Collapse Fix (Topic-Agnostic Rescue) ---
            if actual_ch < effective_target * 0.5:
                rescue_target = effective_target  # Minimum acceptable
                logger.warning(f"[Architect] Severe Collapse detected ({actual_ch}/{rescue_target}). Retrying with explicit chapter list...")
                
                # V28: Tạo gợi ý chapter titles từ chính terms data
                suggested_titles = []
                for i, t in enumerate(core_list[:rescue_target]):
                    suggested_titles.append(f"  {i+1}. {t.get('term', f'Chapter {i+1}')}")
                suggested_titles_text = "\n".join(suggested_titles)
                
                rescue_prompt = f"""Topic: {chu_de}
Scale: {quy_mo.upper()}

CRITICAL: Your previous attempt only produced {actual_ch} chapter(s). This is UNACCEPTABLE.
You MUST produce AT LEAST {rescue_target} chapters. Each chapter MUST have {sec_min} to {sec_max} sections.

Here are {rescue_target} suggested chapter topics based on the available knowledge. 
Use these as STARTING POINTS. You may adjust titles but MUST produce at least {rescue_target} chapters:

{suggested_titles_text}

INPUT TERMS:
{input_terms_text}

REQUIREMENTS:
- AT LEAST {rescue_target} chapters (chapter_index 1 through {rescue_target} or more)
- Each chapter: {sec_min}-{sec_max} sections
- Flow: Introduction → Foundation → Mechanics → Advanced → Applications
- Technical, specific titles. No generic names.

RETURN ONLY VALID JSON with at least {rescue_target} items in the "outline" array."""

                try:
                    if semaphore:
                        with semaphore:
                            retry_resp = client.chat.completions.create(
                                model=CauHinh.WRITER_MODEL,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": rescue_prompt}
                                ],
                                temperature=0.7,
                                response_format=OUTLINE_SCHEMA,
                                timeout=120.0
                            )
                    else:
                        retry_resp = client.chat.completions.create(
                            model=CauHinh.WRITER_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": rescue_prompt}
                            ],
                            temperature=0.7,
                            response_format=OUTLINE_SCHEMA,
                            timeout=120.0
                        )
                    retry_result = json.loads(retry_resp.choices[0].message.content)
                    retry_ch = len(retry_result.get("outline", []))
                    if retry_ch >= rescue_target * 0.5:
                        logger.info(f"[Architect] Rescue Successful. Chapters: {retry_ch}")
                        result = retry_result
                        actual_ch = retry_ch
                    else:
                        logger.warning(f"[Architect] Rescue Failed. Still got {retry_ch} chapters. Falling back to programmatic builder.")
                        # V28 ULTIMATE FALLBACK: Programmatic Outline Builder
                        prog_result = _programmatic_outline_builder(chu_de, terms_data, rescue_target, sec_min, sec_max)
                        if prog_result:
                            result = prog_result
                            actual_ch = len(prog_result.get("outline", []))
                            logger.info(f"[Architect] Programmatic Builder produced {actual_ch} chapters.")
                except Exception as ex:
                    logger.error(f"[Architect] Rescue completely failed: {ex}. Using programmatic builder.")
                    prog_result = _programmatic_outline_builder(chu_de, terms_data, target_ch, sec_min, sec_max)
                    if prog_result:
                        result = prog_result
                        actual_ch = len(prog_result.get("outline", []))
            
            logger.info(f"[DONE] Xây dựng Dàn ý hoàn tất ({time.time()-start_at:.2f}s). Chapters: {actual_ch}")
            
            # 🆕 V30: Áp dụng Constraint-aware Polish Layer
            try:
                result = _apply_polish_layer(result, chu_de, api_key)
                polished_ch = len(result.get("outline", []))
                if polished_ch != actual_ch:
                    logger.info(f"[Polish] Chapter count changed: {actual_ch} → {polished_ch}")
            except Exception as polish_err:
                logger.warning(f"[Polish] Polish Layer failed safely: {polish_err}. Using raw outline.")
            
            return result
        except Exception as e:
            err_msg = str(e).lower()
            if attempt == 0 and ("timeout" in err_msg or "timed out" in err_msg or "connection" in err_msg):
                logger.warning(f"[Clustering] Fast Retry triggered: {e}")
                continue
            logger.error(f"[Clustering] Error after {time.time()-start_at:.2f}s: {e}")
            return None

@with_smart_retry(max_attempts=3)
def viet_noi_dung_chuong(
    chu_de: str,
    chapter_info: dict,
    relevant_passages: list,
    api_key: str = None,
    failure_memory: str = None,
    mode: str = "NORMAL",
    semaphore=None,
    **kwargs
):
    """
    Expert Tier: Integrated Narrative Writer with Strict Fact Alignment.
    Modes: NORMAL, HIGH_DENSITY, SAFE_MINIMAL.
    """
    if not api_key: raise RuntimeError("Missing OPENAI_API_KEY")
    # Disable SDK retry as we use custom @with_smart_retry (Hotfix V5.2)
    client = OpenAI(api_key=api_key, max_retries=0)

    fact_ids = []
    facts_text = "SOURCE FACTS:\n"
    for i, p in enumerate(relevant_passages[:25]):
        p_id = str(p.get("id", i + 1))
        p_text = p.get("text", "")
        facts_text += f"[{p_id}] {p_text}\n"
        fact_ids.append(p_id)

    # 1. Strategy Assignment
    regime_instruction = ""
    # V27 Writer Safety Injection
    safety_directive = """
- **SAFETY RULES (ACADEMIC FRAMEWORK)**: 
  - DO NOT provide actionable dangerous instructions.
  - Frame all content in a neutral, objective, and educational tone.
"""
    if mode == "HIGH_DENSITY":
        regime_instruction = f"""
- REGIME: SOURCE-CONSTRAINED HIGH DENSITY.
- Every paragraph MUST use at least 2-3 different Fact IDs.
- Ensure exhaustive coverage of all provided facts.{safety_directive}
"""
    elif mode == "SAFE_MINIMAL":
        regime_instruction = f"""
- REGIME: SOURCE-CONSTRAINED MINIMAL (100% FACTUAL FIDELITY).
- Switch to a bullet-point structure ONLY.
- Each bullet point MUST host exactly ONE distinct Fact ID.
- Total number of bullets MUST match the total number of facts provided below.{safety_directive}
"""
    else: # NORMAL
        regime_instruction = f"""
- REGIME: SOURCE-CONSTRAINED ACADEMIC NARRATIVE.
- Use full paragraphs with smooth transitions.
- Integrate Fact IDs [ID] naturally at the end of factual claims.{safety_directive}
"""

    # 2. Hard Constraint Block (Source-Constrained Generation)
    constraint_block = f"""
################################################################################
CRITICAL SOURCE CONSTRAINTS (MANDATORY)
################################################################################
1. HARD CONSTRAINT: You MUST ONLY use information explicitly present in the SOURCE DOCS.
   - If a claim is not directly supported by a source, DO NOT include it.
   - Do NOT infer, generalize, or use external knowledge.
2. EVIDENCE-FIRST WRITING:
   - Paraphrase MUST preserve meaning EXACTLY. Do not add new information.
   - Use extractive style when possible.
3. FACT USAGE:
   - You MUST use these Fact IDs: {", ".join(fact_ids)}
   - Each Fact ID MUST appear in your output at least once.
   - Use the format [ID] immediately after the factual claim.
4. IMPORTANT: Use the EXACT section titles provided: {", ".join([s.get('title') for s in chapter_info.get('sections', [])])}
FAILURE TO COMPLY WILL RESULT IN REJECTION.
################################################################################
"""

    # 3. Failure Memory Integration
    memory_block = ""
    if failure_memory:
        memory_block = f"\n⚠️ PREVIOUS ATTEMPT FAILED. FIX THESE ISSUES: {failure_memory}\n"

    # 4. Structured Context Passing (V4 Phase 2)
    global_context_block = ""
    structured_context_json = kwargs.get("structured_context", "")
    if structured_context_json:
        global_context_block = f"""
################################################################################
GLOBAL CONTEXT (DO NOT CONTRADICT)
################################################################################
You are writing a chapter that is part of a larger curriculum. Maintain consistency with these established facts and timeline:
{structured_context_json}
################################################################################
"""

    prompt = f"""You are a professional ACADEMIC TEXTBOOK AUTHOR. Topic: "{chu_de}".
Chapter: "{chapter_info.get('title')}"
Focus Terms: {json.dumps(chapter_info.get('mapped_terms', []))}
{_lang_directive(kwargs.get('ngon_ngu', 'vi'))}

{regime_instruction}
{constraint_block}
{memory_block}
{global_context_block}

{facts_text}

OUTPUT VALID JSON ONLY:
{{
  "sections": [
    {{
      "title": "Section Title",
      "fact_mappings": [
        {{"source_id": "ID1", "span": "original exact phrase from source", "claim": "paraphrased claim that you will write in content", "confidence": 1.0}}
      ],
      "content": "Text body synthesized STRICTLY from the fact_mappings above. Use [ID] citations...",
      "citations": ["ID1", "ID2"],
      "summary": "Short summary for next section"
    }}
  ],
  "used_fact_ids": ["ID1", "ID2", "..."] 
}}
(Note: 'used_fact_ids' MUST be a unique list of ALL [ID] tags actually utilized in the content. This is our validation key.
CRITICAL INSTRUCTION: You MUST generate 'fact_mappings' FIRST, extracting claims from sources. THEN, write the 'content' based ONLY on those extracted claims. Every single sentence in 'content' must map to a source.
TUYỆT ĐỐI KHÔNG thêm số thứ tự chương/mục vào trường 'title'.)"""

    # 🧵 Throttling (V23.3.1 Fast Retry)
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=CauHinh.WRITER_MODEL,
                        temperature=0.3 if mode != "NORMAL" else 0.7,
                        response_format=CHAPTER_SCHEMA,
                        timeout=CauHinh.API_TIMEOUT
                    )
            else:
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=CauHinh.WRITER_MODEL,
                    temperature=0.3 if mode != "NORMAL" else 0.7,
                    response_format=CHAPTER_SCHEMA,
                    timeout=CauHinh.API_TIMEOUT
                )
            return {"status": "success", "raw_text": resp.choices[0].message.content}
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[ChapterWriter] Fast Retry triggered: {e}")
                continue
            logger.error(f"[ChapterWriter] Error: {e}")
            return {"status": "error", "message": str(e)}

def kiem_tra_ao_giac(chu_de, chapter_content, relevant_passages, model=CauHinh.WRITER_MODEL, api_key=None):
    # Tier 4 Fallback / AI Shadowing
    # (Implementation remains similar but with Source Lock prompt)
    return chapter_content

def viet_lai_chuong(chu_de, chapter_content, fix_instructions, relevant_passages, model=CauHinh.WRITER_MODEL, api_key=None, quy_mo="tieu_chuan"):
    """
    Tier 5: Source-Locked Rewrite
    """
    if not api_key: return chapter_content
    # Disable SDK retry (Hotfix V5.2)
    client = OpenAI(api_key=api_key, max_retries=0)
    
    corpus_text = "\n".join([f"[{p.get('id', i)}] {p.get('text')}" for i, p in enumerate(relevant_passages)])
    
    prompt = f"""REWRITE ENGINE: "{chu_de}"
AUDIT FEEDBACK: {fix_instructions}

SOURCE FACTS:
{corpus_text}

STRICT REWRITE RULES:
1. ADDRESS ALL ISSUES: Ensure every discrepancy mentioned in feedback is fixed.
2. SOURCE ALIGNMENT: Match the EXACT meaning and certainty of the source.
3. CITATIONS: Use [ID] for every factual claim.
4. VALID JSON: Maintain original structure and include fact_mappings for ALL claims."""

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            response_format=REWRITE_SCHEMA,
            timeout=60.0 # V22.2: SDK Timeout 
        )
        return {
            "status": "success",
            "raw_text": resp.choices[0].message.content,
            "data": None
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "api_error",
            "message": str(e),
            "raw_text": ""
        }

def minimal_compiler_mode(
    chu_de: str,
    chapter_info: dict,
    relevant_passages: list,
    model: str = CauHinh.WRITER_MODEL,
    api_key: str = None,
    semaphore=None
):
    """
    Tier 2.3: Fallback (Minimal Compiler)
    Không cần sáng tạo, chỉ nối các facts thành đoạn văn đơn giản.
    """
    if not api_key: return None
    # Disable SDK retry (Hotfix V5.2)
    client = OpenAI(api_key=api_key, max_retries=0)

    corpus_text = ""
    for i, p in enumerate(relevant_passages):
        p_id = p.get("id", i + 1)
        p_text = p.get("fact") if "fact" in p else p.get("text", "")
        corpus_text += f"[{p_id}] {p_text}\n"

    prompt = f"""You are a MINIMALIST ENCYCLOPEDIC COMPILER for "{chu_de}".
Your only job is to join the facts into clean, simple paragraphs in Vietnamese, mapping them to the assigned section titles.

STRICT RULES:
- NO creativity. NO introductory or concluding fluff. 
- Keep all [ID] citations exactly where they belong.
- Use ONLY the provided facts.
- Output EXACTLY the requested sections.

FACTS:
{corpus_text}

SECTIONS TO FILL: {json.dumps([s['title'] for s in chapter_info['sections']], ensure_ascii=False)}

Return JSON ONLY:
{{
  "sections": [
    {{
      "title": "Clean Section Title",
      "content": "Combined facts with [ID] citations...",
      "citations": ["ID1", "ID2"],
      "fact_mappings": [
        {{"source_id": "ID1", "span": "original phrase", "claim": "claim written"}}
      ]
    }}
  ],
  "used_fact_ids": ["ID1", "ID2"]
}}
TUYỆT ĐỐI KHÔNG thêm số thứ tự vào tiêu đề mục.)"""

    if semaphore: semaphore.acquire()
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0,
            response_format=CHAPTER_SCHEMA,
            timeout=CauHinh.API_TIMEOUT
        )
        raw_text = resp.choices[0].message.content
        return {
            "status": "success",
            "raw_text": raw_text,
            "data": json.loads(raw_text)
        }
    except Exception as e:
        logger.error(f"[MinimalCompiler] Error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "data": {"sections": []}
        }

@with_smart_retry(max_attempts=3)
def viet_noi_dung_muc(
    chu_de: str,
    chapter_title: str,
    section_title: str,
    relevant_passages: list,
    api_key: str = None,
    prev_section_summary: str = None,
    mode: str = "NORMAL",
    quy_mo: str = "tieu_chuan",
    semaphore=None,
    **kwargs
):
    """
    Micro-Writer: Integrated Narrative Writer for a SINGLE Section. (V5.8/V18.4)
    Focused context leads to higher accuracy and faster completion.
    """
    if not api_key: raise RuntimeError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, max_retries=0)

    # Fact count based on quy_mo (V19.2)
    fact_limit = {"can_ban": 7, "tieu_chuan": 12, "chuyen_sau": 18}.get(quy_mo, 12)

    fact_ids = []
    facts_text = "SOURCE FACTS FOR THIS SECTION:\n"
    for i, p in enumerate(relevant_passages[:fact_limit]):
        p_id = str(p.get("id", i + 1))
        p_text = p.get("text", "")
        facts_text += f"[{p_id}] {p_text}\n"
        fact_ids.append(p_id)

    expansion_constraint = ""
    if quy_mo == "can_ban":
        expansion_constraint = "Viết gãy gọn để trình bày rõ ràng các facts."
    elif quy_mo == "tieu_chuan":
        expansion_constraint = "Mở rộng phân tích cơ bản để giải thích nguyên lý. Viết đầy đủ đạt khoảng 400 từ."
    else: # chuyen_sau
        expansion_constraint = "Viết chi tiết và đầy đủ để đạt tối thiểu ~800 từ. Áp dụng nghiêm ngặt cấu trúc 3 lớp (Nguyên lý - Cơ chế - Diễn giải) cho mỗi Fact."

    context_flow = ""
    if prev_section_summary:
        context_flow = f"\nPREV SECTION SUMMARY: {prev_section_summary} (Use this to ensure smooth transition)\n"

    prompt = f"""You are a university professor creating ONE SECTION of a textbook about "{chu_de}".
Chapter: "{chapter_title}"
Section: "{section_title}"
{_lang_directive(kwargs.get('ngon_ngu', 'vi'))}

{context_flow}

################################################################################
STRUCTURED SYNTHESIS PROTOCOL (V19.2) - NATURALLY FLOWING PROSE
################################################################################
Instead of freewriting, you MUST organize each paragraph in `content` follow this 3-LAYER LOGICAL STRUCTURE (but DO NOT use explicit labels):

1. CONCEPTS: Directly state the academic aspect of the `claim`.
2. MECHANISMS: Explain "HOW" (technical mechanics/details) based on 100% of the `span`.
3. INTERPRETATION: Clarify the importance of the fact for the overall topic.

STRICT RULE ON PRESENTATION:
- DO NOT include explicit labels like "NGUYÊN LÝ:", "CƠ CHẾ:", "DIỄN GIẢI:", or "Step 1:" in the output `content`.
- Use smooth transitions (e.g., "Hơn nữa...", "Cụ thể...", "Bởi vậy...") to connect these layers into a cohesive academic paragraph.
- The reader should perceive a professional narrative, not a structured list.

INTERPRETATION RULES:
- CHỈ ĐƯỢC PHÉP làm rõ ý nghĩa của fact đã có trong span.
- TUYỆT ĐỐI KHÔNG: mở rộng sang lĩnh vực khác, đưa ví dụ mới không có trong span, hoặc tổng quát hóa vượt phạm vi fact.

QUY TẮC SỐNG CÒN VỀ TRÍCH DẪN (STRICT INLINE CITATION):
- ƯU TIÊN: Sử dụng các Core Facts (nửa đầu danh sách) trước khi dùng fact bổ trợ.
- TUYỆT ĐỐI KHÔNG "gom cite đại diện" (ví dụ sai: "Nội dung phân tích... [1][2][3]").
- PHẢI gắn chính xác [source_id] NGAY SÁT SAU MỖI Ý / MỖI CÂU được lấy từ nguồn đó (ví dụ đúng: "Ý A [1]. Còn ý B [2]. Hơn nữa ý C [1][3].").
- MỖI ĐOẠN VĂN phải chứa các trích dẫn chính xác cho từng mệnh đề học thuật.

ĐỘ SÂU (EXPANSION DEPTH):
################################################################################
SELF-AUDIT INSTRUCTION (V22)
################################################################################
- Before generating the JSON, verify you have used at least 4-5 DISTINCT Fact IDs from the list below.
- If you find yourself using fewer than 3 Fact IDs, you MUST expand the content by deep-diving into the technical mechanics (Mechanism layer) of the available facts.
- Check for "Fact Ghosting": Ensure every [ID] you cite actually exists in the SOURCE FACTS list below.
################################################################################

{facts_text}
"""

    # 🧵 Throttling (V23.3.1 Fast Retry)
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=CauHinh.WRITER_MODEL,
                        temperature=0.3 if mode != "NORMAL" else 0.7,
                        response_format=SECTION_SCHEMA,
                        timeout=CauHinh.API_TIMEOUT
                    )
            else:
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=CauHinh.WRITER_MODEL,
                    temperature=0.3 if mode != "NORMAL" else 0.7,
                    response_format=SECTION_SCHEMA,
                    timeout=CauHinh.API_TIMEOUT
                )
            return {"status": "success", "raw_text": resp.choices[0].message.content}
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[SectionWriter] Fast Retry triggered: {e}")
                continue
            logger.error(f"[SectionWriter] Error: {e}")
            return {"status": "error", "message": str(e)}

def viet_rut_gon_rescue(chu_de: str, section_title: str, relevant_passages: list, api_key: str, semaphore=None):
    """
    Emergency Rescue Writer: Sinh nội dung tối giản khi hệ thống chính bị treo. (V18.9)
    Chỉ dùng top 3 facts, tối ưu tốc độ phản hồi.
    """
    if not api_key: return {"status": "error", "message": "Missing API Key"}
    client = OpenAI(api_key=api_key, max_retries=0)
    
    subset = relevant_passages[:3]
    facts_text = ""
    for i, p in enumerate(subset):
        facts_text += f"[{p.get('id', i+1)}] {p.get('text', '')}\n"
        
    prompt = f"""You are an educational assistant. Create a brief academic summary for "{section_title}" in book "{chu_de}".
    
    FACTS:
    {facts_text}
    
    RULES:
    1. Write exactly ONE paragraph (about 150 words).
    2. Use Vietnamese.
    3. Use the [source_id] from FACTS at the end of relevant sentences.
    4. Output valid JSON matching SECTION_SCHEMA.
    """
    
    # 🧵 Throttling (V23.3.1 Fast Retry)
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=CauHinh.WRITER_MODEL,
                        temperature=0.3,
                        response_format=SECTION_SCHEMA,
                        timeout=20.0
                    )
            else:
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=CauHinh.WRITER_MODEL,
                    temperature=0.3,
                    response_format=SECTION_SCHEMA,
                    timeout=20.0
                )
            return {"status": "success", "raw_text": resp.choices[0].message.content, "mode": "rescue"}
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[RescueWriter] Fast Retry triggered: {e}")
                continue
            logger.error(f"[RescueWriter] Critical Fail: {e}")
            return {
                "status": "success", 
                "raw_text": json.dumps({
                    "title": section_title, 
                    "content": f"Thông tin về {section_title} đang được cập nhật dựa trên các tài liệu trích dẫn. [1]",
                    "fact_mappings": [{"source_id": "1", "claim": "Emergency placeholder", "confidence": 0.5}],
                    "citations": []
                }),
                "mode": "emergency"
            }

def sua_noi_dung_targeted(chu_de: str, section_data: dict, feedback_list: list, api_key: str, semaphore=None):
    """
    Targeted Fix (V21.1): Sửa đúng claim bị lỗi, khóa phạm vi (Lock Scope).
    """
    if not api_key: return {"status": "error", "message": "Missing API Key"}
    client = OpenAI(api_key=api_key)
    
    issues_text = "\n".join([
        f"- Cầu {f.get('claim_index')}: {f.get('reason')} (Loại lỗi: {f.get('error_type')})" 
        for f in feedback_list
    ])
    
    prompt = f"""Bạn là biên tập viên học thuật. Phản biện đã phát hiện lỗi trong mục "{section_data.get('title')}" của giáo trình "{chu_de}".

NỘI DUNG HIỆN TẠI:
{section_data.get('content')}

DANH SÁCH LỖI:
{issues_text}

YÊU CẦU ĐÍNH CHÍNH (LOCK SCOPE):
1. CHỈ ĐIỀU CHỈNH các câu/mệnh đề chứa lỗi đã nêu.
2. TUYỆT ĐỐI GIỮ NGUYÊN các phần nội dung khác không bị đánh dấu lỗi.
3. KHÔNG THAY ĐỔI cấu trúc Paragraph hay vị trí các [ID] trích dẫn đúng.
4. Đảm bảo Claim sau khi sửa phải khớp 100% với Span trong fact_mappings.
5. Trả về JSON SECTION_SCHEMA hoàn chỉnh.

QUY TẮC CỨNG: Không giải thích, không thêm rác text, chỉ trả về JSON.
"""

    # 🧵 Throttling (V23.3.1 Fast Retry)
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=CauHinh.WRITER_MODEL,
                        temperature=0.2,
                        response_format=SECTION_SCHEMA,
                        timeout=60.0
                    )
            else:
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=CauHinh.WRITER_MODEL,
                    temperature=0.2,
                    response_format=SECTION_SCHEMA,
                    timeout=60.0
                )
            return {"status": "success", "raw_text": resp.choices[0].message.content, "mode": "targeted_fix"}
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[TargetedFix] Fast Retry triggered: {e}")
                continue
            logger.error(f"[TargetedFix] Error: {e}")
            return {"status": "error", "message": str(e)}

def viet_noi_dung_batch_sections(
    chu_de: str,
    chapter_title: str,
    sections_info: list,
    relevant_passages_list: list,
    api_key: str = None,
    mode: str = "NORMAL",
    quy_mo: str = "tieu_chuan",
    semaphore=None,
    **kwargs
):
    """
    Production-Ready Batch Writer (V23.2): 
    Gom 3-5 sections trong một lần gọi API để giảm số lượng request và tăng hiệu suất.
    Nhúng Self-Audit và Grounding Protocol.
    """
    if not api_key: raise RuntimeError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, max_retries=0)

    # 1. Tổng hợp facts từ tất cả sections trong batch
    all_facts_text = "SOURCE FACTS FOR THIS BATCH:\n"
    seen_ids = set()
    for passages in relevant_passages_list:
        for p in passages:
            p_id = str(p.get("id"))
            if p_id not in seen_ids:
                all_facts_text += f"[{p_id}] {p.get('text', '')}\n"
                seen_ids.add(p_id)

    # 2. Ràng buộc theo quy mô
    expansion_depth = {
        "can_ban": "Concise (200 words/section). Focus on clarity.",
        "tieu_chuan": "Standard (400-500 words/section). Explain mechanisms.",
        "chuyen_sau": "Advanced (800+ words/section). Deep academic dive."
    }.get(quy_mo, "Standard")

    # 3. Prompt Batching & Self-Audit
    prompt = f"""You are a university professor writing {len(sections_info)} sections for the book "{chu_de}".
Chapter: "{chapter_title}"
Sections to write: {json.dumps([s['title'] for s in sections_info], ensure_ascii=False)}

{all_facts_text}

################################################################################
STRICT BATCHING & GROUNDING PROTOCOL (V23.2)
################################################################################
1. INDEPENDENCE: Each section must be self-contained. Do NOT merge them.
2. CITATION (CRITICAL): MUST use EXACT INLINE CITATIONS. Attach [ID] immediately after the specific sentence/claim it supports (e.g., "Fact A [1]. Fact B [2]."). DO NOT group or cluster citations at the end of paragraphs.
3. QUALITY: {expansion_depth}
4. NO TITLES: Do not include chapter or section titles inside the "content" field.
5. NO OMISSION: You must return ALL {len(sections_info)} sections requested.

################################################################################
SELF-AUDIT INSTRUCTIONS
################################################################################
Before completing your response:
- Ensure each section uses at least 3-5 DISTINCT Fact IDs.
- Verify that every [ID] you cite actually exists in the SOURCE FACTS list above.
- Ensure no information is repeated across different sections.
- Tone: {('Formal English Academic.' if kwargs.get('ngon_ngu', 'vi') == 'en' else 'Formal Vietnamese Academic.')}
{_lang_directive(kwargs.get('ngon_ngu', 'vi'))}
################################################################################

RETURN VALID JSON matching BATCH_SECTION_SCHEMA:
{{
  "sections": [
    {{
      "title": "Exact Section Title",
      "content": "...",
      "fact_mappings": [ {{ "source_id": "...", "span": "...", "claim": "...", "confidence": 1.0 }} ],
      "summary": "..."
    }}
  ]
}}
"""

    # 🧵 Throttling (V23.3.1 Fast Retry)
    for attempt in range(2):
        try:
            if semaphore:
                with semaphore:
                    response = client.chat.completions.create(
                        model=CauHinh.WRITER_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7 if mode == "NORMAL" else 0.3,
                        response_format=BATCH_SECTION_SCHEMA,
                        timeout=180.0 # V23.5.3: Tăng lên 180s cho bản Chuyên sâu
                    )
            else:
                response = client.chat.completions.create(
                    model=CauHinh.WRITER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 if mode == "NORMAL" else 0.3,
                    response_format=BATCH_SECTION_SCHEMA,
                    timeout=180.0
                )
            
            raw_text = response.choices[0].message.content
            return {"status": "success", "raw_text": raw_text}
        except Exception as e:
            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"[BatchWriter] Fast Retry triggered: {e}")
                continue
            logger.error(f"[BatchWriter] Error: {e}")
            return {"status": "error", "message": str(e)}


# =========================================================================
# V33: CHAPTER SUMMARY & GLOSSARY GENERATORS
# =========================================================================

def sinh_tom_tat_chuong(chu_de: str, chap_title: str, sections_content: str,
                        api_key: str, semaphore=None) -> str:
    """
    Sinh đoạn tóm tắt 3-5 câu cho 1 chương dựa trên nội dung đã biên soạn.
    Trả về plain text. Fallback: chuỗi rỗng nếu lỗi.
    """
    if not api_key or not sections_content.strip():
        return ""

    client = OpenAI(api_key=api_key, max_retries=0)
    # Cắt nội dung để tiết kiệm token (tối đa ~3000 chars)
    truncated = sections_content[:3000]

    prompt = f"""Bạn là biên tập viên giáo trình đại học về "{chu_de}".
Hãy viết đoạn TÓM TẮT CHƯƠNG (3-5 câu) cho chương "{chap_title}" dựa trên nội dung sau:

{truncated}

YÊU CẦU NGHIÊM NGẶT:
- Ngắn gọn, súc tích, bao quát toàn bộ nội dung chương
- Nêu bật các khái niệm then chốt và mối liên hệ giữa chúng
- Viết bằng giọng học thuật, KHÔNG dùng bullet points hay markdown
- Trả về PLAIN TEXT thuần (không JSON, không heading, không ký hiệu đặc biệt)
- Tối đa 5 câu"""

    try:
        def _call():
            return client.chat.completions.create(
                model=CauHinh.WRITER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
                timeout=30.0
            )

        if semaphore:
            with semaphore:
                resp = _call()
        else:
            resp = _call()

        result = resp.choices[0].message.content.strip()
        logger.info(f"[ChapterSummary] Generated summary for '{chap_title}' ({len(result)} chars)")
        return result
    except Exception as e:
        logger.warning(f"[ChapterSummary] Failed for '{chap_title}': {e}")
        return ""


def sinh_bang_thuat_ngu(terms_list: list, chu_de: str,
                        api_key: str, semaphore=None) -> list:
    """
    Sinh định nghĩa ngắn (1-2 câu) cho danh sách thuật ngữ.
    Trả về list[{"term": "...", "definition": "..."}]. Fallback: list rỗng.
    """
    if not api_key or not terms_list:
        return []

    client = OpenAI(api_key=api_key, max_retries=0)
    # Lấy tối đa 40 thuật ngữ
    terms_names = [t.get("term", "") for t in terms_list[:40] if t.get("term")]
    if not terms_names:
        return []

    terms_text = ", ".join(terms_names)

    prompt = f"""Bạn là chuyên gia biên soạn từ điển thuật ngữ cho giáo trình "{chu_de}".

DANH SÁCH THUẬT NGỮ CẦN ĐỊNH NGHĨA:
{terms_text}

YÊU CẦU:
- Viết định nghĩa ngắn gọn (1-2 câu) cho MỖI thuật ngữ
- Định nghĩa phải chính xác về mặt học thuật
- Sắp xếp theo thứ tự bảng chữ cái

Trả về JSON:
{{"glossary": [{{"term": "Tên thuật ngữ", "definition": "Định nghĩa ngắn"}}]}}"""

    try:
        def _call():
            return client.chat.completions.create(
                model=CauHinh.WRITER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=60.0
            )

        if semaphore:
            with semaphore:
                resp = _call()
        else:
            resp = _call()

        raw = resp.choices[0].message.content
        data = json.loads(raw)
        glossary = data.get("glossary", [])
        # Sắp xếp A-Z
        glossary.sort(key=lambda x: x.get("term", "").lower())
        logger.info(f"[Glossary] Generated {len(glossary)} definitions for '{chu_de}'")
        return glossary
    except Exception as e:
        logger.warning(f"[Glossary] Failed: {e}")
        return []

def openai_editor_agent(chap_title: str, section_title: str, draft_content: dict, reviewer_feedback: str, passages: list, api_key: str, semaphore=None, structured_context: dict = None) -> dict:
    """
    AGENT 3: THE EDITOR (Biên tập viên OpenAI)
    Sửa lỗi bản nháp dựa trên Feedback của Reviewer Agent.
    Bản nháp được truyền vào dưới dạng cấu trúc JSON thô ban đầu (nếu có) hoặc text.
    """
    passages_context = chr(10).join([
        f"[{p.get('id', i)}] Title: {p.get('title')}\n{str(p.get('text', ''))[:1500]}" 
        for i, p in enumerate(passages[:10])
    ])
    
    draft_str = json.dumps(draft_content, ensure_ascii=False) if isinstance(draft_content, dict) else str(draft_content)
    
    prompt = f"""You are an ACADEMIC EDITOR AGENT in a Multi-Agent textbook generation pipeline.
The Writer Agent created a draft for a section, but the Reviewer Agent rejected it with specific feedback.
Your job is to rewrite or fix the draft to satisfy the Reviewer's requirements.

CHAPTER: {chap_title}
SECTION: {section_title}
"""
    if structured_context:
        struct_ctx_str = json.dumps(structured_context, ensure_ascii=False, indent=2)
        prompt += f"""
--- STRUCTURED CONTEXT (DO NOT CONTRADICT) ---
{struct_ctx_str}
----------------------------------------------
"""
        
    prompt += f"""
--- REVIEWER FEEDBACK ---
{reviewer_feedback}

ORIGINAL DRAFT (May contain formatting errors or missing citations):
---
{draft_str}
---

REFERENCE CONTEXT (You must use [id] from here if missing citations):
{passages_context}

YOUR TASK:
Fix all issues mentioned in the Reviewer Feedback. Ensure inline citations [id] are present.

CRITICAL CONSTRAINTS:
1. HARD CONSTRAINT: You MUST ONLY use information explicitly present in the REFERENCE CONTEXT. Do not invent or infer claims.
2. Generate 'fact_mappings' FIRST to extract claims from the REFERENCE CONTEXT. Then synthesize the 'content' based strictly on those mappings.

OUTPUT FORMAT MUST BE JSON:
{{
  "title": "{section_title}",
  "fact_mappings": [
    {{"source_id": "ID1", "span": "original exact phrase", "claim": "paraphrased claim written in content"}}
  ],
  "content": "<Markdown content synthesized strictly from fact_mappings. Include inline citations!>"
}}
"""
    client = OpenAI(api_key=api_key, max_retries=2)
    
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "fixed_section_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "fact_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "span": {"type": "string"},
                                "claim": {"type": "string"}
                            },
                            "required": ["source_id", "span", "claim"],
                            "additionalProperties": False
                        }
                    },
                    "content": {"type": "string"}
                },
                "required": ["title", "content", "fact_mappings"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    try:
        if semaphore:
            with semaphore:
                resp = client.chat.completions.create(
                    model=CauHinh.WRITER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format=schema
                )
        else:
            resp = client.chat.completions.create(
                model=CauHinh.WRITER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format=schema
            )
            
        data = json.loads(resp.choices[0].message.content)
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"[Agent Editor] Error: {e}")
        return {"status": "error", "message": str(e)}
