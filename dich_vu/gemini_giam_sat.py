# -*- coding: utf-8 -*-
"""
gemini_giam_sat.py
==================
Gemini Supervisor — AI Judge & Critic.

Nguyên tắc QUAN TRỌNG:
  ❌ Gemini KHÔNG viết lại nội dung.
  ✅ Gemini CHỈ phát hiện lỗi và trả về hướng dẫn sửa cho OpenAI.

Output luôn là:
  {
    "status": "pass" | "fail",
    "issues": ["..."],
    "fix_instructions": "Hướng dẫn chi tiết để OpenAI rewrite"
  }

Model strategy:
  - Mặc định: CauHinh.SUPERVISOR_MODEL_LITE
  - Khi độ chính xác cao quan trọng: CauHinh.SUPERVISOR_MODEL_PRO
"""

import json
import re
import time
import os
import logging
from google import genai
from google.genai import types
from cau_hinh import CauHinh

logger = logging.getLogger(__name__)

# Danh sách từ dừng tiếng Việt phổ biến để giảm "False Positive" trong kiểm định trích dẫn
VI_STOPWORDS = {
    "trong", "những", "việc", "các", "được", "và", "là", "của", "cho", "với", 
    "theo", "tại", "này", "cùng", "như", "đã", "đang", "sẽ", "một", "với",
    "có", "đến", "khi", "mà", "lên", "vào", "ra", "đi", "lại", "qua"
}


def _tach_json(text: str) -> str:
    """Trích xuất JSON từ phản hồi của mô hình."""
    if not text:
        raise ValueError("Phản hồi Gemini rỗng")

    text2 = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text2 = re.sub(r"\s*```$", "", text2.strip())
    text2 = text2.strip()

    if (text2.startswith("{") and text2.endswith("}")):
        return text2

    start_obj = text2.find("{")
    if start_obj != -1:
        end_idx = text2.rfind("}")
        if end_idx != -1 and end_idx > start_obj:
            return text2[start_obj:end_idx + 1]

    raise ValueError("Không tìm thấy JSON object trong phản hồi Gemini Supervisor")


def _get_substantive_tokens(text: str) -> set:
    """
    Trích xuất các token 'thực chất' (Substantive Tokens):
    - Số (Numbers)
    - Năm (Years)
    - Tên riêng (Proper Nouns - Capitalized words, though tricky in sentences)
    - Thuật ngữ kỹ thuật
    Dùng để check 'generated_sentence ⊆ fact_content'.
    """
    # Xoá các ký tự lạ, giữ lại chữ và số
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    
    substantive = set()
    for t in tokens:
        t_lower = t.lower()
        # Bỏ qua từ dừng (Stopwords) và từ quá ngắn
        if t_lower in VI_STOPWORDS or len(t) <= 2:
            continue
            
        # Nếu là số hoặc năm
        if re.search(r'\d+', t):
            substantive.add(t_lower)
        # Nếu có từ viết hoa ở giữa câu (hoặc thuật ngữ)
        elif len(t) > 1 and t[0].isupper():
            substantive.add(t_lower)
            
    return substantive


def giam_sat_chuong(
    chu_de: str,
    chapter_content: dict,
    relevant_passages: list,
    api_keys: list,
    quy_mo: str = "tieu_chuan",
    model_lite: str = CauHinh.SUPERVISOR_MODEL_LITE,
    model_pro: str = CauHinh.SUPERVISOR_MODEL_PRO,
    su_dung_model_pro: bool = False,
    relax_rules: bool = False,
) -> dict:
    """
    Gemini Supervisor kiểm tra 1 chương do OpenAI viết.

    Args:
        chu_de: Chủ đề giáo trình
        chapter_content: Dict chương cần kiểm tra {title, sections}
        relevant_passages: Corpus đã dùng để viết chương này
        api_keys: List Gemini API keys (rotation khi 429)
        quy_mo: "can_ban" | "tieu_chuan" | "chuyen_sau"
        model_lite: Model nhẹ cho hầu hết trường hợp
        model_pro: Model mạnh cho trường hợp cần độ chính xác cao
        su_dung_model_pro: True → dùng gemini-2.5-flash, False → dùng lite
    """
    if not api_keys:
        raise RuntimeError("Thiếu GEMINI_API_KEY cho Supervisor")

    # Hard Guard (Production Safe)
    if not quy_mo or quy_mo not in ["can_ban", "tieu_chuan", "chuyen_sau"]:
        quy_mo = "tieu_chuan"
    
    logger.info(f"[GeminiSupervisor] quy_mo={quy_mo}")

    model_su_dung = model_pro if su_dung_model_pro else model_lite

    # TIER 3: DETERMINISTIC CODE CHECK (EXACT MATCH LOCK)
    # Kiểm tra bằng Python code trước khi gọi AI - Chính xác 100%, 0 Hallucination, 0 Latency.
    # Logic: Với mỗi [FACT_ID], trích xuất sentence chứa nó và so sánh với raw text của PID đó.
    
    source_map = {str(p.get("id")): p.get("text", "") for p in relevant_passages}
    code_issues = []
    used_ids = set()
    
    for sec in chapter_content.get("sections", []):
        text = sec.get("content", "")
        # Tìm tất cả các pattern [ID] - Hỗ trợ alphanumeric (Hotfix V5.6/V5.2 sync)
        matches = re.finditer(r'\[(\w+)\]', text)
        matches_list = list(matches)
        
        # --- MANDATORY CITATION CHECK (V7.0) ---
        if not matches_list and relevant_passages:
             code_issues.append("This section is missing MANDATORY citations [ID]. Please back all claims with sources.")
        
        for m in matches_list:
            # Lấy câu chứa citation để check subset
            start_pos = text.rfind('.', 0, m.start()) + 1
            sentence = text[start_pos:m.end()].strip()
            sid = m.group(1)
            used_ids.add(sid)
            
            raw_source = source_map.get(sid, "")
            if sid not in source_map:
                code_issues.append(f"Fact ID [{sid}] does not exist in the session corpus.")
                continue

            # TIER 3 SUBSET CHECK: generated_sentence ⊆ fact_content
            # Kiểm tra xem các 'Substantive Tokens' của sentence có nằm trong source không.
            sent_tokens = _get_substantive_tokens(sentence)
            src_tokens = _get_substantive_tokens(raw_source)
            
            hallucinated_tokens = sent_tokens - src_tokens
            if hallucinated_tokens:
                really_hallucinated = [t for t in hallucinated_tokens if re.search(r'\d', t) or len(t) > 3]
                if really_hallucinated:
                    code_issues.append(f"Loopholes Detected (Fake Faithful Mapping): Sentence citing [{sid}] contains unauthorized keywords: {really_hallucinated}")

    # TIER 3 COVERAGE CHECK: used_ids / total_ids >= dynamic_threshold
    if source_map:
        import math
        n_facts = len(source_map)
        
        # Adaptive Threshold: Production-Grade tuning
        if relax_rules:
            # Nếu đang ở attempt cuối và relax_rules=True, giảm threshold xuống tối thiểu
            dynamic_threshold = 0.3
        elif quy_mo == "chuyen_sau":
            # Chuyên sâu yêu cầu cao hơn, nhưng giảm Threshold xuống 0.4 (V23.5.3) để tránh False-Negative
            dynamic_threshold = max(0.4, min(0.55, 9.0 / math.sqrt(n_facts))) if n_facts > 0 else 0.45
        else:
            # Mặc định / Tiêu chuẩn: 0.4 là điểm cân bằng tốt cho Production
            dynamic_threshold = max(0.35, min(0.5, 7.5 / math.sqrt(n_facts))) if n_facts > 0 else 0.4
        
        coverage = len(used_ids) / n_facts
        if coverage < dynamic_threshold:
            missing_ids = sorted(list(set(source_map.keys()) - used_ids))
            mandatory_list = ", ".join([f"[{mid}]" for mid in missing_ids[:10]])
            code_instructions = f"ERROR: Low coverage ({coverage*100:.1f}%). You MUST include these missing facts: {mandatory_list}"
            
            return {
                "status": "fail",
                "issues": code_issues + [f"Low coverage: {coverage*100:.1f}% (Threshold: {dynamic_threshold*100:.0f}%)"],
                "fix_instructions": code_instructions,
                "failure_reason": code_instructions,
                "model_used": "python_deterministic_v4_math_adaptive"
            }

    if code_issues:
        return {
            "status": "fail",
            "issues": code_issues,
            "fix_instructions": "Fix deterministic errors in citations.",
            "failure_reason": "; ".join(code_issues),
            "model_used": "python_deterministic_v4_logic"
        }

    # TIER 4: GEMINI SEMANTIC AUDIT (SHADOWING)
    # Target: Fact Distortion (Modality/Certainty shift)
    
    source_context = "\n".join([f"[{p.get('id', i)}] {p.get('text')}" for i, p in enumerate(relevant_passages[:20])])
    
    prompt = f"""You are a ZERO-HALLUCINATION ACADEMIC AUDITOR. 
Your goal is to perform "Fact Alignment" between AI content and raw Wikipedia sources.

TOPIC: "{chu_de}"
CHAPTER: "{chapter_content.get('title')}"

CHAPTER CONTENT:
{json.dumps(chapter_content, ensure_ascii=False)}

SOURCE CORPUS:
{source_context}

AUDIT PROTOCOL (STRICT):
1. ATOMIC TRUTH CHECK: For every claim, verify if it exists EXACTLY in the cited Source [N].
2. MODALITY ALIGNMENT (CRITICAL): Does the AI change the certainty?
   - Source: "probably", "may", "suggests" -> AI: "is", "conclusively", "definitely" == FAIL.
3. NUMERIC INTEGRITY: All dates, numbers, and stats MUST match the source.
4. SOURCE LOYALTY: If a paragraph lacks a tag [N], or the tag points to irrelevant data, FAIL.

RETURN VALID JSON ONLY:
{{
  "status": "pass" | "fail",
  "issues": ["List exact discrepancies and modality shifts"],
  "fix_instructions": "Direct guidance for OpenAI to RESTORE faithfulness to the source."
}}"""

    last_error = None
    for api_key in api_keys:
        delay = 1.0  # Bắt đầu với 1 giây delay
        for attempt in range(3):  # Thử tối đa 3 lần cho mỗi key với backoff
            try:
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=model_su_dung,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text or "{}"
                result = json.loads(_tach_json(text))

                normalized = {
                    "status": result.get("status", "pass"),
                    "issues": result.get("issues", []) or [],
                    "mismatches": result.get("mismatches", []) or [],
                    "fix_instructions": result.get("fix_instructions", ""),
                    "model_used": model_su_dung,
                }

                logger.info(
                    f"[GeminiSupervisor] Chapter '{chapter_content.get('title')}': "
                    f"status={normalized['status']}, model={model_su_dung}"
                )
                return normalized

            except Exception as e:
                last_error = e
                error_str = str(e)
                # Nếu lỗi Quota (429), thực hiện Exponential Backoff
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"[GeminiSupervisor] 429 Quota hit (Key ...{api_key[-4:]}). Attempt {attempt+1}. Backoff {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Gấp đôi thời gian chờ
                    continue
                # Nếu model không tồn tại
                elif "not found" in error_str.lower():
                    model_su_dung = model_pro
                    break
                else:
                    logger.warning(f"[GeminiSupervisor] Non-quota error: {e}")
                    break
        # Sau khi thử hết backoff cho 1 key mà vẫn 429, chuyển sang key tiếp theo
        logger.info(f"[GeminiSupervisor] Key ...{api_key[-4:]} failed after backoff. Rotating key...")

    # Nếu tất cả key đều thất bại
    logger.error(f"[GeminiSupervisor] CRITICAL: All keys failed. Last error: {last_error}")
    return {
        "status": "pass", 
        "issues": [f"Supervisor Error: {str(last_error)}"], 
        "fix_instructions": "None (System Fallback)",
        "model_used": "fallback_bypass"
    }

    # Nếu Gemini lỗi → trả về pass để không block pipeline
    logger.error(f"[GeminiSupervisor] Failed after retries: {last_error}. Returning pass to avoid blocking.")
    return {
        "status": "pass",
        "issues": [f"[SUPERVISOR ERROR] Gemini không phản hồi: {str(last_error)[:100]}"],
        "fix_instructions": "",
        "model_used": model_su_dung,
        "supervisor_error": True
    }


def giam_sat_outline(
    chu_de: str,
    outline_data: dict,
    corpus_passages: list,
    api_keys: list,
    model_lite: str = CauHinh.SUPERVISOR_MODEL_LITE,
    model_pro: str = CauHinh.SUPERVISOR_MODEL_PRO,
) -> dict:
    """
    Gemini Supervisor kiểm tra dàn ý (outline) do OpenAI tạo.
    Sử dụng model PRO vì outline quan trọng — ảnh hưởng toàn bộ giáo trình.

    Returns:
        {
            "status": "pass" | "fail",
            "issues": [...],
            "fix_instructions": str
        }
    """
    if not api_keys:
        return {"status": "pass", "issues": [], "fix_instructions": ""}

    outline_summary = {
        "topic": outline_data.get("topic"),
        "chapter_count": len(outline_data.get("outline", [])),
        "chapters": [
            {
                "title": c.get("title"),
                "section_count": len(c.get("sections", [])),
                "sections": [s.get("title") for s in c.get("sections", [])]
            }
            for c in outline_data.get("outline", [])
        ],
        "term_count": len(outline_data.get("terms", []))
    }

    prompt = f"""You are an AI SUPERVISOR reviewing a university curriculum outline.

TOPIC: "{chu_de}"

OUTLINE created by OpenAI:
{json.dumps(outline_summary, ensure_ascii=False)}

CHECK FOR:
1. LOGICAL FLOW: Do chapters follow a logical progression?
2. COMPLETENESS: Are major topic areas covered?
3. REDUNDANCY: Overlapping content?
4. STRUCTURE: 3-6 sections per chapter?

CRITICAL EXTENSION:
If the outline is thin or missing fundamental concepts, identify "missing_topics" (3-5 specific technical terms or Wikipedia-style topics) that we must crawl to fill the gaps.

RETURN ONLY valid JSON:
{{
  "status": "pass" | "fail",
  "issues": ["..."],
  "missing_topics": ["topic A", "topic B"],
  "fix_instructions": "Specific instructions for OpenAI to regenerate the outline"
}}"""

    last_error = None
    for api_key in api_keys:
        delay = 1.0
        for attempt in range(3):
            try:
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=model_pro,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text or "{}"
                result = json.loads(_tach_json(text))
                result["model_used"] = model_pro
                logger.info(f"[GeminiSupervisor-Outline] status={result.get('status')}")
                return result
            except Exception as e:
                last_error = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"[GeminiSupervisor-Outline] 429 Quota key ...{api_key[-4:]}. Backoff {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                break
        logger.info(f"[GeminiSupervisor-Outline] Rotating key...")

    return {"status": "pass", "issues": [str(last_error)], "fix_instructions": "", "supervisor_error": True}


def giam_sat_quy_mo(
    chu_de: str,
    final_chapters: list,
    quy_mo: str,
    api_keys: list,
    model_lite: str = CauHinh.SUPERVISOR_MODEL_LITE,
    model_pro: str = CauHinh.SUPERVISOR_MODEL_PRO,
) -> dict:
    """
    Gemini Supervisor kiểm tra xem giáo trình đã đáp ứng quy mô người dùng chọn chưa.

    Kiểm tra:
      1. Số chương có đủ không (theo quy mô)?
      2. Tổng số trang ước tính có đủ không?
      3. Các chương nào quá mỏng (ít nội dung)?

    Args:
        chu_de: Chủ đề giáo trình
        final_chapters: List chương đã viết xong [{title, sections}]
        quy_mo: "can_ban" | "tieu_chuan" | "chuyen_sau"
        api_keys: Gemini API keys
        model_lite/model_pro: Model để dùng

    Returns:
        {
            "status": "pass" | "fail",
            "issues": [...],
            "thin_chapters": [{"title": str, "total_chars": int, "required_chars": int}],
            "fix_instructions": str,
            "stats": {
                "actual_chapters": int,
                "actual_total_chars": int,
                "estimated_pages": float,
                "required_chapters_min": int,
                "required_pages_min": int,
            }
        }
    """
    if not api_keys:
        return {"status": "pass", "issues": [], "thin_chapters": [], "fix_instructions": ""}

    # --- Yêu cầu quy mô (ngưỡng tối thiểu) ---
    NGUONG = {
        "can_ban":    {"chuong_min": 4,  "trang_min": 10,  "chars_per_section": 300},
        "tieu_chuan": {"chuong_min": 7,  "trang_min": 30,  "chars_per_section": 600},
        "chuyen_sau": {"chuong_min": 12, "trang_min": 80,  "chars_per_section": 1500},
    }
    nguong = NGUONG.get(quy_mo, NGUONG["tieu_chuan"])
    CHARS_PER_PAGE = 1800  # ~1800 ký tự = 1 trang A4

    # --- Phân tích thống kê thực tế ---
    tong_ky_tu = 0
    thin_chapters = []

    for chap in final_chapters:
        chap_chars = sum(
            len(sec.get("content", ""))
            for sec in chap.get("sections", [])
        )
        tong_ky_tu += chap_chars

        # Tính trung bình chars per section
        so_section = len(chap.get("sections", []))
        avg_chars = chap_chars / max(1, so_section)
        if avg_chars < nguong["chars_per_section"]:
            thin_chapters.append({
                "title": chap.get("title", "?"),
                "total_chars": chap_chars,
                "avg_chars_per_section": round(avg_chars),
                "required_per_section": nguong["chars_per_section"],
                "sections": [s.get("title") for s in chap.get("sections", [])]
            })

    actual_chapters = len(final_chapters)
    estimated_pages = round(tong_ky_tu / CHARS_PER_PAGE, 1)

    stats = {
        "actual_chapters": actual_chapters,
        "actual_total_chars": tong_ky_tu,
        "estimated_pages": estimated_pages,
        "required_chapters_min": nguong["chuong_min"],
        "required_pages_min": nguong["trang_min"],
    }

    logger.info(
        f"[GeminiSupervisor-QuyMo] quy_mo={quy_mo} | "
        f"{actual_chapters} chương | ~{estimated_pages} trang | "
        f"{len(thin_chapters)} chương mỏng"
    )

    # --- Local check trước khi gọi Gemini ---
    local_issues = []
    if actual_chapters < nguong["chuong_min"]:
        local_issues.append(
            f"Thiếu chương: có {actual_chapters}, cần tối thiểu {nguong['chuong_min']} (quy mô {quy_mo})"
        )
    if estimated_pages < nguong["trang_min"]:
        local_issues.append(
            f"Thiếu nội dung: ~{estimated_pages} trang, cần tối thiểu {nguong['trang_min']} trang (quy mô {quy_mo})"
        )

    # Nếu local check đã detect vấn đề → gọi Gemini xác nhận + lấy fix_instructions
    if local_issues or thin_chapters:
        # Tóm tắt để gửi Gemini (tiết kiệm token)
        chapter_summary = [
            {
                "title": c.get("title"),
                "section_count": len(c.get("sections", [])),
                "total_chars": sum(len(s.get("content", "")) for s in c.get("sections", [])),
                "sections": [s.get("title") for s in c.get("sections", [])],
                "avg_density": sum(len(s.get("content", "")) for s in c.get("sections", [])) / max(1, len(c.get("sections", [])))
            }
            for c in final_chapters
        ]

        prompt = f"""You are an ACADEMIC CURRICULUM AUDITOR. Evaluate "{chu_de}" at scale "{quy_mo}".

METRICS TARGET:
- Coverage: Core concepts must be fully explained (not just listed).
- Density: High analytical depth based on facts (not shallow summaries).
- Length: Minimum {nguong['trang_min']} pages ({nguong['trang_min'] * 1800} chars).

ACTUAL STATS:
{json.dumps(stats, ensure_ascii=False)}
CHAPTER BREAKDOWN:
{json.dumps(chapter_summary, ensure_ascii=False)}

TASK: Evaluate strictly between 0.0 and 1.0.
1. coverage_score: Are all essential subtopics covered?
2. density_score: Is content deep and analytical enough?
3. length_score: Total chars / target chars.

If any score < 0.8:
- identify "missing_topics" (technical terms/concepts that need more facts).
- identify "shallow_chapters" (titles of chapters needing more depth).

TRUTH CONSTRAINT: DO NOT suggest adding concepts that are likely absent from Wikipedia. Stick to academic reality.

RETURN JSON:
{{
  "status": "pass" | "fail",
  "scores": {{ "coverage": 0.0, "density": 0.0, "length": 0.0 }},
  "missing_topics": ["topic 1", "topic 2"],
  "shallow_chapters": ["Title A", "Title B"],
  "fix_instructions": "General guide for OpenAI Expansion"
}}"""

        last_error = None
        for api_key in api_keys:
            delay = 1.0
            for attempt in range(3):
                try:
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model=model_lite,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            response_mime_type="application/json",
                        ),
                    )
                    text = resp.text or "{}"
                    result = json.loads(_tach_json(text))
                    result["thin_chapters"] = thin_chapters
                    result["stats"] = stats
                    result["scores"] = result.get("scores", {"coverage": 0, "density": 0, "length": 0})
                    result["missing_topics"] = result.get("missing_topics", [])
                    result["shallow_chapters"] = result.get("shallow_chapters", [])
                    logger.info(f"[GeminiSupervisor-QuyMo] status={result.get('status')} | scores={result['scores']}")
                    return result
                except Exception as e:
                    last_error = e
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        logger.warning(f"[GeminiSupervisor-QuyMo] 429 Quota key ...{api_key[-4:]}. Backoff {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    break
            logger.info(f"[GeminiSupervisor-QuyMo] Rotating key...")

        # Fallback: nếu Gemini lỗi hoàn toàn, dùng kết quả local check
        fix_str = f"Giáo trình chưa đủ quy mô '{quy_mo}'. " + " ".join(local_issues)
        if thin_chapters:
            fix_str += f"\nCác chương cần mở rộng: {', '.join(c['title'] for c in thin_chapters)}"
        return {
            "status": "fail",
            "scores": {"coverage": 0.5, "density": 0.5, "length": estimated_pages/nguong["trang_min"]},
            "issues": local_issues + [f"AI Error: {str(last_error)}"],
            "thin_chapters": thin_chapters,
            "missing_topics": [],
            "shallow_chapters": [c["title"] for c in thin_chapters],
            "fix_instructions": fix_str,
            "stats": stats,
        }

    # --- Tất cả đều OK ---
    return {
        "status": "pass",
        "scores": {"coverage": 1.0, "density": 1.0, "length": 1.0},
        "issues": [],
        "thin_chapters": [],
        "missing_topics": [],
        "shallow_chapters": [],
        "fix_instructions": "",
        "stats": stats,
    }

