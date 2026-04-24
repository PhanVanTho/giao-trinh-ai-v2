import os
import re

path = r'd:\tu_dong_giao_trinh\dich_vu\openai_da_buoc.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Stable start point (end of the previous try-block)
start_pattern = r'if attempt == 0 and \("timeout" in str\(e\)\.lower\(\) or "connection" in str\(e\)\.lower\(\)\):'
# Stable end point (start of the next stable logic)
end_pattern = r'start_at = time\.time\(\)'

match_start = re.search(start_pattern, content)
match_end = re.search(end_pattern, content)

if match_start and match_end:
    # We want to keep the start_pattern and the end_pattern, replacing everything in between.
    # The corrupted block starts right after the start_pattern.
    
    new_middle = """
                logger.warning(f"[Tier2Outline] Fast Retry triggered: {e}")
                continue
            logger.error(f"[Tier2Outline] Error: {e}")
            raise e

def nhom_thuat_ngu_va_tao_dan_y(terms_data: dict, api_key: str, chu_de: str, so_chuong: int = 10, quy_mo: str = "tieu_chuan", semaphore=None):
    \"\"\"
    Cognitive Layer: Clustering & Outline.
    Phân cụm thuật ngữ theo lộ trình: Nền tảng -> Cốt lõi -> Ứng dụng.
    \"\"\"
    if not api_key: raise RuntimeError("Thiếu OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # 💎 Diamond Guard: Kiểm tra độ dày dữ liệu trước khi xây dựng dàn ý
    all_terms = terms_data.get("core_terms", []) + terms_data.get("support_terms", [])
    if len(all_terms) < 4:
        logger.warning(f"[Architect] Insufficient data: only {len(all_terms)} terms found. Topic drift likely.")
        raise InsufficientDataError(f"Không tìm thấy đủ thuật ngữ chuyên ngành (chỉ có {len(all_terms)}). Vui lòng thử chủ đề rộng hơn.")

    # 🛠️ Adaptive Structuring: Soft Feasibility Scaling (V24.1)
    cfg = get_structure_config(quy_mo)
    doc_count = len(terms_data.get("core_terms", [])) # Using terms as proxy for info density
    ch_min, ch_max = cfg["ch"]
    sec_min, sec_max = cfg["sec"]
    
    # 💎 Soft Scaling Safety Valve
    if quy_mo == "chuyen_sau":
        if doc_count < 15:
            ch_min, ch_max = 8, 10
            logger.warning(f"[Architect] Low density ({doc_count}). Scaling down to 8-10 chapters.")
        elif doc_count < 25:
            ch_min, ch_max = 10, 12
            logger.info(f"[Architect] Moderate density ({doc_count}). Scaling to 10-12 chapters.")
    
    target_ch = so_chuong if so_chuong > 0 else f"{ch_min} to {ch_max}"

    # 💎 V23.3+ Optimization: Chuyển đổi data sang text list
    core_list = terms_data.get("core_terms", [])
    input_terms_text = "\\n".join([f"- {t.get('term')}" for t in core_list[:80]])

    system_prompt = \"\"\"You are an expert curriculum designer. Organize technical terms into a logical academic outline. 
Directly generate the outline. Do NOT explain reasoning or provide long introductions. Return VALID JSON matching OUTLINE_SCHEMA.\"\"\"

    prompt = f\"\"\"Topic: {chu_de}
Scale: {quy_mo.upper()}

TASK: Create a structured curriculum outline using the terms provided.

INPUT TERMS:
{input_terms_text}

REQUIREMENTS:
1. STRUCTURE:
- EXACTLY {target_ch} chapters.
- Each chapter must have {sec_min} to {sec_max} sections.
- For high chapter counts (12+), focus each chapter on a laser-focused technical aspect.

2. FLOW: Introduction → Foundation → Mechanics → Advanced → Applications.
3. NAMING: Specific, technical, descriptive. No generic \"Introduction\" or \"Overview\" titles.
4. NO REASONING: Output only the JSON.

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
}}
\"\"\"

    """
    
    fixed_content = content[:match_start.end()] + new_middle + content[match_end.start():]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print("SUCCESS")
else:
    print(f"FAILED: start={bool(match_start)}, end={bool(match_end)}")
