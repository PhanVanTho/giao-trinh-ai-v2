import os

path = r'd:\tu_dong_giao_trinh\dich_vu\openai_da_buoc.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# We want to replace the corrupted block.
# Looking at the view_file output:
# Line 523 (index 522): except Exception as e:
# Line 524 (index 523):     if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
# Line 525 (index 524):            # Adaptive Structuring...
# ...
# Line 610 (index 609):     start_at = time.time()

# Corrected block for the end of tao_dan_y_tu_passages and start of nhom_thuat_ngu_va_tao_dan_y
new_lines = [
    '            return json.loads(resp.choices[0].message.content)\n',
    '        except Exception as e:\n',
    '            if attempt == 0 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):\n',
    '                logger.warning(f"[Tier2Outline] Fast Retry triggered: {e}")\n',
    '                continue\n',
    '            logger.error(f"[Tier2Outline] Error: {e}")\n',
    '            raise e\n',
    '\n',
    'def nhom_thuat_ngu_va_tao_dan_y(terms_data: dict, api_key: str, chu_de: str, so_chuong: int = 10, quy_mo: str = "tieu_chuan", semaphore=None):\n',
    '    """\n',
    '    Cognitive Layer: Clustering & Outline.\n',
    '    Phân cụm thuật ngữ theo lộ trình: Nền tảng -> Cốt lõi -> Ứng dụng.\n',
    '    """\n',
    '    if not api_key: raise RuntimeError("Thiếu OPENAI_API_KEY")\n',
    '    client = OpenAI(api_key=api_key)\n',
    '    \n',
    '    # 💎 Diamond Guard: Kiểm tra độ dày dữ liệu trước khi xây dựng dàn ý\n',
    '    all_terms = terms_data.get("core_terms", []) + terms_data.get("support_terms", [])\n',
    '    if len(all_terms) < 4:\n',
    '        logger.warning(f"[Architect] Insufficient data: only {len(all_terms)} terms found. Topic drift likely.")\n',
    '        raise InsufficientDataError(f"Không tìm thấy đủ thuật ngữ chuyên ngành (chỉ có {len(all_terms)}). Vui lòng thử chủ đề rộng hơn.")\n',
    '\n',
    '    # 🛠️ Adaptive Structuring: Soft Feasibility Scaling (V24.1)\n',
    '    cfg = get_structure_config(quy_mo)\n',
    '    doc_count = len(terms_data.get("core_terms", [])) # Using terms as proxy for info density\n',
    '    ch_min, ch_max = cfg["ch"]\n',
    '    sec_min, sec_max = cfg["sec"]\n',
    '    \n',
    '    # 💎 Soft Scaling Safety Valve\n',
    '    if quy_mo == "chuyen_sau":\n',
    '        if doc_count < 15:\n',
    '            ch_min, ch_max = 8, 10\n',
    '            logger.warning(f"[Architect] Low density ({doc_count}). Scaling down to 8-10 chapters.")\n',
    '        elif doc_count < 25:\n',
    '            ch_min, ch_max = 10, 12\n',
    '            logger.info(f"[Architect] Moderate density ({doc_count}). Scaling to 10-12 chapters.")\n',
    '    \n',
    '    target_ch = so_chuong if so_chuong > 0 else f"{ch_min} to {ch_max}"\n',
    '\n',
    '    # 💎 V23.3+ Optimization: Chuyển đổi data sang text list\n',
    '    core_list = terms_data.get("core_terms", [])\n',
    '    input_terms_text = "\\n".join([f"- {t.get(\'term\')}" for t in core_list[:80]])\n',
    '\n',
    '    system_prompt = """You are an expert curriculum designer. Organize technical terms into a logical academic outline. \n',
    'Directly generate the outline. Do NOT explain reasoning or provide long introductions. Return VALID JSON matching OUTLINE_SCHEMA."""\n',
    '\n',
    '    prompt = f"""Topic: {chu_de}\n',
    'Scale: {quy_mo.upper()}\n',
    '\n',
    'TASK: Create a structured curriculum outline using the terms provided.\n',
    '\n',
    'INPUT TERMS:\n',
    '{input_terms_text}\n',
    '\n',
    'REQUIREMENTS:\n',
    '1. STRUCTURE:\n',
    '- EXACTLY {target_ch} chapters.\n',
    '- Each chapter must have {sec_min} to {sec_max} sections.\n',
    '- For high chapter counts (12+), focus each chapter on a laser-focused technical aspect.\n',
    '\n',
    '2. FLOW: Introduction → Foundation → Mechanics → Advanced → Applications.\n',
    '3. NAMING: Specific, technical, descriptive. No generic "Introduction" or "Overview" titles.\n',
    '4. NO REASONING: Output only the JSON.\n',
    '\n',
    'RETURN ONLY JSON matching this format:\n',
    '{{\n',
    '  "topic": "{chu_de}",\n',
    '  "terms": [ {{ "term": "...", "meaning": "mô tả ngắn" }} ],\n',
    '  "outline": [\n',
    '    {{\n',
    '      "chapter_index": 1,\n',
    '      "title": "...",\n',
    '      "sections": [ {{ "title": "...", "recommended_pids": [] }} ]\n',
    '    }}\n',
    '  ]\n',
    '}}\"\"\"\n',
    '\n'
]

# Replacement range (1-indexed lines 522 to 609 inclusive)
# Index range: 521 to 609
final_lines = lines[:521] + new_lines + lines[609:]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print("Repair complete.")
