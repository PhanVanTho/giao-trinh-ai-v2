# -*- coding: utf-8 -*-
import json
import re
from openai import OpenAI
from cau_hinh import CauHinh

def _tach_json(text: str) -> str:
    if not text:
        raise ValueError("Phản hồi rỗng")
    text2 = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text2 = re.sub(r"\s*```$", "", text2.strip())
    if text2.strip().startswith("{") and text2.strip().endswith("}"):
        return text2
    m = re.search(r"\{.*\}", text2, flags=re.DOTALL)
    if not m:
        start = text2.find("{")
        end = text2.rfind("}")
        if start != -1 and end != -1:
            return text2[start:end+1]
        raise ValueError("Không tìm thấy JSON trong phản hồi")
    return m.group(0)

import numpy as np
from google import genai
from google.genai import types

def _tinh_diem_ky_thuat(book: dict, outline_len: int) -> dict:
    """"Stage 1: Tính điểm kỹ thuật (Automated Metrics)"""
    book_vi = book.get("book_vi", book)
    if not book_vi or not book_vi.get("chapters"):
        return {"coverage": 0.0, "citation": 0.0, "structure": 0.0}

    # 1. Coverage Score
    chapters = book_vi.get("chapters", [])
    coverage = min(1.0, len(chapters) / max(1, outline_len))

    # 2. Citation Score (Paragraph Check) & 3. Structure Score
    total_paragraphs = 0
    cited_paragraphs = 0
    total_text_length = 0
    markdown_elements_count = 0

    for ch in chapters:
        for sec in ch.get("sections", []):
            content = sec.get("content", "")
            if not content: continue
            
            # Khởi tạo thuật toán theo đoạn văn thay vì câu
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            total_paragraphs += len(paragraphs)
            
            for p in paragraphs:
                if re.search(r'\[\d+\]', p):
                    cited_paragraphs += 1
            
            # Đếm structure (như markdown heading, bullet, list)
            total_text_length += len(content)
            markdown_elements_count += len(re.findall(r'(#+\s|\d+\.\s|-\s|\*\*)', content))

    citation_score = (cited_paragraphs / total_paragraphs) if total_paragraphs > 0 else 0.0
    structure_score = min(1.0, (markdown_elements_count * 100) / max(1, total_text_length)) # Hệ số chuẩn hoá tạm tính

    return {
        "coverage": round(coverage, 2),
        "citation": round(citation_score, 2),
        "structure": round(structure_score, 2)
    }

def _cross_evaluate_llm(judge_model_type: str, book_to_judge: dict) -> float:
    """Stage 2: Cross-LLM Judge.
    Nếu judge=OpenAI thì nó sẽ chấm book của Gemini.
    Trả về điểm từ 0.0 đến 1.0 (quy đổi từ thang 1-10).
    """
    book_vi = book_to_judge.get("book_vi", book_to_judge)
    if not book_vi or not book_vi.get("chapters"):
        return 0.0

    prompt = f"""
Bạn là một CHUYÊN GIA ĐÁNH GIÁ HỌC THUẬT.
Hãy chấm điểm chất lượng của đoạn giáo trình sau trên thang điểm từ 1 đến 10 dựa trên:
1. Độ chính xác và logic.
2. Tính dễ hiểu và sư phạm.

Giáo trình:
{json.dumps([{"title": c.get('title'), "sections": [s.get('title') for s in c.get('sections', [])]} for c in book_vi.get('chapters', [])], ensure_ascii=False)}

ĐẦU RA BẮT BUỘC (Chỉ trả về 1 con số nguyên hoặc thập phân từ 1 đến 10, không kèm text nào khác):
"""
    try:
        if judge_model_type == "openai":
            # Disable SDK retry (Hotfix V5.2)
            client = OpenAI(api_key=CauHinh.OPENAI_API_KEY, max_retries=0)
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.1
            )
            text = resp.choices[0].message.content.strip()
            score = float(re.findall(r"[\d.]+", text)[0])
            return min(1.0, max(0.0, score / 10.0))
        elif judge_model_type == "gemini":
            api_key = CauHinh.GEMINI_API_KEYS[0] if isinstance(CauHinh.GEMINI_API_KEYS, list) else CauHinh.GEMINI_API_KEYS
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=CauHinh.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            text = resp.text.strip()
            score = float(re.findall(r"[\d.]+", text)[0])
            return min(1.0, max(0.0, score / 10.0))
    except Exception as e:
        print(f"[WARN] Cross-Eval failed for target {judge_model_type}: {e}")
        return None # Trả về None nếu hệ thống lỗi để cân bằng lại trọng số đánh giá

def danh_gia_giao_trinh(book_openai: dict, book_gemini: dict, time_openai: float, time_gemini: float, outline_len: int = 5) -> dict:
    """
    Stage 1: Tính toán Automated Metrics
    Stage 2: Cross-LLM Rating (OpenAI chấm Gemini, Gemini chấm OpenAI)
    Stage 4: Weighted Final Score
    """
    # STAGE 1
    metrics_openai = _tinh_diem_ky_thuat(book_openai, outline_len)
    metrics_gemini = _tinh_diem_ky_thuat(book_gemini, outline_len)

    # Calculate generation speed score (normalized, lower is better. Max wait ~ 300s)
    speed_openai = max(0.0, 1.0 - (time_openai / 200.0))
    speed_gemini = max(0.0, 1.0 - (time_gemini / 200.0))

    # STAGE 2
    # OpenAI evaluates Gemini's book
    score_from_openai_judge = _cross_evaluate_llm("openai", book_gemini)
    # Gemini evaluates OpenAI's book
    score_from_gemini_judge = _cross_evaluate_llm("gemini", book_openai)

    # STAGE 4: Weighted Final Score (Quality vs Performance)
    def calc_quality(llm, cov, cit, struc):
        if llm is None:
            # Rebalance weights: Coverage 0.50, Citation 0.25, Structure 0.25
            return (0.50 * cov) + (0.25 * cit) + (0.25 * struc)
        else:
            # Quality Weights: 0.40 LLM, 0.30 Coverage, 0.15 Citation, 0.15 Structure
            return (0.40 * llm) + (0.30 * cov) + (0.15 * cit) + (0.15 * struc)
        
    def calc_performance(time_taken):
        # Normalize speed (0 to 1, where 0 is > 300s, 1 is 0s)
        return max(0.0, 1.0 - (time_taken / 300.0))

    qual_openai = calc_quality(score_from_gemini_judge, metrics_openai["coverage"], metrics_openai["citation"], metrics_openai["structure"])
    qual_gemini = calc_quality(score_from_openai_judge, metrics_gemini["coverage"], metrics_gemini["citation"], metrics_gemini["structure"])
    
    perf_openai = calc_performance(time_openai)
    perf_gemini = calc_performance(time_gemini)

    # Quyết định winner dựa trên Quality Score. Nếu bằng nhau thì dùng Performance (tốc độ).
    if qual_openai > qual_gemini:
        winner = "openai"
        reason = f"OpenAI Quality: {qual_openai:.2f} > Gemini Quality: {qual_gemini:.2f}"
    elif qual_gemini > qual_openai:
        winner = "gemini"
        reason = f"Gemini Quality: {qual_gemini:.2f} > OpenAI Quality: {qual_openai:.2f}"
    else:
        # Tie breaker
        if perf_openai >= perf_gemini:
            winner = "openai"
            reason = f"Tie in Quality ({qual_openai:.2f}). OpenAI is faster ({time_openai:.1f}s vs {time_gemini:.1f}s)."
        else:
            winner = "gemini"
            reason = f"Tie in Quality ({qual_gemini:.2f}). Gemini is faster ({time_gemini:.1f}s vs {time_openai:.1f}s)."

    return {
        "evaluation": {
            "openai": {
                "quality": {
                    "llm_peer_score": round(score_from_gemini_judge, 2) if score_from_gemini_judge is not None else None,
                    "coverage": metrics_openai["coverage"],
                    "citation": metrics_openai["citation"],
                    "structure": metrics_openai["structure"],
                },
                "performance": {
                    "time_seconds": round(time_openai, 1),
                    "speed_score": round(perf_openai, 2)
                },
                "tong_diem": round(qual_openai, 2) # Backward compatibility UI
            },
            "gemini": {
                "quality": {
                    "llm_peer_score": round(score_from_openai_judge, 2) if score_from_openai_judge is not None else None,
                    "coverage": metrics_gemini["coverage"],
                    "citation": metrics_gemini["citation"],
                    "structure": metrics_gemini["structure"],
                },
                "performance": {
                    "time_seconds": round(time_gemini, 1),
                    "speed_score": round(perf_gemini, 2)
                },
                "tong_diem": round(qual_gemini, 2) # Backward compatibility UI
            }
        },
        "winner": winner,
        "reasoning": reason
    }
