# -*- coding: utf-8 -*-
# EKRE V26.2 - Adaptive Threshold & Safe Degradation
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import quote
import time
import math
import statistics
import random
import os
import json
import logging
import threading
import numpy as np
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import uuid
from openai import OpenAI
from cau_hinh import CauHinh
from .lam_sach_van_ban import remove_diacritics

logger = logging.getLogger(__name__)

# --- CACHE HỆ THỐNG (V23.1 Production-Ready) ---
WIKI_CACHE = {}
WIKI_LOCK = threading.Lock()
OUTLINE_CACHE = {} # Cache cho outline theo topic
OUTLINE_LOCK = threading.Lock()
SEED_CACHE = {} # Cache cho Truth Seed Extraction (V27)
SEED_LOCK = threading.Lock()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# --- UTILS ---
def _cat_text(text: str, max_chars: int):
    text = "\n".join([ln.strip() for ln in (text or "").splitlines() if ln.strip()])
    return text[:max_chars]

def safe_parse_json(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except:
        return None

def score_knowledge_base(docs):
    """Calculates coverage score: (count * 10) + (total_chars / 1000)"""
    if not docs: return 0
    total_chars = sum(len(d.get("text", "")) for d in docs)
    return (len(docs) * 10) + (total_chars / 1000)

def hard_rule_filter(title: str, content: str):
    if not title or not content: return True
    low_t = title.lower()
    low_c = content.lower()
    if title.replace(".","").replace(" ","").isdigit(): return True
    # Đã nâng cấp Disambiguation Filter đa ngôn ngữ
    forbidden_types = ["disambiguation", "danh sách", "list of", "phim", "movie", "định hướng", "may refer to", "có các nghĩa sau"]
    if any(x in low_t for x in forbidden_types): return True
    if any(x in low_c for x in ["may refer to:", "có thể là:", "định hướng"]) and len(low_c) < 1000: return True
    # V26.2: Nới lỏng ngưỡng độ dài xuống 400 (từ 500)
    if len(content) < 400: return True
    return False


# =============================================================================
# EKRE V26.2 - ADAPTIVE HELPERS
# =============================================================================

def detect_topic_complexity(topic: str, api_key: str) -> str:
    """
    Dùng OpenAI để phân loại độ phức tạp của chủ đề:
      - 'high'   : Chuyên sâu, kỹ thuật cao, ít nguồn (VD: Topological Quantum Computing)
      - 'medium' : Phổ biến trong học thuật, nguồn trung bình (VD: Machine Learning)
      - 'low'    : Phổ thông, dễ tìm nguồn (VD: World War II)
    Fallback: 'medium' nếu gọi API lỗi.
    """
    if not api_key:
        return "medium"
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL,
            messages=[
                {"role": "system", "content": "You are a knowledge classifier. Return ONLY one word: 'high', 'medium', or 'low'."},
                {"role": "user", "content": f"Classify the academic topic complexity for Wikipedia retrieval: '{topic}'.\n- 'high': niche, technical, sparse Wikipedia coverage\n- 'medium': standard academic topic\n- 'low': broad, popular, rich Wikipedia coverage\nReply with only one word."}
            ],
            max_tokens=5,
            temperature=0
        )
        result = resp.choices[0].message.content.strip().lower()
        if result in ("high", "medium", "low"):
            return result
    except Exception as e:
        logger.warning(f"[EKRE] Complexity detection failed: {e}. Defaulting to 'medium'.")
    return "medium"


def get_similarity_floor(complexity: str) -> float:
    """Lấy ngưỡng tương đồng ban đầu dựa trên độ phức tạp chủ đề."""
    return CauHinh.EKRE_SIM_FLOORS.get(complexity, 0.40)


def get_relaxation_step(yield_count: int, target_min: int) -> float:
    """
    Tính bước giãn chuẩn thông minh dựa trên mức độ thiếu hụt dữ liệu.
    - Thiếu nhiều (>70%) → giãn mạnh (0.03)
    - Thiếu vừa (>40%)  → giãn trung bình (0.02)
    - Gần đủ (<40%)     → giãn nhẹ (0.01)
    """
    if target_min <= 0:
        return 0.02
    deficit_ratio = max(0, (target_min - yield_count) / target_min)
    if deficit_ratio > 0.70:
        return 0.03
    elif deficit_ratio > 0.40:
        return 0.02
    else:
        return 0.01


def compute_quality_score(doc: dict) -> float:
    """
    V26.2: Công thức Reweighted Quality Score.
    score = (relevance_score^2) * log(text_length)
    Ưu tiên tính liên quan hơn độ dài, phù hợp với RAG production.
    """
    sim = doc.get("relevance_score", 0)
    text_len = max(1, len(doc.get("text", "")))
    return (sim ** 2) * math.log(text_len)


def _apply_adaptive_yield_gate(
    raw_docs: list,
    topic: str,
    api_key_openai: str,
    quy_mo: str,
    complexity: str,
    fetch_title_func,
    ai_titles: list,
    truth_seed: dict = None
) -> tuple:
    """
    EKRE V26.2.1 - Adaptive Yield Gate với Safe Degradation.

    QUAN TRỌNG: Nhận full raw_docs (chưa lọc) để mỗi iteration có thể 
    chạy hybrid_semantic_filter với ngưỡng thấp hơn trên toàn bộ tập dữ liệu.

    Quy trình:
    1. Chạy hybrid_semantic_filter trên raw_docs với ngưỡng hiện tại.
    2. Nếu thiếu dữ liệu → giảm ngưỡng từng bước nhỏ.
    3. Kiểm tra 3 phanh an toàn sau mỗi vòng.
    4. Trả về (filtered_docs, analytics_dict)
    """
    from dich_vu.vector_search import precompute_embeddings, hybrid_semantic_filter_cached

    target_min    = CauHinh.EKRE_TARGET_YIELD.get(quy_mo, 15)
    min_sim_floor = CauHinh.EKRE_MIN_SIM_FLOOR
    min_quality   = CauHinh.EKRE_MIN_QUALITY_FLOOR
    min_avg_sim   = CauHinh.EKRE_MIN_AVG_SIM.get(complexity, 0.32)
    max_attempts  = CauHinh.EKRE_MAX_RELAXATION_ATTEMPTS
    low_ratio_limit = CauHinh.EKRE_LOW_RATIO_BRAKE
    quality_std   = CauHinh.EKRE_QUALITY_STANDARD

    current_sim_threshold = get_similarity_floor(complexity)
    current_quality       = float(quality_std)
    quality_floor         = float(CauHinh.EKRE_QUALITY_RESCUE)

    analytics = {
        "relaxation_attempts": 0,
        "stop_reason": "INITIAL",
        "avg_sim": 0.0,
        "median_sim": 0.0,
        "std_sim": 0.0,
        "low_priority_count": 0,
        "confidence_score": 0.0,
    }

    allow_last_attempt = False
    filtered = []
    re_scored = []

    # --- EKRE V27: Cache Embeddings (Save ~80% API cost) ---
    logger.info(f"[ADAPTIVE] Precomputing embeddings for {len(raw_docs)} docs...")
    main_v, sub_v, doc_vectors = precompute_embeddings(raw_docs, topic, topic, api_key_openai)

    for attempt in range(max_attempts + 1):  # +1 cho Soft Landing
        analytics["relaxation_attempts"] = attempt

        # --- 1. Chạy hybrid_semantic_filter_cached trên TOÀN BỘ raw_docs với ngưỡng mới ---
        re_scored = hybrid_semantic_filter_cached(
            raw_docs, main_v, sub_v, doc_vectors, topic, 
            threshold=current_sim_threshold, truth_seed=truth_seed
        )

        # --- 2. Apply Quality Score (Reweighted formula) ---
        priority_filtered = []
        low_priority_docs = []
        for doc in re_scored:
            q_score = compute_quality_score(doc)
            doc["quality_score"] = q_score
            if q_score >= current_quality:
                doc["is_low_priority"] = False
                priority_filtered.append(doc)
            elif q_score >= min_quality:
                # Soft Filter: không xóa, đánh dấu low_priority
                doc["is_low_priority"] = True
                low_priority_docs.append(doc)

        # Gộp: normal trước, low_priority sau
        current_batch = priority_filtered + low_priority_docs
        yield_count   = len(current_batch)
        low_count     = len(low_priority_docs)
        low_ratio     = low_count / max(1, yield_count)

        # --- 3. Tính các Metrics ---
        sims = [d.get("relevance_score", 0) for d in current_batch]
        avg_sim    = statistics.mean(sims) if sims else 0.0
        median_sim = statistics.median(sims) if sims else 0.0
        std_sim    = statistics.stdev(sims) if len(sims) > 1 else 0.0
        
        # V27: Top-K Coverage Score for Confidence
        good_sims = sorted([d.get("relevance_score", 0) for d in priority_filtered], reverse=True)
        best_doc_sim = good_sims[0] if good_sims else (max(sims) if sims else 0.0)
        
        # Lấy điểm trung bình của Top K (K=3)
        top_k = 3
        if good_sims:
            top_k_sims = good_sims[:top_k]
            avg_top_k = sum(top_k_sims) / top_k
        else:
            avg_top_k = 0.0
            
        coverage_score = avg_top_k # Coverage lúc này dựa vào chất lượng Top K
        
        # Công thức mới: Kết hợp sức mạnh của doc tốt nhất và mức độ phủ của Top K
        confidence = best_doc_sim * min(1.0, coverage_score)

        analytics.update({
            "avg_sim": round(avg_sim, 4),
            "median_sim": round(median_sim, 4),
            "std_sim": round(std_sim, 4),
            "low_priority_count": low_count,
            "confidence_score": round(confidence, 4),
        })

        logger.info(
            f"[ADAPTIVE] Attempt={attempt} | Sim≥{current_sim_threshold:.3f} | "
            f"Yield={yield_count}/{target_min} | AvgSim={avg_sim:.3f} | "
            f"LowRatio={low_ratio:.2f} | Confidence={confidence:.3f}"
        )

        # --- 4. Kiểm tra điều kiện PASS ---
        is_quality_ok = avg_sim >= min_avg_sim and yield_count >= target_min
        if is_quality_ok:
            filtered = current_batch
            analytics["stop_reason"] = "TARGET_MET"
            logger.info(f"[ADAPTIVE] ✅ TARGET_MET. Confidence={confidence:.3f}")
            break

        # Giữ lại batch tốt nhất nếu đây là lần cuối (Soft Landing)
        if allow_last_attempt:
            filtered = current_batch
            analytics["stop_reason"] = "HARD_FLOOR_REACHED"
            logger.warning(f"[ADAPTIVE] 🛑 HARD_FLOOR_REACHED after Soft Landing. Confidence={confidence:.3f}")
            break

        # --- 5. Phanh an toàn (Noise Brakes) ---
        if avg_sim < min_avg_sim and yield_count > 0:
            filtered = current_batch
            analytics["stop_reason"] = "NOISE_BRAKE_AVG_SIM"
            logger.warning(
                f"[ADAPTIVE] 🔴 NOISE_BRAKE_AVG_SIM: avg_sim={avg_sim:.3f} < {min_avg_sim}. Halting."
            )
            break

        if low_ratio > low_ratio_limit and yield_count > 0:
            filtered = current_batch
            analytics["stop_reason"] = "NOISE_BRAKE_LOW_RATIO"
            logger.warning(
                f"[ADAPTIVE] 🔴 NOISE_BRAKE_LOW_RATIO: {low_ratio:.2f} > {low_ratio_limit}. Halting."
            )
            break

        if attempt >= max_attempts:
            filtered = current_batch
            analytics["stop_reason"] = "MAX_ATTEMPTS_REACHED"
            logger.warning(f"[ADAPTIVE] ⚠️ MAX_ATTEMPTS_REACHED. Confidence={confidence:.3f}")
            break

        # --- 6. Giãn ngưỡng (Step-down) ---
        step = get_relaxation_step(yield_count, target_min)
        new_sim = current_sim_threshold - step
        new_quality = max(min_quality, current_quality * 0.9)

        # Kiểm tra Soft Landing
        if new_sim <= min_sim_floor:
            new_sim = min_sim_floor
            allow_last_attempt = True  # Cho phép 1 lần thử cuối ở đáy
            logger.info(f"[ADAPTIVE] 🟡 Approaching Hard Floor. Enabling Soft Landing for last attempt.")

        current_sim_threshold = max(min_sim_floor, new_sim)
        current_quality       = new_quality
        logger.info(
            f"[ADAPTIVE] ↘ Relaxing thresholds → Sim={current_sim_threshold:.3f}, Quality={current_quality:.1f} (step={step})"
        )

    # Nếu vòng lặp kết thúc không có kết quả nào
    if not filtered and re_scored:
        filtered = re_scored
        analytics["stop_reason"] = "FALLBACK_ACCEPT_ALL"
        logger.warning("[ADAPTIVE] ⚠️ No docs passed gates. Accepting all scored docs as fallback.")

    logger.info(
        f"[ADAPTIVE] Final → StopReason={analytics['stop_reason']} | "
        f"Yield={len(filtered)} | Confidence={analytics['confidence_score']:.3f} | "
        f"LowPriority={analytics['low_priority_count']}"
    )
    return filtered, analytics

def keyword_overlap(q: str, doc_title: str) -> bool:
    """Kiểm tra giao thoa từ khóa (normalized) giữa query và title."""
    q_norm = remove_diacritics(q).lower()
    t_norm = remove_diacritics(doc_title).lower()
    q_words = set(re.findall(r'\w+', q_norm))
    t_words = set(re.findall(r'\w+', t_norm))
    # Ưu tiên các từ có độ dài trên 2 ký tự (tránh các từ như 'là', 'và', 'the')
    meaningful_q = {w for w in q_words if len(w) > 2}
    meaningful_t = {w for w in t_words if len(w) > 2}
    return len(meaningful_q & meaningful_t) > 0

# --- WIKIPEDIA CORE ---
def _get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=False)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "AntigravityBot/1.1 (Academic Curriculum Builder)"})
    return session

def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"

def _page_url(lang: str, title: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

@lru_cache(maxsize=100)
def tim_kiem_tieu_de(lang: str, tu_khoa: str, gioi_han: int = 3):
    session = _get_session()
    try:
        r = session.get(_api(lang), params={"action": "query", "list": "search", "srsearch": tu_khoa, "srlimit": gioi_han, "format": "json"}, timeout=45, verify=False)
        return [it["title"] for it in r.json().get("query", {}).get("search", [])]
    except: return []

def lay_noi_dung_va_lien_ket(lang: str, title: str, max_links: int = 50):
    cache_key = f"{lang}:{title}"
    with WIKI_LOCK:
        if cache_key in WIKI_CACHE:
            return WIKI_CACHE[cache_key]

    session = _get_session()
    try:
        r = session.get(_api(lang), params={"action": "query", "prop": "extracts|links", "titles": title, "explaintext": 1, "redirects": 1, "plnamespace": 0, "pllimit": max_links, "format": "json"}, timeout=45, verify=False)
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if "missing" in page:
            # V29: Wikipedia Title Auto-Correction
            # Nếu tên bài bị sai (VD: "Cách mạng tháng 8" thay vì "Cách mạng tháng Tám")
            # Ta dùng tính năng search để lấy tên chuẩn nhất
            search_results = tim_kiem_tieu_de(lang, title, gioi_han=1)
            if search_results:
                corrected_title = search_results[0]
                # Tránh lặp vô hạn nếu API cứ báo missing
                if corrected_title.lower() != title.lower():
                    logger.info(f"[WIKI] Auto-corrected title '{title}' -> '{corrected_title}'")
                    return lay_noi_dung_va_lien_ket(lang, corrected_title, max_links)
            return "", [], ""
            
        res = page.get("extract", ""), [l.get("title") for l in page.get("links", []) if l.get("title")], _page_url(lang, page.get("title", title))
        
        with WIKI_LOCK:
            WIKI_CACHE[cache_key] = res
        return res
    except: return "", [], ""

def extract_truth_seed(lang: str, title: str, intro: str, api_key: str) -> dict:
    """V27: Trích xuất Truth Seed (Category bằng API + Alias bằng LLM) để neo Entity."""
    cache_key = f"{lang}:{title}"
    with SEED_LOCK:
        if cache_key in SEED_CACHE:
            return SEED_CACHE[cache_key]

    seed = {
        "entity_name": title,
        "aliases": [title, title.lower(), remove_diacritics(title).lower()],
        "best_en_alias": None,
        "categories": []
    }

    # 1. API: Lấy categories (Deterministic)
    session = _get_session()
    try:
        r = session.get(_api(lang), params={
            "action": "query", "prop": "categories", "titles": title, 
            "clshow": "!hidden", "cllimit": 20, "format": "json"
        }, timeout=15, verify=False)
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        if "categories" in page:
            for c in page["categories"]:
                cat_title = c.get("title", "").replace("Thể loại:", "").replace("Category:", "").strip()
                if cat_title:
                    seed["categories"].append(cat_title)
    except Exception as e:
        logger.error(f"[Truth Seed] Error fetching categories: {e}")

    if intro and api_key:
        prompt = f"""Extract aliases, alternative names, acronyms, or common English names for the entity "{title}" based on the text below.
Return ONLY a JSON object with two fields:
1. "aliases": array of strings. Example: ["Alias 1", "Alias 2"].
2. "best_en_alias": the most accurate English name for this entity (or null if none exists).
TEXT: {intro[:1000]}"""
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=CauHinh.WRITER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            parsed = safe_parse_json(content)
            if parsed and isinstance(parsed, dict):
                aliases = parsed.get("aliases", [])
                best_en = parsed.get("best_en_alias")
                if best_en and isinstance(best_en, str):
                    seed["best_en_alias"] = best_en
                    aliases.append(best_en)
            else:
                # Fallback list parsing
                import ast
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != -1:
                    aliases = ast.literal_eval(content[start:end])
                else:
                    aliases = []
                    
            if isinstance(aliases, list):
                for a in aliases:
                    if isinstance(a, str) and a not in seed["aliases"]:
                        seed["aliases"].extend([a, a.lower(), remove_diacritics(a).lower()])
        except Exception as e:
            logger.error(f"[Truth Seed] LLM Alias Error: {e}")

    # Auto Alias Expansion cho địa danh
    lower_t = title.lower()
    if "tỉnh" in lower_t and not lower_t.startswith("tỉnh"):
        seed["aliases"].extend([f"Tỉnh {title}", f"tinh {remove_diacritics(title).lower()}"])
    if "thành phố" in lower_t and not lower_t.startswith("thành phố"):
        seed["aliases"].extend([f"Thành phố {title}", f"thanh pho {remove_diacritics(title).lower()}"])
        
    # Xóa trùng lặp
    seed["aliases"] = list(set(seed["aliases"]))
    
    with SEED_LOCK:
        SEED_CACHE[cache_key] = seed
        
    logger.info(f"[Truth Seed] Anchored '{title}': {len(seed['aliases'])} aliases, {len(seed['categories'])} categories.")
    return seed

# --- ADAPTIVE ENGINES (V17.1) ---
def semantic_query_deduplicate(queries: list, api_key: str, threshold: float = 0.9):
    """Loại bỏ các truy vấn trùng lặp ngữ nghĩa cao (> 0.9) để tiết kiệm tài nguyên."""
    if not queries or len(queries) < 2: return queries
    from openai import OpenAI
    import numpy as np
    client = OpenAI(api_key=api_key)
    
    # Batch embed queries
    q_texts = [q["title"] for q in queries]
    resp = client.embeddings.create(model="text-embedding-3-small", input=q_texts)
    vectors = [np.array(e.embedding) / np.linalg.norm(e.embedding) for e in resp.data]
    
    keep_indices = []
    for i, v in enumerate(vectors):
        is_duplicate = False
        for prev_idx in keep_indices:
            sim = np.dot(v, vectors[prev_idx])
            if sim > threshold:
                is_duplicate = True; break
        if not is_duplicate:
            keep_indices.append(i)
    return [queries[i] for i in keep_indices]

# --- MULTI-AGENT DISCOVERY (Planner -> Searcher -> Critic) ---

def agent_curriculum_planner(topic: str, quy_mo: str, api_key: str) -> list:
    """AGENT 1: The Curriculum Planner (Vạch ra Bản đồ tri thức)"""
    client = OpenAI(api_key=api_key)
    count = {"can_ban": 3, "tieu_chuan": 5, "chuyen_sau": 8}.get(quy_mo, 5)
    
    prompt = f"""You are an Academic Curriculum Architect.
Map out exactly {count} distinct core pillars of knowledge required to write a comprehensive textbook about '{topic}'.
FOCUS on the most important sub-topics (e.g., Origins, Characteristics, Cultural/Social Impact, Key Concepts, Real-world manifestations).
Return MUST be JSON: {{"pillars": ["Pillar 1", "Pillar 2", ...]}}"""
    
    try:
        resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL, 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        data = json.loads(resp.choices[0].message.content)
        raw_pillars = data.get("pillars", [])
        clean_pillars = []
        for p in raw_pillars:
            if isinstance(p, dict):
                clean_pillars.append(p.get("title", str(p)))
            else:
                clean_pillars.append(str(p))
        return clean_pillars
    except Exception as e:
        logger.error(f"[Planner Agent] Error: {e}")
        return ["Tổng quan", "Lịch sử hình thành", "Đặc điểm cơ bản"]

def agent_search_specialist(topic: str, pillars: list, api_key: str, truth_seed: dict = None) -> list:
    """AGENT 2: The Search Specialist (Tự dò dẫm Wikipedia)"""
    client = OpenAI(api_key=api_key)
    queries = []
    
    seed_context = ""
    if truth_seed:
        seed_context = f"\nContext/Core Entity: {truth_seed.get('entity_name', topic)}\nAliases: {truth_seed.get('aliases', [])}\nCategories: {truth_seed.get('categories', [])}\n"
    
    for p in pillars:
        prompt = f"""You are a Wikipedia Search Specialist.
Topic: {topic}{seed_context}
Sub-topic (Pillar): {p}
Generate EXACTLY 2 highly probable Vietnamese Wikipedia article titles for this sub-topic.
Return MUST be JSON: {{"keywords": ["keyword 1", "keyword 2"]}}"""
        try:
            resp = client.chat.completions.create(
                model=CauHinh.WRITER_MODEL, 
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            kw = json.loads(resp.choices[0].message.content).get("keywords", [])
            
            # Tính năng Auto-Correction / Trial-and-Error
            found_for_pillar = False
            for k in kw:
                res = tim_kiem_tieu_de("vi", k, gioi_han=1)
                if res:
                    queries.append({"title": res[0], "lang": "vi", "reason": p})
                    found_for_pillar = True
                    break # Chỉ cần 1 bài cực chuẩn cho mỗi pillar
            
            # Nếu cả 2 keyword đều hỏng, fallback tìm chính tên pillar
            if not found_for_pillar:
                res = tim_kiem_tieu_de("vi", p, gioi_han=1)
                if res: queries.append({"title": res[0], "lang": "vi", "reason": p})
        except Exception as e:
            logger.error(f"[Search Agent] Error on pillar '{p}': {e}")
            continue
            
    return queries

def agent_knowledge_critic(topic: str, pillars: list, found_queries: list, api_key: str) -> list:
    """AGENT 3: The Knowledge Critic (Kiểm toán rỗng)"""
    client = OpenAI(api_key=api_key) # Dùng OpenAI cho nhanh và ổn định JSON
    found_reasons = [q["reason"] for q in found_queries]
    
    prompt = f"""You are a Knowledge Critic.
Topic: {topic}
Required Pillars: {pillars}
Successfully Retrieved Pillars: {found_reasons}

List any Required Pillars that are COMPLETELY MISSING from the retrieved list.
Return MUST be JSON: {{"missing_pillars": ["...", "..."]}}"""
    try:
        resp = client.chat.completions.create(
            model=CauHinh.WRITER_MODEL, 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("missing_pillars", [])
    except Exception as e:
        logger.error(f"[Critic Agent] Error: {e}")
        return []

def agent_link_curator(topic: str, raw_links: list, api_key: str, max_links: int = 15, truth_seed: dict = None) -> list:
    """
    AGENT 4 (V8 Upgraded): The Link Curator
    Lọc link để Spidering bằng Entity Overlap + BM25 + Category heuristics,
    thay vì LLM chậm và đắt đỏ.
    """
    if not raw_links: return []
    
    unique_links = list(set(raw_links))
    
    # 1. Khởi tạo BM25 Lite
    from .lam_sach_van_ban import remove_diacritics
    import math
    from collections import Counter
    import re
    
    def _tokenize_vi(text: str) -> list:
        t = remove_diacritics(text).lower()
        return [w for w in re.findall(r'\w+', t) if len(w) > 2]
    
    query_tokens = _tokenize_vi(topic)
    doc_tokens_list = [_tokenize_vi(link) for link in unique_links]
    
    N = len(doc_tokens_list)
    df = Counter()
    for tokens in doc_tokens_list:
        for t in set(tokens):
            df[t] += 1
            
    idf = {}
    for t in query_tokens:
        idf[t] = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        
    avgdl = sum(len(t) for t in doc_tokens_list) / max(1, N)
    
    # Truth Seed aliases
    seed_aliases = []
    if truth_seed and "aliases" in truth_seed:
        seed_aliases = [remove_diacritics(a).lower() for a in truth_seed["aliases"]]
        
    scored_links = []
    for i, link in enumerate(unique_links):
        tokens = doc_tokens_list[i]
        link_lower_norm = remove_diacritics(link).lower()
        
        # A. Tính điểm BM25
        dl = len(tokens)
        bm25_score = 0.0
        tf = Counter(tokens)
        for t in query_tokens:
            if t in tf:
                freq = tf[t]
                bm25_score += idf[t] * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * dl / avgdl))
                
        # B. Tính điểm Entity Overlap
        overlap_score = 0.0
        if seed_aliases:
            if any(a in link_lower_norm for a in seed_aliases):
                overlap_score = 5.0  # Boost mạnh nếu chứa entity chính
                
        # C. Category/Broad page penalty
        penalty = 0.0
        broad_terms = ["việt nam", "hoa kỳ", "thế giới", "châu á", "lịch sử", "địa lý", "quốc gia", "tỉnh", "thành phố", "danh sách"]
        if any(link_lower_norm == b for b in broad_terms):
            penalty = 10.0 # Trừ điểm các trang quá rộng nếu nó chỉ là 1 chữ
            
        total_score = bm25_score + overlap_score - penalty
        scored_links.append((total_score, link))
        
    # Sắp xếp và lấy top
    scored_links.sort(key=lambda x: x[0], reverse=True)
    
    # Lọc bỏ các link có điểm < 0 (bị phạt quá nặng)
    best_links = [link for score, link in scored_links if score > 0][:max_links]
    
    return best_links

def multi_agent_identify_wiki_titles(topic: str, quy_mo: str = "tieu_chuan", api_key: str = None, is_expansion: bool = False, truth_seed: dict = None, **kwargs):
    """
    Orchestrator: Planner -> Searcher -> Critic Loop
    Thay thế cho openai_identify_wiki_titles cũ.
    """
    if not api_key: return []
    
    logger.info(f"[MULTI-AGENT] 🗺️ Planner is mapping knowledge pillars for '{topic}'...")
    pillars = agent_curriculum_planner(topic, quy_mo, api_key)
    logger.info(f"[MULTI-AGENT] Target Pillars: {pillars}")
    
    logger.info(f"[MULTI-AGENT] 🔎 Search Specialist is hunting {len(pillars)} pillars on Wikipedia...")
    queries = agent_search_specialist(topic, pillars, api_key, truth_seed)
    
    # Critic Audit (Vòng lặp)
    logger.info(f"[MULTI-AGENT] 🧐 Knowledge Critic is auditing coverage...")
    missing = agent_knowledge_critic(topic, pillars, queries, api_key)
    
    if missing:
        logger.warning(f"[MULTI-AGENT] ⚠️ Critic found missing pillars: {missing}. Dispatching Searcher again...")
        # Lần tìm kiếm bổ sung có thể nới lỏng hoặc dùng chính tên chủ đề + missing
        refined_missing = [f"{topic} {m}" for m in missing]
        extra_queries = agent_search_specialist(topic, refined_missing, api_key, truth_seed)
        queries.extend(extra_queries)
        logger.info(f"[MULTI-AGENT] Recovered {len(extra_queries)} additional sources.")
    else:
        logger.info(f"[MULTI-AGENT] ✅ Critic approved 100% coverage.")
        
    # Deduplicate (Loại bỏ các bài trùng lặp ngữ nghĩa)
    final_queries = semantic_query_deduplicate(queries, api_key)
    logger.info(f"[MULTI-AGENT] Final Deduplicated Queries: {[q['title'] for q in final_queries]}")
    return final_queries


def is_title_relevant(truth_seed: dict, title: str) -> bool:
    """Kiểm tra title có chứa bất kỳ alias nào của Truth Seed hay không."""
    t_norm = remove_diacritics(title).lower()
    t_raw_lower = title.lower()
    
    for a in truth_seed.get("aliases", []):
        if a.lower() in t_raw_lower or a.lower() in t_norm:
            return True
            
    entity_name = truth_seed.get("entity_name", "")
    q_words = set(w for w in re.findall(r'\w+', remove_diacritics(entity_name).lower()) if len(w) > 2)
    t_words = set(w for w in re.findall(r'\w+', t_norm) if len(w) > 2)
    
    if not q_words: return False
    
    overlap_ratio = len(q_words & t_words) / max(1, len(q_words))
    
    # V8.1 Adaptive Relaxation: Soften the entity overlap requirement
    if len(q_words) <= 2:
        if overlap_ratio >= 0.5: return True # Nới lỏng: chỉ cần khớp 1 phần
    else:
        if overlap_ratio >= 0.4: return True
        
    # Categories fallback (Level 2 relaxation)
    for cat in truth_seed.get("categories", []):
        cat_words = set(w for w in re.findall(r'\w+', remove_diacritics(cat).lower()) if len(w) > 2)
        if len(cat_words & t_words) > 0:
            return True
            
    return False

# --- ADAPTIVE KNOWLEDGE RETRIEVAL ENGINE (EKRE-V27 DIAMOND) ---
def ekre_discovery_engine(topic: str, api_keys_list: list, quy_mo: str = "tieu_chuan", api_key_openai: str = None, original_topic: str = None):
    from .vector_search import hybrid_semantic_filter, deduplicate_by_embedding, ensure_topic_diversity, coverage_aware_ranking
    from .lam_sach_van_ban import chia_doan, lam_sach_trang
    
    # V8.1: Tách biệt original_topic (cho Wikipedia search) và topic (cho content framing)
    # Nếu topic bị reframe ("Cương thi: Phân tích học thuật..."), original_topic vẫn giữ nguyên "Cương thi"
    search_topic = original_topic or topic
    
    # Diamond X-Ray Stats
    xray = {
        "step": "Discovery",
        "topic": topic,
        "search_topic": search_topic,
        "expanded_queries": [],
        "stats": {"retrieved": 0, "filtered": 0, "final": 0},
        "rejection_reasons": {"disambiguation": 0, "low_relevance": 0, "duplicate": 0}
    }

    logger.info(f"[EKRE-V27] Starting Diamond Discovery: {search_topic} (framed as: {topic})")
    
    # --- STAGE 1: Exact Match & Truth Seed Anchoring ---
    # CRITICAL: Luôn dùng search_topic (ngắn gọn) để tìm trên Wikipedia
    exact_titles = tim_kiem_tieu_de("vi", search_topic, gioi_han=1)
    main_entity = exact_titles[0] if exact_titles else search_topic
    
    content, links, url = lay_noi_dung_va_lien_ket("vi", main_entity)
    intro = content[:1000] if content else ""
    truth_seed = extract_truth_seed("vi", main_entity, intro, api_key_openai)
    
    all_raw_docs = []
    seen_titles = set()
    all_internal_links = []
    stats_lock = threading.Lock()
    
    if content and not hard_rule_filter(main_entity, content):
        all_raw_docs.append({
            "title": main_entity, 
            "text": content, 
            "intro": intro,
            "url": url, 
            "lang": "vi", 
            "subtopic": "Core Entity", 
            "id": str(uuid.uuid4())[:8],
            "is_core": True,
            "categories": truth_seed.get("categories", [])
        })
        seen_titles.add(main_entity.lower())
        if links: all_internal_links.extend(links)
        
    # --- STAGE 2: Semantic Expansion (Guided) ---
    # V8.1: Dùng search_topic (gốc) cho Planner/Searcher để tìm đúng bài Wikipedia
    ai_titles = multi_agent_identify_wiki_titles(search_topic, quy_mo, api_key_openai, truth_seed=truth_seed)
    xray["expanded_queries"] = [q["title"] for q in ai_titles]
    
    def fetch_title(item, sr_limit=5):
        query = item["title"]; lang = item["lang"]
        actual_titles = tim_kiem_tieu_de(lang, query, gioi_han=sr_limit) 
        if not actual_titles: return None
        
        docs = []
        for t in actual_titles:
            lower_t = t.lower()
            with stats_lock:
                xray["stats"]["retrieved"] += 1 
                if lower_t in seen_titles: 
                    xray["rejection_reasons"]["duplicate"] += 1
                    continue
                    
                # V27: Title-Level Hard Filter
                if not is_title_relevant(truth_seed, t):
                    xray["rejection_reasons"]["low_relevance"] += 1
                    continue
                    
                seen_titles.add(lower_t)
                
            content, links, url = lay_noi_dung_va_lien_ket(lang, t)
            
            with stats_lock:
                if links:
                    all_internal_links.extend(links)
            
            # Diamond Filter: Disambiguation + Quality
            if hard_rule_filter(t, content): 
                with stats_lock:
                    xray["rejection_reasons"]["disambiguation"] += 1
                continue
            
            # Diamond Intro: Paragraph + First Heading (khoảng 1000 chars)
            intro_limit = 1000
            intro = content[:intro_limit]
            
            docs.append({
                "title": t, 
                "text": content, 
                "intro": intro,
                "url": url, 
                "lang": lang, 
                "subtopic": item.get("reason", "Main"), 
                "id": str(uuid.uuid4())[:8]
            })
            # V21.7: Removed old retrieved increment here to avoid triple-counting
            # (Now moved to top of loop)
        return docs if docs else None

    if ai_titles:
        # V29 Hybrid: Deterministic Backbone — thu thập tất cả trước, trim sau
        # Loại bỏ race condition: thứ tự kết quả theo thứ tự AI titles, không theo thread speed
        sr_initial = 10 if quy_mo == "chuyen_sau" else 5
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda x: fetch_title(x, sr_limit=sr_initial), ai_titles))
        for i, r in enumerate(results):
            if r: all_raw_docs.extend(r)
            if i % 2 == 0: logger.info(f"[FETCH] Discovery Progress: {len(all_raw_docs)} docs collected...")
            if len(all_raw_docs) >= 500: 
                logger.warning(f"[EKRE] Soft Cap reached (500 docs).")
                break

    # --- ADAPTIVE TRIGGER: CROSS-LINGUAL PROGRESSIVE RETRIEVAL ---
    target_score = {"can_ban": 30, "tieu_chuan": 60, "chuyen_sau": 120}.get(quy_mo, 60)
    current_score = score_knowledge_base(all_raw_docs)
    
    best_en_alias = truth_seed.get("best_en_alias")
    coverage_low = current_score < (target_score * 0.6)
    
    # Global Topic Heuristic (Science, IT, Global Culture, etc)
    global_keywords = ["ai", "thuật toán", "công nghệ", "sinh học", "vật lý", "văn hóa toàn cầu", "hội chứng"]
    is_global_topic = any(k in search_topic.lower() for k in global_keywords)
    
    enable_en = (coverage_low or is_global_topic or (best_en_alias is not None)) and (best_en_alias is not None)
    
    if enable_en:
        logger.info(f"[CROSS-LINGUAL] Triggering EN Retrieval for '{best_en_alias}' (Coverage Low: {coverage_low}, Global: {is_global_topic})")
        # Fetch top 3 EN docs using best_en_alias
        en_titles = tim_kiem_tieu_de("en", best_en_alias, gioi_han=3)
        if en_titles:
            for en_t in en_titles:
                en_content, en_links, en_url = lay_noi_dung_va_lien_ket("en", en_t)
                if en_content and not hard_rule_filter(en_t, en_content):
                    all_raw_docs.append({
                        "title": en_t, 
                        "text": en_content, 
                        "intro": en_content[:1000],
                        "url": en_url, 
                        "lang": "en", 
                        "subtopic": "EN Anchor Expansion", 
                        "id": str(uuid.uuid4())[:8],
                        "is_en_anchor": True
                    })
                    if en_links: all_internal_links.extend(en_links)
    
    # Update score after potential EN expansion
    current_score = score_knowledge_base(all_raw_docs)

    if current_score < target_score:
        logger.info(f"[EKRE] Under-spec KB ({current_score:.1f} < {target_score}). Agent 4 (Spider) is activating...")
        
        # Gọi Agent 4 (The Link Curator) lọc ra các link xuất sắc nhất
        spider_limit = 15 if quy_mo == "chuyen_sau" else 10
        selected_links = agent_link_curator(search_topic, all_internal_links, api_key_openai, max_links=spider_limit, truth_seed=truth_seed)
        
        if selected_links:
            logger.info(f"[MULTI-AGENT] 🕸️ Agent 4 selected {len(selected_links)} internal links for deep spidering.")
            # Chuyển đổi format link thành dạng input cho fetch_title
            spider_queries = [{"title": link, "lang": "vi", "reason": "Spidering Expansion"} for link in selected_links]
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                exp_results = list(executor.map(lambda x: fetch_title(x, sr_limit=2), spider_queries))
            for r in exp_results:
                if r: all_raw_docs.extend(r)
                if len(all_raw_docs) >= 1000:
                    logger.warning(f"[EKRE] Global safety limit (1000) reached during spidering.")
                    break
        else:
            logger.warning(f"[MULTI-AGENT] 🕸️ Agent 4 could not find any valuable links. Falling back to Gemini Expansion...")
            from .gemini_da_buoc import generate_related_topics_gemini
            extra_topics = generate_related_topics_gemini(search_topic, all_raw_docs, quy_mo, api_keys_list)
            if extra_topics:
                extra_titles = multi_agent_identify_wiki_titles(f"{search_topic}: {', '.join(extra_topics)}", quy_mo, api_key_openai, is_expansion=True)
                with ThreadPoolExecutor(max_workers=10) as executor:
                    exp_results = list(executor.map(lambda x: fetch_title(x, sr_limit=3), extra_titles))
                for r in exp_results:
                    if r: all_raw_docs.extend(r)

    # =========================================================================
    # EKRE V26.2: ADAPTIVE THRESHOLD & SAFE DEGRADATION
    # =========================================================================

    # --- Step 1: Nhận diện độ phức tạp của chủ đề ---
    complexity = detect_topic_complexity(search_topic, api_key_openai)
    xray["complexity"] = complexity
    logger.info(f"[EKRE-V26.2] Topic Complexity: {complexity.upper()} | RawDocs: {len(all_raw_docs)}")

    # --- Step 2: Adaptive Yield Gate (Safe Degradation) ---
    # QUAN TRỌNG: Truyền all_raw_docs (đầy đủ) để mỗi iteration có thể mở rộng tập kết quả
    hardened, adaptive_analytics = _apply_adaptive_yield_gate(
        raw_docs=all_raw_docs,
        topic=search_topic,
        api_key_openai=api_key_openai,
        quy_mo=quy_mo,
        complexity=complexity,
        fetch_title_func=fetch_title,
        ai_titles=ai_titles,
        truth_seed=truth_seed
    )
    xray["adaptive"] = adaptive_analytics

    # =========================================================================
    # EKRE V28: MULTI-AGENT CRITIC (Gemini LLM Validator)
    # =========================================================================
    from dich_vu.gemini_da_buoc import gemini_critic_agent
    
    # Sắp xếp theo điểm Vector để đưa bài tốt nhất cho Gemini đọc
    hardened = sorted(hardened, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Lấy top 30 bài tốt nhất để tiết kiệm chi phí và thời gian
    top_candidates = hardened[:30]
    final_approved = []
    
    logger.info(f"[MULTI-AGENT] Sending {len(top_candidates)} top candidates to Critic Agent (Gemini)...")
    
    def _run_critic(doc):
        res = gemini_critic_agent(search_topic, doc.get("title", ""), doc.get("text", ""), api_keys_list)
        doc["critic_approved"] = res.get("is_approved", False)
        doc["critic_reason"] = res.get("reason", "")
        doc["critic_score"] = res.get("confidence_score", 0)
        return doc
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        evaluated_candidates = list(executor.map(_run_critic, top_candidates))
        
    for doc in evaluated_candidates:
        if doc.get("critic_approved"):
            final_approved.append(doc)
        else:
            # Lưu vết vào xray rejection
            with stats_lock:
                xray["rejection_reasons"]["low_relevance"] += 1
            
    logger.info(f"[MULTI-AGENT] Critic approved {len(final_approved)} / {len(top_candidates)} docs.")
    xray["adaptive"]["critic_yield"] = len(final_approved)
    hardened = final_approved

    # --- Step 3: Diversity Coverage Check & Final Expansion ---
    unique_topics_count = len({d.get("subtopic") for d in hardened})
    dropped_count = len(all_raw_docs) - len(hardened)
    logger.info(
        f"[EKRE-V26.2] Post-Gate → Diversity:{unique_topics_count} clusters | "
        f"Yield:{len(hardened)} | QualityDropped:{dropped_count}"
    )

    if unique_topics_count < 5 and quy_mo == "chuyen_sau":
        logger.warning(f"[EKRE-V26.2] Critical Low Diversity ({unique_topics_count} < 5). Triggering final niche expansion...")
        extra_titles = multi_agent_identify_wiki_titles(
            f"{search_topic}: unique specialized mechanics and niche aspects", "can_ban", api_key_openai, is_expansion=True
        )
        niche_floor = CauHinh.EKRE_MIN_SIM_FLOOR  # Nới lỏng tối đa cho niche expansion
        with ThreadPoolExecutor(max_workers=5) as executor:
            for r in executor.map(fetch_title, extra_titles):
                if r:
                    new_scored = hybrid_semantic_filter(r, search_topic, search_topic, api_key_openai, threshold=niche_floor)
                    for nd in new_scored:
                        q = compute_quality_score(nd)
                        if q >= CauHinh.EKRE_MIN_QUALITY_FLOOR:
                            nd["is_low_priority"] = q < CauHinh.EKRE_QUALITY_STANDARD
                            nd["quality_score"] = q
                            hardened.append(nd)

    # --- Step 5: Final Deduplication & Cleanup ---
    hardened = deduplicate_by_embedding(hardened, api_key_openai, threshold=0.92, anchors=ai_titles)

    # --- Step 6: Final Stats Trace ---
    avg_len     = sum(len(d.get("text", "")) for d in hardened) / max(1, len(hardened))
    final_sims  = [d.get("relevance_score", 0) for d in hardened]
    avg_sim     = statistics.mean(final_sims) if final_sims else 0.0
    low_p_count = sum(1 for d in hardened if d.get("is_low_priority", False))

    logger.info(
        f"[STRUCTURE] Yield={len(hardened)} | AvgLen={avg_len:.0f} | AvgSim={avg_sim:.3f} | "
        f"Diversity={unique_topics_count} | LowPriority={low_p_count} | "
        f"Confidence={adaptive_analytics.get('confidence_score', 0):.3f} | "
        f"StopReason={adaptive_analytics.get('stop_reason', 'N/A')}"
    )

    xray["stats"]["filtered"] = len(hardened)
    xray["stats"]["final"]    = len(hardened)
    xray["stats"]["avg_sim"]  = round(avg_sim, 4)
    xray["stats"]["low_priority_count"] = low_p_count
    xray["stats"]["stop_reason"] = adaptive_analytics.get("stop_reason", "N/A")
    xray["stats"]["confidence_score"] = adaptive_analytics.get("confidence_score", 0.0)

    # V29.1: Build evaluated_docs list for UI
    evaluated_docs = []
    hardened_ids = {d.get("id") for d in hardened}
    for d in all_raw_docs:
        status = "kept" if d.get("id") in hardened_ids else "dropped"
        # Identify drop reason mapping roughly
        if status == "kept":
            reason = "Đạt chuẩn (Multi-Agent)"
        elif d.get("critic_reason"):
            reason = f"Critic Loại: {d.get('critic_reason')} (Score: {d.get('critic_score', 0)})"
        elif d.get("is_low_priority"):
            reason = "Điểm Vector thấp (Soft-kept)"
        else:
            reason = "Loại bỏ (Nhiễu/Trùng/Vector kém)"
            
        evaluated_docs.append({
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "status": status,
            "score": round(d.get("relevance_score", 0.0), 3),
            "reason": reason
        })
    
    # Sort docs: kept first, then high score
    evaluated_docs.sort(key=lambda x: (x["status"] == "kept", x["score"]), reverse=True)
    xray["evaluated_docs"] = evaluated_docs

    clean_docs = [lam_sach_trang(d) for d in hardened]
    return {
        "passages": chia_doan(clean_docs),
        "candidates": {d["title"]: {"url": d["url"], "lang": d["lang"]} for d in hardened},
        "hardened_docs": hardened,
        "xray": xray
    }

# --- LEGACY / UTILITY ---
def smart_search_crawl(missing_topics: list, ti_le_en: float = 0.8):
    if not missing_topics: return []
    from .lam_sach_van_ban import chia_doan, lam_sach_trang
    results = []
    def crawl_one(t):
        # Ưu tiên ngôn ngữ dựa trên tỉ lệ yêu cầu (V18.3)
        langs = ["en", "vi"] if random.random() < ti_le_en else ["vi", "en"]
        for lang in langs:
            titles = tim_kiem_tieu_de(lang, t, gioi_han=1)
            for title in titles:
                content, _, url = lay_noi_dung_va_lien_ket(lang, title)
                if content and len(content) > 500:
                    return {"title": title, "text": content, "url": url, "lang": lang, "id": str(uuid.uuid4())[:8]}
        return None
    with ThreadPoolExecutor(max_workers=5) as executor:
        found = [r for r in executor.map(crawl_one, missing_topics[:8]) if r]
    return chia_doan([lam_sach_trang(f) for f in found])

def tao_tai_lieu_tu_wikipedia(chu_de, so_trang_hat_giong=10, so_trang_lien_ket=0, quy_mo="tieu_chuan", **kwargs):
    """Legacy wrapper for backward compatibility."""
    from .vector_search import tao_vector_db
    res = ekre_discovery_engine(chu_de, CauHinh.GEMINI_API_KEYS, quy_mo, CauHinh.OPENAI_API_KEY)
    return res["passages"] # Simplified return for legacy paths
