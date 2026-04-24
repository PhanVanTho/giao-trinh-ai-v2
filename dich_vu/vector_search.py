# -*- coding: utf-8 -*-
# Vector Search V26.2 - Multi-Factor Scoring & Source Diversity
import numpy as np
import math
import time
import logging
from openai import OpenAI
from cau_hinh import CauHinh

import threading
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# --- CACHE HỆ THỐNG (V23.1 Production-Ready) ---
EMBEDDING_CACHE = {}
EMBEDDING_LOCK = threading.Lock()

def _normalize(v):
    """Chuẩn hóa vector (L2 norm) để dùng dot product thay cho cosine similarity."""
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _keyword_overlap(q: str, doc_title: str) -> bool:
    """Kiểm tra giao thoa từ khóa (normalized) giữa query và title."""
    # Import locally to avoid issues
    import unicodedata
    import re
    def norm_text(t):
        t = unicodedata.normalize('NFKD', t or "")
        t = "".join([c for c in t if not unicodedata.combining(c)])
        t = t.replace('đ', 'd').replace('Đ', 'D')
        return t.lower()
    
    q_norm = norm_text(q)
    t_norm = norm_text(doc_title)
    q_words = {w for w in re.findall(r'\w+', q_norm) if len(w) > 2}
    t_words = {w for w in re.findall(r'\w+', t_norm) if len(w) > 2}
    return len(q_words & t_words) > 0

def deduplicate_by_embedding(docs, api_key, threshold=0.9, anchors=None):
    """
    Loại bỏ các tài liệu trùng lặp ngữ nghĩa cao (> 0.9).
    💎 V24.1: Topic Anchor Guard - Luôn giữ documentation quan trọng (anchors).
    """
    if not docs or len(docs) < 2: return docs
    client = OpenAI(api_key=api_key)
    
    # 1. Embed all docs
    texts = [d.get("title", "") + " " + d.get("text", "")[:400] for d in docs]
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = [_normalize(e.embedding) for e in resp.data]
    
    # 2. Sequential Dedup (O(N^2) but N is small)
    unique_indices = []
    skipped_count = 0
    
    # Chuẩn hóa danh sách anchor titles (nếu có)
    anchor_titles = {a.get("title", "").lower() for a in anchors} if anchors else set()

    for i, v in enumerate(vectors):
        is_dup = False
        doc_title = docs[i].get("title", "").lower()
        
        # Nếu là core topic / anchor -> SKIP DEDUP (Giữ lại bằng mọi giá)
        if doc_title in anchor_titles:
            unique_indices.append(i)
            continue

        for out_idx in unique_indices:
            sim = np.dot(v, vectors[out_idx])
            if sim > threshold:
                is_dup = True
                skipped_count += 1
                break
        if not is_dup:
            unique_indices.append(i)
            
    logger.info(f"[Deduplication] Pruned {skipped_count} near-identical docs. Anchors protected: {len(anchor_titles)}. Remaining: {len(unique_indices)}")
    return [docs[i] for i in unique_indices]

def ensure_topic_diversity(docs, max_per_type=3):
    """
    Đảm bảo tính đa dạng của tài liệu theo subtopic_id.
    """
    from collections import defaultdict
    counts = defaultdict(int)
    # Docs are assumed to be sorted by relevance
    diverse_list = []
    
    # Priority for unique subtopics
    for d in docs:
        st_id = d.get("subtopic_id", "general")
        if counts[st_id] < max_per_type:
            diverse_list.append(d)
            counts[st_id] += 1
        else:
            logger.debug(f"[Diversity] Skipping doc from topic '{st_id}' (Limit {max_per_type} reached)")
            
    logger.info(f"[Diversity] Balanced KB covers {len(counts)} sub-topics with {len(diverse_list)} docs.")
    return diverse_list

def precompute_embeddings(docs, main_topic, subtopic, api_key):
    """
    V27 EKRE Optimization: Embed 1 lần duy nhất cho toàn bộ văn bản.
    """
    if not docs: return None, None, []
    client = OpenAI(api_key=api_key, max_retries=0)

    # 1. Embed targets
    resp = client.embeddings.create(model="text-embedding-3-small", input=[main_topic, subtopic])
    main_v = _normalize(resp.data[0].embedding)
    sub_v = _normalize(resp.data[1].embedding)
    
    # 2. Embed all docs in one batch (respecting max input count)
    doc_texts = [d.get("title", "") + " " + d.get("text", "")[:300] for d in docs]
    doc_vectors = []
    for i in range(0, len(doc_texts), 500):
        batch = doc_texts[i:i+500]
        resp_batch = client.embeddings.create(model="text-embedding-3-small", input=batch)
        doc_vectors.extend([_normalize(e.embedding) for e in resp_batch.data])
        
    return main_v, sub_v, doc_vectors


VIET_STOPWORDS = {"của", "và", "là", "các", "những", "cho", "trong", "một", "ở", "tại", "với", "từ", "đến", "về"}

def compute_bm25_score(query_terms, doc_text, avg_doc_len, doc_len, doc_freqs, N, k1=1.5, b=0.75):
    score = 0.0
    text_lower = doc_text.lower()
    for term in query_terms:
        tf = text_lower.count(term)
        if tf == 0: continue
        df = doc_freqs.get(term, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / max(1, avg_doc_len)))
        score += idf * (numerator / denominator)
    return score

def hybrid_semantic_filter_cached(docs, main_v, sub_v, doc_vectors, main_topic, threshold=0.4, truth_seed=None):
    """
    EKRE V27: True Hybrid Retrieval (BM25 Lexical + Vector Semantic)
    """
    if not docs or not doc_vectors: return []
    
    # Chuẩn bị cho BM25
    N = len(docs)
    avg_doc_len = sum(len(d.get("text", "")) for d in docs) / max(1, N)
    
    entity_name = main_topic
    if truth_seed and "entity_name" in truth_seed:
        entity_name = truth_seed["entity_name"]
        
    query_words = [w for w in re.findall(r'\w+', entity_name.lower()) if w not in VIET_STOPWORDS and len(w) > 1]
    
    # Tính Document Frequency cho BM25
    doc_freqs = {}
    for w in query_words:
        doc_freqs[w] = sum(1 for d in docs if w in d.get("text", "").lower())
        
    # Xác định Entity Confidence để gán trọng số
    has_core_entity = any(d.get("is_core", False) for d in docs)
    lexical_weight = 0.7 if has_core_entity else 0.5

    # --- Pass 1: Semantic & Lexical Scoring ---
    scored_docs = []
    max_bm25 = 0.0
    
    for doc, doc_v in zip(docs, doc_vectors):
        # 1. Semantic
        sim_main = np.dot(doc_v, main_v)
        sim_sub = np.dot(doc_v, sub_v)
        semantic_score = 0.6 * sim_main + 0.4 * sim_sub
        
        # 2. Lexical (BM25)
        doc_text = doc.get("text", "")
        doc_len = len(doc_text)
        bm25_text = compute_bm25_score(query_words, doc_text, avg_doc_len, doc_len, doc_freqs, N)
        
        # Boost tiêu đề
        title_lower = doc.get("title", "").lower()
        title_match = sum(1 for w in query_words if w in title_lower)
        bm25_title = title_match * 2.0 # Trọng số cao cho title
        
        bm25_total = bm25_text + bm25_title
        if bm25_total > max_bm25: max_bm25 = bm25_total
        
        doc["semantic_score"] = semantic_score
        doc["bm25_score"] = bm25_total
        scored_docs.append(doc)
        
    # Chuẩn hóa BM25 và Rerank
    final_passed = []
    for doc in scored_docs:
        bm25_norm = (doc["bm25_score"] / max_bm25) if max_bm25 > 0 else 0.0
        # Hybrid Score
        raw_sim = lexical_weight * bm25_norm + (1 - lexical_weight) * doc["semantic_score"]
        
        # Category Boost
        if truth_seed and "categories" in truth_seed:
            doc_cats = doc.get("categories", [])
            overlap = set(doc_cats) & set(truth_seed["categories"])
            if overlap:
                raw_sim += 0.05 * len(overlap) # Boost nhẹ cho mỗi category khớp
        
        # V8.1 Cross-lingual Language Bonus
        if doc.get("lang") == "en":
            en_bonus = 0.0
            best_en = truth_seed.get("best_en_alias") if truth_seed else None
            # Tặng điểm nếu tựa đề hoặc nội dung chứa best_en_alias
            if best_en and (best_en.lower() in doc.get("title", "").lower() or best_en.lower() in doc.get("text", "").lower()[:1000]):
                en_bonus += 0.15
            # Phạt nhẹ nếu bài EN quá dài mà không chứa entity (nguy cơ drift)
            if not best_en and len(doc.get("text", "")) > 10000:
                en_bonus -= 0.1
            raw_sim += en_bonus
                
        if raw_sim >= threshold or doc.get("is_core", False) or doc.get("is_en_anchor", False):
            doc["relevance_score"] = raw_sim
            final_passed.append(doc)
            
    if not final_passed:
        return []
    
    if not scored_docs:
        return []

    # --- Pass 2: Multi-Factor Scoring (V26.2) ---
    # Compute quality_score = (sim^2) * log(len) for normalization
    for doc in scored_docs:
        sim = doc.get("relevance_score", 0)
        text_len = max(1, len(doc.get("text", "")))
        doc["quality_score"] = (sim ** 2) * math.log(text_len)
    
    # Normalize quality_score to [0, 1] range for multi-factor formula
    max_quality = max(d["quality_score"] for d in scored_docs)
    if max_quality > 0:
        for doc in scored_docs:
            doc["quality_score_norm"] = doc["quality_score"] / max_quality
    else:
        for doc in scored_docs:
            doc["quality_score_norm"] = 0.0
    
    # --- Pass 3: Source-Level Frequency Weighting (V26.2) ---
    from collections import defaultdict, Counter
    
    # Count how many docs come from each source URL
    source_freq = Counter(d.get("url", "unknown") for d in scored_docs)
    
    for doc in scored_docs:
        sim_component   = doc.get("relevance_score", 0)
        quality_component = doc.get("quality_score_norm", 0)
        priority_weight = 0.0 if doc.get("is_low_priority", False) else 1.0
        
        # Multi-Factor Final Score
        multi_factor = (
            0.7 * sim_component +
            0.2 * quality_component +
            0.1 * priority_weight
        )
        
        # Source Diversity Weight: 1 / frequency (nguồn xuất hiện nhiều → bị giảm điểm)
        url = doc.get("url", "unknown")
        source_weight = 1.0 / max(1, source_freq[url])
        
        doc["multi_factor_score"] = multi_factor
        doc["source_weight"] = source_weight
        doc["diversity_score"] = multi_factor * (0.8 + 0.2 * source_weight)
    
    # --- Pass 4: Diversity Re-ranking with Topic Penalty ---
    topic_counts = defaultdict(int)
    scored_docs = sorted(scored_docs, key=lambda x: x["diversity_score"], reverse=True)
    
    re_ranked = []
    for d in scored_docs:
        sub = d.get("subtopic", "General")
        # Diversity Penalty: -0.05 for every existing doc in this topic
        penalty = topic_counts[sub] * 0.05
        d["diversity_score"] = d["diversity_score"] - penalty
        topic_counts[sub] += 1
        re_ranked.append(d)
    
    result = sorted(re_ranked, key=lambda x: x["diversity_score"], reverse=True)
    
    logger.info(
        f"[HybridFilter-V26.2] Input={len(docs)} | Passed={len(result)} | "
        f"Sources={len(source_freq)} | Threshold={threshold:.3f}"
    )
    return result

def hybrid_semantic_filter(docs, main_topic, subtopic, api_key, threshold=0.4):
    """
    Wrapper dùng cho backward compatibility.
    """
    main_v, sub_v, doc_vectors = precompute_embeddings(docs, main_topic, subtopic, api_key)
    return hybrid_semantic_filter_cached(docs, main_v, sub_v, doc_vectors, main_topic, threshold)

def coverage_aware_ranking(docs, subtopics, target_per_topic=2):
    """
    EKRE LAYER 7: Coverage-aware Ranking.
    Boost score cho những tài liệu thuộc subtopic đang bị thiếu hụt doc.
    """
    if not docs: return []
    
    from collections import defaultdict
    topic_counts = defaultdict(int)
    
    # 1. Đếm số lượng doc hiện có cho mỗi topic (giả sử đã qua Diversity filter)
    for d in docs:
        topic_counts[d.get("subtopic_id")] += 1
        
    # 2. Boost those which have 0 or 1 doc
    for d in docs:
        stid = d.get("subtopic_id")
        if topic_counts[stid] < target_per_topic:
            # Boost 20% relevance score nếu topic này đang 'đói' dữ liệu
            d["relevance_score"] *= 1.2
            
    return sorted(docs, key=lambda x: x.get("relevance_score", 0), reverse=True)

def apply_diversity_control(all_docs, max_per_subtopic=10):
    """
    EKRE LAYER 5: Diversity & Bias Control.
    Đảm bảo mỗi subtopic đóng góp một lượng tài liệu đủ sâu.
    Tăng lên 10 để tránh việc KB bị thu hẹp quá mức ở quy mô lớn.
    """
    from collections import defaultdict
    topic_counts = defaultdict(int)
    balanced_docs = []
    
    # Sort theo điểm Relevance tổng thể trước
    all_docs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    for doc in all_docs:
        stid = doc.get("subtopic_id", "general")
        if topic_counts[stid] < max_per_subtopic:
            balanced_docs.append(doc)
            topic_counts[stid] += 1
            
    return balanced_docs

def tao_vector_db(passages: list, api_key: str, start_id: int = 1):
    """
    Tạo Embeddings cho danh sách các passages bằng OpenAI text-embedding-3-small.
    Lưu ý: api_key phải hợp lệ.
    """
    if not passages or not api_key:
        return []

    client = OpenAI(api_key=api_key, max_retries=1)
    texts = [p.get("text", "") for p in passages]
    
    # OpenAI API cho phép max 2048 chuỗi / request, ta chia batch nhỏ để an toàn
    batch_size = 500
    all_embeddings = [None] * len(texts)
    
    # 1. Tách các text chưa có trong cache
    to_embed_indices = []
    to_embed_texts = []
    
    with EMBEDDING_LOCK:
        for idx, t in enumerate(texts):
            if t in EMBEDDING_CACHE:
                all_embeddings[idx] = EMBEDDING_CACHE[t]
            else:
                to_embed_indices.append(idx)
                to_embed_texts.append(t)
    
    if to_embed_texts:
        logger.info(f"[Embedding] {len(to_embed_texts)}/{len(texts)} new texts to embed.")
        for i in range(0, len(to_embed_texts), batch_size):
            batch_texts = to_embed_texts[i:i + batch_size]
            batch_indices = to_embed_indices[i:i + batch_size]
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts,
                    timeout=15.0 # V22.2: Hard Timeout
                )
                # Lấy vector ra và chuẩn hóa luôn
                batch_vectors = [_normalize(np.array(e.embedding)) for e in resp.data]
                
                with EMBEDDING_LOCK:
                    for idx_in_batch, v in enumerate(batch_vectors):
                        real_idx = batch_indices[idx_in_batch]
                        all_embeddings[real_idx] = v
                        # Lưu vào cache
                        EMBEDDING_CACHE[to_embed_texts[i + idx_in_batch]] = v
                
                # Giãn cách để tránh Rate Limit (V18.3)
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"[WARN] Lỗi khi tạo embeddings batch: {e}")
                time.sleep(2)
    else:
        logger.info(f"[Embedding] 100% Cache Hit for {len(texts)} texts.")

    # Gắn vector và ID vào từng passage (dựa trên start_id để tránh collision)
    passages_db = []
    for idx, (p, emb) in enumerate(zip(passages, all_embeddings)):
        if emb is None: continue # Skip những doc không embed được
        db_item = dict(p) # Copy
        db_item["vector"] = emb
        db_item["id"] = start_id + idx # Link tiếp tục ID
        passages_db.append(db_item)
        
    return passages_db

def tim_kiem_vector(query: str, passages_db: list, api_key: str, top_k: int = 8):
    """
    V26.2: Tìm kiếm thông minh (Semantic Search) với Multi-Factor Ranking.
    - Ưu tiên docs không phải low_priority.
    - Giới hạn cứng MAX_CHUNKS_PER_SOURCE = 3 (từ CauHinh).
    - Source-level weighting giảm điểm nguồn xuất hiện nhiều.
    """
    if not passages_db or not query or not api_key:
        return []
        
    # Snap-shot safety
    local_db = list(passages_db)
    
    # Filter những item thiếu vector
    valid_items = [p for p in local_db if "vector" in p and p["vector"] is not None]
    missing_count = len(local_db) - len(valid_items)
    
    if missing_count > 0:
        logger.warning(f"[STABILITY] {missing_count} passages in DB missing 'vector' key. Skipping...")
        
    if not valid_items: return []
    
    client = OpenAI(api_key=api_key)
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
            timeout=15.0
        )
        query_vector = _normalize(np.array(resp.data[0].embedding))
    except Exception as e:
        logger.error(f"[ERROR] Lỗi khi tạo embedding cho query '{query}': {e}")
        return []
        
    # Tính dot product
    vectors = np.array([p["vector"] for p in valid_items])
    if vectors.shape[0] == 0: return []
    
    scores = np.dot(vectors, query_vector)
    sorted_indices = np.argsort(scores)[::-1]
    
    # V26.2: Giới hạn cứng từ cấu hình
    max_chunks_per_url = getattr(CauHinh, 'EKRE_MAX_CHUNKS_PER_SOURCE', 3)
    
    # Ngưỡng điểm chất lượng: 50% điểm của top 1 (nới lỏng hơn V24 để tăng recall)
    top_score = float(scores[int(sorted_indices[0])])
    min_acceptable_score = max(0.25, top_score * 0.50)
    
    # --- V26.2: Two-Tier Collection (Normal → Low Priority) ---
    normal_results = []
    low_priority_results = []
    url_counts = {}
    
    for idx in sorted_indices:
        p = dict(valid_items[int(idx)])
        current_score = float(scores[int(idx)])
        
        if current_score < min_acceptable_score:
            continue
            
        url = p.get("url", "unknown")
        url_counts[url] = url_counts.get(url, 0) + 1
        
        # Giới hạn cứng: MAX_CHUNKS_PER_SOURCE
        if url_counts[url] > max_chunks_per_url:
            continue
        
        # Source-level weighting: giảm điểm cho nguồn xuất hiện nhiều
        source_weight = 1.0 / max(1, url_counts[url])
        
        # Multi-factor score cho ranking cuối cùng
        is_low = p.get("is_low_priority", False)
        priority_weight = 0.0 if is_low else 1.0
        
        final_score = (
            0.7 * current_score +
            0.2 * source_weight +
            0.1 * priority_weight
        )
        
        p["score"] = current_score
        p["final_score"] = final_score
        
        if is_low:
            low_priority_results.append(p)
        else:
            normal_results.append(p)
    
    # Xếp hạng: Normal trước, Low Priority sau (cùng sắp theo final_score giảm dần)
    normal_results.sort(key=lambda x: x["final_score"], reverse=True)
    low_priority_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    results = normal_results[:top_k]
    
    # Bổ sung từ low_priority nếu chưa đủ top_k
    if len(results) < top_k:
        remaining = top_k - len(results)
        results.extend(low_priority_results[:remaining])
    
    # Fallback cuối cùng: nếu vẫn thiếu, bỏ qua max_chunks
    if len(results) < top_k:
        used_ids = {p["id"] for p in results}
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            p = dict(valid_items[int(idx)])
            current_score = float(scores[int(idx)])
            
            if p["id"] not in used_ids and current_score >= min_acceptable_score:
                p["score"] = current_score
                p["final_score"] = current_score
                results.append(p)
                used_ids.add(p["id"])
                
    return results
