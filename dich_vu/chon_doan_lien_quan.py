import numpy as np
import re
from collections import Counter

def chon_top_doan(passages, query: str, top_k: int = 120):
    """
    Chọn top_k passage liên quan nhất bằng Keyword Overlap (Lighweight alternative to TF-IDF).
    Loại bỏ phụ thuộc scikit-learn để deploy nhẹ hơn.
    """
    if not passages:
        return []
    if top_k <= 0:
        return passages

    # Tokenize query
    q_words = set(re.findall(r'\w+', query.lower()))
    if not q_words:
        return passages[:top_k]

    scores = []
    for p in passages:
        text = p.get("text", "").lower()
        # Tính điểm dựa trên số lần xuất hiện của từ khóa
        score = sum(text.count(word) for word in q_words)
        
        # Thưởng cho tiêu đề nếu khớp
        title = p.get("title", "").lower()
        score += sum(title.count(word) for word in q_words) * 5
        
        scores.append(score)

    scores = np.array(scores)
    idx = np.argsort(-scores)
    k = min(top_k, len(passages))
    
    chosen = []
    for i in idx[:k]:
        p = dict(passages[int(i)])
        p["score"] = float(scores[int(i)])
        chosen.append(p)

    return chosen
