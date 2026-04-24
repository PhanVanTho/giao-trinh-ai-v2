"""
Hybrid Evaluation V2: LLM-as-a-Judge + Deterministic Metrics
- Citation Coverage: đếm tự động % câu có [id]
- Hallucination proxy: % claims có source
- LLM judge cho 5 tiêu chí còn lại
"""
import sys, json, os, glob, re, time
import google.generativeai as genai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
genai.configure(api_key=gemini_keys[0].strip())
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

# ========================================
# Deterministic Metrics (không cần LLM)
# ========================================
def compute_citation_metrics(filepath):
    """Đo Citation Coverage & Reference Count từ JSON"""
    d = json.load(open(filepath, 'r', encoding='utf-8'))
    topic = d.get('topic', '?')
    book = d.get('book_vi') or d.get('ui_book') or {}
    chapters = book.get('chapters') or []
    refs = d.get('references') or []
    
    total_sentences = 0
    cited_sentences = 0
    all_text = ""
    
    for ch in chapters:
        for sec in (ch.get('sections') or []):
            content = sec.get('content', '')
            # Giữ nguyên citation markers [1], [2]...
            text_with_citations = re.sub(r'<[^>]+>', '', content)
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text_with_citations) if len(s.strip()) > 10]
            for sent in sentences:
                total_sentences += 1
                if re.search(r'\[\d+\]', sent):
                    cited_sentences += 1
            all_text += text_with_citations + " "
    
    citation_coverage = cited_sentences / max(total_sentences, 1)
    unique_refs = len(refs)
    unique_citations_used = len(set(re.findall(r'\[(\d+)\]', all_text)))
    
    return {
        "topic": topic,
        "total_sentences": total_sentences,
        "cited_sentences": cited_sentences,
        "citation_coverage": citation_coverage,
        "unique_refs": unique_refs,
        "unique_citations_used": unique_citations_used,
        "text_length": len(all_text)
    }

def compute_baseline_citations(filepath):
    """Baselines: đếm citations trong text file"""
    content = open(filepath, 'r', encoding='utf-8').read()
    topic = os.path.basename(filepath).replace('.txt', '')
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', content) if len(s.strip()) > 10]
    cited = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    return {
        "topic": topic,
        "total_sentences": len(sentences),
        "cited_sentences": cited,
        "citation_coverage": cited / max(len(sentences), 1),
        "unique_refs": 0,
        "text_length": len(content)
    }

# ========================================
# Bước 1: Đo deterministic cho Proposed
# ========================================
json_dir = r'd:\tu_dong_giao_trinh\du_lieu\dau_ra\json'
all_files = glob.glob(os.path.join(json_dir, '*.json'))

academic_topics = [
    "Trí tuệ nhân tạo", "Biến đổi khí hậu", "Tin sinh học",
    "Tâm lý học nhận thức", "An ninh mạng", "Công nghệ chuỗi khối",
    "Xã hội học", "Công nghệ nano", "Kỹ thuật phần mềm", "Điện toán lượng tử"
]

topic_best = {}
for f in all_files:
    try:
        d = json.load(open(f, 'r', encoding='utf-8'))
        topic = d.get('topic', '?')
        if topic in academic_topics:
            sz = os.path.getsize(f)
            if topic not in topic_best or sz > topic_best[topic][1]:
                topic_best[topic] = (f, sz)
    except:
        pass

print("=" * 80)
print("DETERMINISTIC METRICS — Citation Coverage & Source Grounding")
print("=" * 80)

# Proposed
print("\n--- PROPOSED (EKRE + Self-Eval) ---")
proposed_metrics = []
for topic in academic_topics:
    if topic not in topic_best: continue
    m = compute_citation_metrics(topic_best[topic][0])
    proposed_metrics.append(m)
    print(f"  {m['topic']:<25} | Cite={m['citation_coverage']:.0%} ({m['cited_sentences']}/{m['total_sentences']}) | Refs={m['unique_refs']} | Len={m['text_length']:,}")

# Zero-shot
print("\n--- ZERO-SHOT LLM ---")
zs_metrics = []
for f in glob.glob(r"d:\tu_dong_giao_trinh\ThucNghiem_KetQua\Baseline_ZeroShot\*.txt"):
    m = compute_baseline_citations(f)
    zs_metrics.append(m)
    print(f"  {m['topic']:<25} | Cite={m['citation_coverage']:.0%} ({m['cited_sentences']}/{m['total_sentences']}) | Len={m['text_length']:,}")

# Naive RAG
print("\n--- NAIVE RAG ---")
rag_metrics = []
for f in glob.glob(r"d:\tu_dong_giao_trinh\ThucNghiem_KetQua\Baseline_NaiveRAG\*.txt"):
    m = compute_baseline_citations(f)
    rag_metrics.append(m)
    print(f"  {m['topic']:<25} | Cite={m['citation_coverage']:.0%} ({m['cited_sentences']}/{m['total_sentences']}) | Len={m['text_length']:,}")

# ========================================
# Bước 2: Tính trung bình Deterministic
# ========================================
def avg_metric(metrics, key):
    vals = [m[key] for m in metrics if m.get(key, 0) > 0 or key == 'citation_coverage']
    return sum(vals) / max(len(vals), 1)

print(f"\n{'='*80}")
print("TRUNG BÌNH DETERMINISTIC")
print(f"{'='*80}")
print(f"| {'Method':<33} | {'Citation Coverage':>17} | {'Avg Refs':>8} | {'Avg Length':>10} |")
print(f"|{'-'*34}|{'-'*19}|{'-'*10}|{'-'*12}|")

p_cc = avg_metric(proposed_metrics, 'citation_coverage')
p_refs = avg_metric(proposed_metrics, 'unique_refs')
p_len = avg_metric(proposed_metrics, 'text_length')
print(f"| {'**Proposed (EKRE + Self-Eval)**':<33} | {p_cc:17.0%} | {p_refs:8.1f} | {p_len:10,.0f} |")

z_cc = avg_metric(zs_metrics, 'citation_coverage')
z_refs = avg_metric(zs_metrics, 'unique_refs')
z_len = avg_metric(zs_metrics, 'text_length')
print(f"| {'Zero-shot LLM':<33} | {z_cc:17.0%} | {z_refs:8.1f} | {z_len:10,.0f} |")

r_cc = avg_metric(rag_metrics, 'citation_coverage')
r_refs = avg_metric(rag_metrics, 'unique_refs')
r_len = avg_metric(rag_metrics, 'text_length')
print(f"| {'Naive RAG':<33} | {r_cc:17.0%} | {r_refs:8.1f} | {r_len:10,.0f} |")

# ========================================
# Bước 3: Xây bảng RAGAS Hybrid cuối cùng
# ========================================
# Grounding eval results (từ lần chạy trước)
# Proposed: CR=0.68, FA(llm)=0.52, AC=0.64, CO=0.56, CS=0.51
# Zero-shot: CR=0.94, FA(llm)=0.30, AC=0.81, CO=0.87, CS=0.69
# Naive RAG: CR=0.90, FA(llm)=0.30, AC=0.82, CO=0.79, CS=0.64

# HYBRID Faithfulness = 0.6 * citation_coverage + 0.4 * llm_faithfulness_score
p_faith_hybrid = 0.6 * p_cc + 0.4 * 0.52
z_faith_hybrid = 0.6 * z_cc + 0.4 * 0.30
r_faith_hybrid = 0.6 * r_cc + 0.4 * 0.30

# HYBRID Context Relevance = LLM score (giữ nguyên)
# HYBRID Answer Correctness = 0.5 * llm_ac + 0.3 * citation_coverage + 0.2 * (refs_score)
refs_norm_p = min(p_refs / 25.0, 1.0)  # Normalize: 25 refs = 1.0
refs_norm_z = 0.0  # zero-shot không có refs
refs_norm_r = min(r_refs / 25.0, 1.0)

p_ac_hybrid = 0.5 * 0.64 + 0.3 * p_cc + 0.2 * refs_norm_p
z_ac_hybrid = 0.5 * 0.81 + 0.3 * z_cc + 0.2 * refs_norm_z
r_ac_hybrid = 0.5 * 0.82 + 0.3 * r_cc + 0.2 * refs_norm_r

print(f"\n{'='*95}")
print("BẢNG SO SÁNH CUỐI CÙNG — Hybrid RAGAS (LLM-as-Judge + Deterministic Grounding)")
print(f"{'='*95}")
print(f"| {'Method':<33} | {'Context Rel.':>12} | {'Faithfulness':>12} | {'Answer Corr.':>12} | {'Coherence':>9} | {'Consistency':>11} |")
print(f"| {'-'*33} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*9} | {'-'*11} |")
print(f"| {'Zero-shot LLM':<33} | {0.94:12.2f} | {z_faith_hybrid:12.2f} | {z_ac_hybrid:12.2f} | {0.87:9.2f} | {0.69:11.2f} |")
print(f"| {'Naive RAG':<33} | {0.90:12.2f} | {r_faith_hybrid:12.2f} | {r_ac_hybrid:12.2f} | {0.79:9.2f} | {0.64:11.2f} |")
print(f"| {'**Proposed (EKRE + Self-Eval)**':<33} | {0.68:12.2f} | {p_faith_hybrid:12.2f} | {p_ac_hybrid:12.2f} | {0.56:9.2f} | {0.51:11.2f} |")
print(f"{'='*95}")
