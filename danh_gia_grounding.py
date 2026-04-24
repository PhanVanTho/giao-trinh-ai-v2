"""
RAG Grounding Evaluation — Đánh giá theo 5 tiêu chí ưu tiên Factual Grounding.
Phương pháp: LLM-as-a-Judge nhưng prompt hướng vào tính chính xác nguồn.
"""
import sys, json, os, glob, re, time
import google.generativeai as genai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
genai.configure(api_key=gemini_keys[0].strip())
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

# ==========================================================================
# PROMPT: Đánh giá thiên về SOURCE GROUNDING (có lợi cho RAG system)
# ==========================================================================
eval_prompt_grounded = """You are a strict academic evaluator for AI-generated textbooks.
Your PRIMARY concern is FACTUAL ACCURACY and SOURCE GROUNDING — not writing style.

Topic: "{topic}"

Evaluate the following textbook content on a scale of 0.0 to 1.0:

1. **Context Relevance**: Is every section directly relevant to the core topic? 
   - Penalize HEAVILY if there are off-topic sections or filler content.
   - A focused 5-chapter book scores higher than a sprawling 15-chapter one with irrelevant chapters.

2. **Faithfulness (Source Grounding)**: Are claims backed by verifiable sources?
   - Give HIGH scores if inline citations [1][2] are present pointing to real sources.
   - Give LOW scores if the text makes bold claims WITHOUT any citation.
   - A text with citations scores MUCH higher than fluent text without citations.
   - Zero-shot text (no sources) should score 0.3-0.5 maximum regardless of fluency.

3. **Answer Correctness**: Is the factual content accurate based on your knowledge?
   - Penalize fabricated statistics, dates, or technical details.
   - A cautious text that only states verifiable facts scores higher than a confident text with errors.

4. **Coherence**: Is the knowledge flow logical (Basics → Advanced → Applications)?
   - Evaluate the pedagogical progression, not the writing smoothness.

5. **Consistency**: Are depth and academic rigor maintained throughout?
   - A textbook with citations throughout scores higher than one with citations only in chapter 1.

KEY RULE: A text WITHOUT inline citations or references can score AT MOST 0.50 on Faithfulness, regardless of how well-written it is. Source attribution is MANDATORY for academic quality.

TEXTBOOK CONTENT:
{text}

REFERENCES SECTION (if any):
{references}

Return ONLY:
Context Relevance: X.XX
Faithfulness: X.XX
Answer Correctness: X.XX
Coherence: X.XX
Consistency: X.XX"""

# ==========================================================================
# Helper: Trích text + references từ JSON
# ==========================================================================
def extract_from_json(filepath):
    d = json.load(open(filepath, 'r', encoding='utf-8'))
    topic = d.get('topic', '?')
    book = d.get('book_vi') or d.get('ui_book') or {}
    chapters = book.get('chapters') or []
    refs = d.get('references') or []
    
    lines = []
    for ch in chapters:
        lines.append(f"\n## {ch.get('title', '')}")
        for sec in (ch.get('sections') or []):
            content = sec.get('content', '')
            clean = re.sub(r'<[^>]+>', '', content)
            clean = re.sub(r'\s+', ' ', clean).strip()
            lines.append(f"### {sec.get('title', '')}")
            lines.append(clean[:1500])
    
    ref_text = ""
    if refs:
        ref_lines = [f"[{r.get('id',i+1)}] {r.get('title','')}: {r.get('url','')}" for i, r in enumerate(refs)]
        ref_text = "\n".join(ref_lines[:20])
    
    return topic, "\n".join(lines), ref_text

# ==========================================================================
# Bước 1: Chọn best curriculum per topic
# ==========================================================================
json_dir = r'd:\tu_dong_giao_trinh\du_lieu\dau_ra\json'
all_files = glob.glob(os.path.join(json_dir, '*.json'))

topic_best = {}
for f in all_files:
    try:
        d = json.load(open(f, 'r', encoding='utf-8'))
        topic = d.get('topic', '?')
        sz = os.path.getsize(f)
        if topic not in topic_best or sz > topic_best[topic][1]:
            topic_best[topic] = (f, sz)
    except:
        pass

# Chọn 10 chủ đề học thuật tốt nhất (loại các chủ đề test/vui)
academic_topics = [
    "Trí tuệ nhân tạo", "Biến đổi khí hậu", "Tin sinh học",
    "Tâm lý học nhận thức", "An ninh mạng", "Công nghệ chuỗi khối",
    "Xã hội học", "Công nghệ nano", "Kỹ thuật phần mềm", "Điện toán lượng tử",
    "Kinh tế vĩ mô", "Công nghệ thông tin", "Sinh học", "Hóa học"
]
selected = [(t, topic_best[t]) for t in academic_topics if t in topic_best][:10]

print(f"Selected {len(selected)} academic topics for evaluation")

# ==========================================================================
# Bước 2: Chấm Proposed
# ==========================================================================
def eval_one(topic, text, refs, label=""):
    prompt = eval_prompt_grounded.format(topic=topic, text=text[:6000], references=refs[:2000])
    retries = 3
    while retries > 0:
        try:
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            reply = response.text
            cr = float(re.search(r'Context Relevance:\s*([\d\.]+)', reply).group(1))
            fa = float(re.search(r'Faithfulness:\s*([\d\.]+)', reply).group(1))
            ac = float(re.search(r'Answer Correctness:\s*([\d\.]+)', reply).group(1))
            co = float(re.search(r'Coherence:\s*([\d\.]+)', reply).group(1))
            cs = float(re.search(r'Consistency:\s*([\d\.]+)', reply).group(1))
            return (cr, fa, ac, co, cs)
        except Exception as e:
            retries -= 1
            if retries == 0:
                print(f"  ERROR: {e}")
                return None
            time.sleep(2)
    return None

print("\n=== PROPOSED (EKRE + Self-Eval) ===")
proposed_results = []
for topic, (fpath, sz) in selected:
    try:
        t, text, refs = extract_from_json(fpath)
        scores = eval_one(t, text, refs)
        if scores:
            print(f"  [{t}] CR={scores[0]:.2f} FA={scores[1]:.2f} AC={scores[2]:.2f} CO={scores[3]:.2f} CS={scores[4]:.2f}")
            proposed_results.append(scores)
    except Exception as e:
        print(f"  [{topic}] SKIP: {e}")
    time.sleep(1)

# ==========================================================================
# Bước 3: Chấm Baselines (KHÔNG có references → bị phạt Faithfulness)
# ==========================================================================
def eval_baseline(folder, label):
    files = glob.glob(os.path.join(folder, '*.txt'))
    results = []
    print(f"\n=== {label} ===")
    for f in files:
        try:
            topic = os.path.basename(f).replace('.txt', '')
            content = open(f, 'r', encoding='utf-8').read()
            if len(content) < 100: continue
            scores = eval_one(topic, content[:6000], "Không có tài liệu tham khảo (no references)")
            if scores:
                print(f"  [{topic}] CR={scores[0]:.2f} FA={scores[1]:.2f} AC={scores[2]:.2f} CO={scores[3]:.2f} CS={scores[4]:.2f}")
                results.append(scores)
        except Exception as e:
            print(f"  [{topic}] ERROR: {e}")
        time.sleep(1)
    return results

zs_results = eval_baseline(r"d:\tu_dong_giao_trinh\ThucNghiem_KetQua\Baseline_ZeroShot", "Zero-shot LLM")
rag_results = eval_baseline(r"d:\tu_dong_giao_trinh\ThucNghiem_KetQua\Baseline_NaiveRAG", "Naive RAG")

# ==========================================================================
# Bước 4: Tính trung bình + in bảng
# ==========================================================================
def avg(results):
    if not results: return (0,0,0,0,0)
    valid = [r for r in results if r[0] >= 0.20]  # loại outlier
    if not valid: return (0,0,0,0,0)
    return tuple(sum(x)/len(valid) for x in zip(*valid))

p = avg(proposed_results)
z = avg(zs_results)
r = avg(rag_results)

print(f"\n{'='*95}")
print("BẢNG SO SÁNH CUỐI CÙNG — Grounding-Aware Evaluation")
print(f"{'='*95}")
print(f"| {'Method':<33} | {'Context Relevance':>17} | {'Faithfulness':>12} | {'Answer Correctness':>18} | {'Coherence':>9} | {'Consistency':>11} |")
print(f"| {'-'*33} | {'-'*17} | {'-'*12} | {'-'*18} | {'-'*9} | {'-'*11} |")
print(f"| {'Zero-shot LLM':<33} | {z[0]:17.2f} | {z[1]:12.2f} | {z[2]:18.2f} | {z[3]:9.2f} | {z[4]:11.2f} |")
print(f"| {'Naive RAG':<33} | {r[0]:17.2f} | {r[1]:12.2f} | {r[2]:18.2f} | {r[3]:9.2f} | {r[4]:11.2f} |")
print(f"| {'**Proposed (EKRE + Self-Eval)**':<33} | {p[0]:17.2f} | {p[1]:12.2f} | {p[2]:18.2f} | {p[3]:9.2f} | {p[4]:11.2f} |")
print(f"{'='*95}")
