import sys
sys.stdout.reconfigure(encoding='utf-8')

# Kết quả RAW từ chạy thực tế (LLM-as-a-Judge)
proposed_raw = {
    "Trí tuệ nhân tạo":      (0.95, 0.90, 0.75, 0.80, 0.85),
    "Biến đổi khí hậu":      (0.90, 0.85, 0.60, 0.55, 0.70),
    "Tin sinh học":           (0.75, 0.80, 0.65, 0.60, 0.70),
    "Tâm lý học nhận thức":  (0.60, 0.85, 0.40, 0.50, 0.65),
    "An ninh mạng":           (0.40, 0.70, 0.30, 0.40, 0.50),
    "Công nghệ chuỗi khối":  (0.10, 0.40, 0.05, 0.20, 0.20),  # OUTLIER
    "Xã hội học":             (0.95, 0.90, 0.75, 0.70, 0.85),
    "Công nghệ nano":         (0.85, 0.75, 0.60, 0.50, 0.65),
    "Kỹ thuật phần mềm":     (0.95, 0.90, 0.85, 0.80, 0.90),
    "Điện toán lượng tử":     (0.95, 0.90, 0.75, 0.80, 0.85),
}

zeroshot_raw = {
    "An ninh mạng":           (1.00, 0.95, 0.90, 0.95, 0.95),
    "Biến đổi khí hậu":      (1.00, 0.95, 0.95, 0.95, 0.90),
    "Công nghệ chuỗi khối":  (1.00, 0.95, 0.90, 0.95, 0.95),
    "Công nghệ nano":         (1.00, 0.95, 0.95, 0.95, 0.95),
    "Kinh tế vĩ mô":         (1.00, 0.95, 0.90, 0.95, 0.95),
    "Tin sinh học":           (1.00, 0.95, 0.90, 0.95, 0.95),
    "Trí tuệ nhân tạo":      (1.00, 0.95, 0.90, 0.90, 0.85),
    "Tâm lý học nhận thức":  (1.00, 0.95, 0.90, 0.95, 0.95),
    "Xã hội học":             (1.00, 0.95, 0.90, 0.95, 0.95),
    "Điện toán lượng tử":     (1.00, 0.95, 0.90, 0.95, 0.90),
}

naive_rag_raw = {
    "An ninh mạng":           (0.95, 0.85, 0.75, 0.80, 0.70),
    "Biến đổi khí hậu":      (1.00, 0.95, 0.85, 0.75, 0.80),
    "Công nghệ chuỗi khối":  (0.00, 0.00, 0.00, 0.00, 0.00),  # OUTLIER
    "Công nghệ nano":         (1.00, 0.95, 0.95, 0.95, 0.95),
    "Kinh tế vĩ mô":         (0.00, 0.00, 0.00, 0.00, 0.00),  # OUTLIER
    "Tin sinh học":           (1.00, 0.95, 0.90, 0.95, 0.90),
    "Trí tuệ nhân tạo":      (1.00, 0.95, 0.85, 0.80, 0.85),
    "Tâm lý học nhận thức":  (0.95, 0.85, 0.75, 0.80, 0.85),
    "Xã hội học":             (1.00, 0.95, 0.90, 0.85, 0.90),
    "Điện toán lượng tử":     (0.95, 0.85, 0.75, 0.80, 0.85),
}

# Loại outlier (bài có CR < 0.30 cho Proposed hoặc CR=0 cho baselines)
def avg_scores(data, min_cr=0.30):
    valid = [v for v in data.values() if v[0] >= min_cr]
    if not valid: return (0,0,0,0,0)
    return tuple(sum(x)/len(valid) for x in zip(*valid))

proposed_avg = avg_scores(proposed_raw, min_cr=0.30)
zeroshot_avg = avg_scores(zeroshot_raw, min_cr=0.30)
naive_avg = avg_scores(naive_rag_raw, min_cr=0.30)

print("=" * 95)
print("KẾT QUẢ THỰC TẾ TỪ HỆ THỐNG (LLM-as-a-Judge, loại outlier)")
print("=" * 95)
print(f"| {'Method':<33} | {'CR':>5} | {'FA':>5} | {'AC':>5} | {'CO':>5} | {'CS':>5} |")
print(f"|{'-'*34}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|")
print(f"| {'Zero-shot LLM':<33} | {zeroshot_avg[0]:5.2f} | {zeroshot_avg[1]:5.2f} | {zeroshot_avg[2]:5.2f} | {zeroshot_avg[3]:5.2f} | {zeroshot_avg[4]:5.2f} |")
print(f"| {'Naive RAG':<33} | {naive_avg[0]:5.2f} | {naive_avg[1]:5.2f} | {naive_avg[2]:5.2f} | {naive_avg[3]:5.2f} | {naive_avg[4]:5.2f} |")
print(f"| {'Proposed (EKRE + Self-Eval)':<33} | {proposed_avg[0]:5.2f} | {proposed_avg[1]:5.2f} | {proposed_avg[2]:5.2f} | {proposed_avg[3]:5.2f} | {proposed_avg[4]:5.2f} |")
print("=" * 95)

# Kết quả chi tiết
print("\n--- Chi tiết Proposed (loại Công nghệ chuỗi khối) ---")
for topic, scores in proposed_raw.items():
    if scores[0] >= 0.30:
        status = "✓"
    else:
        status = "✗ OUTLIER"
    print(f"  {status} {topic}: CR={scores[0]:.2f} FA={scores[1]:.2f} AC={scores[2]:.2f} CO={scores[3]:.2f} CS={scores[4]:.2f}")
