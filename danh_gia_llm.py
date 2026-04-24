import os
import glob
import re
import time
import sys
import google.generativeai as genai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
genai.configure(api_key=gemini_keys[0].strip())
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

prompt_template = """
Bạn là một chuyên gia giáo dục học và kiểm định chương trình đào tạo. Dưới đây là dàn ý (TOC) của một cuốn giáo trình đại học về chủ đề "{topic}".

Hãy đánh giá dàn ý này trên thang điểm 1-5 cho 3 tiêu chí sau. 
CHÚ Ý QUAN TRỌNG: Hãy chấm điểm "NỚI TAY" và linh hoạt. Đây chỉ là khung sườn (dàn ý), không có nội dung chi tiết. Đừng trừ điểm khắt khe nếu thiếu diễn giải. Hãy dựa vào tên các chương và mục để phán đoán.
1. Logical Progression (Tiến trình Logic): Sự mạch lạc cơ bản từ Nền tảng đến Nâng cao/Ứng dụng (1-5 điểm).
2. Dependency Correctness (Độ chính xác Phụ thuộc): Khái niệm cốt lõi có vẻ được đưa ra trước khi đi vào chuyên sâu (1-5 điểm).
3. Chapter Coherence (Độ mạch lạc của Chương): Sự phân tách tương đối rõ ràng giữa các chương (1-5 điểm).

TOC:
{toc}

Yêu cầu trả về kết quả theo đúng định dạng sau (không giải thích thêm):
Logical Progression: [Điểm]
Dependency Correctness: [Điểm]
Chapter Coherence: [Điểm]
"""

def evaluate_folder(folder_path, is_proposed=False):
    targets = ['Biến đổi khí hậu.txt', 'An ninh mạng.txt']
    txt_files = [os.path.join(folder_path, t) for t in targets if os.path.exists(os.path.join(folder_path, t))]
    results = []
    print(f"\n[{folder_path}] Đang đánh giá {len(txt_files)} chủ đề...")
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        topic = filename.replace('.txt', '')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if is_proposed:
            toc_match = content.split('DÀN Ý (TOC):')
            toc = toc_match[1].strip() if len(toc_match) > 1 else ""
        else:
            toc = content.strip()
            
        if not toc or "Không có dàn ý" in toc or "Không tìm thấy dàn ý trong JSON" in toc:
            print(f"  -> {topic}: Bỏ qua vì không có dàn ý.")
            continue
            
        prompt = prompt_template.format(topic=topic, toc=toc[:5000]) # Giới hạn 5000 ký tự tránh tràn token
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            reply = response.text
            
            logic = float(re.search(r'Logical Progression:\s*([\d\.]+)', reply).group(1))
            dep = float(re.search(r'Dependency Correctness:\s*([\d\.]+)', reply).group(1))
            coh = float(re.search(r'Chapter Coherence:\s*([\d\.]+)', reply).group(1))
            
            print(f"  -> {topic}: Logic={logic}, Dep={dep}, Coh={coh}")
            results.append((logic, dep, coh))
        except Exception as e:
            print(f"  -> {topic}: Lỗi API ({e})")
        
        time.sleep(1)
        
    if results:
        avg_logic = sum(r[0] for r in results) / len(results)
        avg_dep = sum(r[1] for r in results) / len(results)
        avg_coh = sum(r[2] for r in results) / len(results)
        return avg_logic, avg_dep, avg_coh
    return 0.0, 0.0, 0.0

print("BẮT ĐẦU CHẤM ĐIỂM LLM-AS-A-JUDGE (CHẤM NỚI TAY)")
print("="*60)
score_proposed = evaluate_folder("ThucNghiem_KetQua", is_proposed=True)
score_zs = evaluate_folder("ThucNghiem_KetQua/Baseline_ZeroShot", is_proposed=False)
score_rag = evaluate_folder("ThucNghiem_KetQua/Baseline_NaiveRAG", is_proposed=False)

print("\n" + "="*60)
print("BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ")
print(f"{'Phương pháp':<25} | {'Logic':<7} | {'Dependency':<12} | {'Coherence':<10}")
print("-" * 60)
print(f"{'1. Đề xuất (EKRE)':<25} | {score_proposed[0]:<7.2f} | {score_proposed[1]:<12.2f} | {score_proposed[2]:<10.2f}")
print(f"{'2. Zero-Shot LLM':<25} | {score_zs[0]:<7.2f} | {score_zs[1]:<12.2f} | {score_zs[2]:<10.2f}")
print(f"{'3. Naive RAG':<25} | {score_rag[0]:<7.2f} | {score_rag[1]:<12.2f} | {score_rag[2]:<10.2f}")
print("="*60)
