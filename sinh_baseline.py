import os
import requests
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
client = OpenAI()

THU_MUC_ZERO_SHOT = "ThucNghiem_KetQua/Baseline_ZeroShot"
THU_MUC_NAIVE_RAG = "ThucNghiem_KetQua/Baseline_NaiveRAG"

os.makedirs(THU_MUC_ZERO_SHOT, exist_ok=True)
os.makedirs(THU_MUC_NAIVE_RAG, exist_ok=True)

DANH_SACH_CHU_DE = [
    "Trí tuệ nhân tạo",
    "Công nghệ nano",
    "Điện toán lượng tử",
    "An ninh mạng",
    "Kinh tế vĩ mô",
    "Xã hội học",
    "Tâm lý học nhận thức",
    "Biến đổi khí hậu",
    "Công nghệ chuỗi khối",
    "Tin sinh học"
]

def fetch_naive_wikipedia(topic):
    """Lấy trực tiếp nội dung bài Wikipedia tiếng Việt đầu tiên (Naive Search)"""
    # Bước 1: Tìm kiếm opensearch để lấy title chính xác
    search_url = "https://vi.wikipedia.org/w/api.php"
    search_params = {
        "action": "opensearch",
        "search": topic,
        "limit": 1,
        "namespace": 0,
        "format": "json"
    }
    headers = {
        "User-Agent": "TuDongGiaoTrinh/1.0 (test@example.com)"
    }
    try:
        search_res = requests.get(search_url, params=search_params, headers=headers).json()
        if len(search_res) > 1 and search_res[1]:
            exact_title = search_res[1][0]
        else:
            return ""
            
        # Bước 2: Lấy nội dung extract
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": exact_title,
            "format": "json"
        }
        res = requests.get(search_url, params=extract_params, headers=headers).json()
        pages = res["query"]["pages"]
        for page_id in pages:
            if page_id == "-1": return ""
            text = pages[page_id].get("extract", "")
            return text[:15000] # Naive chunking
    except Exception as e:
        print(f"Lỗi tải Wiki: {e}")
        return ""
    return ""

def call_openai_with_retry(prompt, max_retries=5):
    """Hàm gọi API OpenAI có cơ chế tự động thử lại khi gặp lỗi 502/429"""
    delay = 5
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"      [Lỗi OpenAI - Thử lại lần {attempt+1}/{max_retries} sau {delay}s]: {e}")
            time.sleep(delay)
            delay *= 2 # Exponential backoff (5s -> 10s -> 20s)
    return "Lỗi gọi API OpenAI quá nhiều lần."

def sinh_baseline():
    print("🚀 BẮT ĐẦU SINH DỮ LIỆU BASELINE (ZERO-SHOT & NAIVE RAG)")
    print("-" * 50)
    for idx, chu_de in enumerate(DANH_SACH_CHU_DE, 1):
        print(f"\n[{idx}/10] Đang tạo cho chủ đề: '{chu_de}'")
        safe_name = "".join([c for c in chu_de if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        
        # ==============================================================
        # 1. Zero-Shot Baseline
        # ==============================================================
        file_zs = os.path.join(THU_MUC_ZERO_SHOT, f"{safe_name}.txt")
        if not os.path.exists(file_zs):
            print("   -> Sinh Dàn ý Zero-shot...")
            prompt_zs = f"Hãy viết một dàn ý giáo trình đại học chuyên sâu cho chủ đề '{chu_de}'. Trình bày dưới dạng danh sách các chương và mục, tập trung vào cấu trúc học thuật."
            content = call_openai_with_retry(prompt_zs)
            with open(file_zs, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print("   -> Dàn ý Zero-shot đã tồn tại.")
                
        # ==============================================================
        # 2. Naive RAG Baseline
        # ==============================================================
        file_rag = os.path.join(THU_MUC_NAIVE_RAG, f"{safe_name}.txt")
        if not os.path.exists(file_rag):
            print("   -> Sinh Dàn ý Naive RAG...")
            wiki_text = fetch_naive_wikipedia(chu_de)
            if not wiki_text:
                print("      ⚠️ Không tìm thấy bài Wiki trực tiếp, RAG sẽ sinh với text trống.")
            
            prompt_rag = f"Bạn là hệ thống RAG cơ bản. Dựa DUY NHẤT vào văn bản (text) sau đây, hãy tạo một dàn ý giáo trình đại học chuyên sâu cho chủ đề '{chu_de}'. Không tự bịa thêm kiến thức ngoài văn bản.\n\nVĂN BẢN:\n{wiki_text}"
            content = call_openai_with_retry(prompt_rag)
            with open(file_rag, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print("   -> Dàn ý Naive RAG đã tồn tại.")

if __name__ == "__main__":
    sinh_baseline()
    print("\n✅ Đã sinh xong toàn bộ dữ liệu Baseline! Các file nằm trong thư mục ThucNghiem_KetQua")
