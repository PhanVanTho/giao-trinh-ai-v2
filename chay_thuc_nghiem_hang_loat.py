import requests
import time
import json
import os

# =========================================================================
# CẤU HÌNH THỰC NGHIÊM
# =========================================================================
API_URL = "http://127.0.0.1:5000" # Đảm bảo server Flask của bạn đang chạy
THU_MUC_JSON_GOC = "du_lieu/dau_ra/json"   # Thư mục hệ thống xuất file JSON
THU_MUC_THUC_NGHIEM = "ThucNghiem_KetQua"

# Danh sách 10 chủ đề bạn muốn test (Hãy thay đổi theo ý muốn)
DANH_SACH_CHU_DE = [
    "Biến đổi khí hậu",
    "An ninh mạng"
    
]

if not os.path.exists(THU_MUC_THUC_NGHIEM):
    os.makedirs(THU_MUC_THUC_NGHIEM)

# =========================================================================
# HÀM XỬ LÝ CHÍNH
# =========================================================================
def rut_trich_dan_y(ma_cv):
    """Đọc file JSON đầu ra của hệ thống để rút trích Dàn ý (TOC)"""
    file_path = os.path.join(THU_MUC_JSON_GOC, f"{ma_cv}.json")
    if not os.path.exists(file_path):
        return "Không tìm thấy file JSON để lấy dàn ý."
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        toc = []
        for chap in data.get("book_vi", {}).get("chapters", []):
            toc.append(f"Chương: {chap.get('title')}")
            for sec in chap.get("sections", []):
                toc.append(f"  - {sec.get('title')}")
        
        if not toc:
            return "Không tìm thấy cấu trúc 'chapters' bên trong file JSON."
            
        return "\n".join(toc)
    except Exception as e:
        return f"Lỗi đọc dàn ý: {str(e)}"

def chay_thuc_nghiem():
    tong_so = len(DANH_SACH_CHU_DE)
    thanh_cong = 0
    danh_sach_kqs = []
    
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM ({tong_so} CHỦ ĐỀ)")
    print("-" * 50)
    
    for idx, chu_de in enumerate(DANH_SACH_CHU_DE, 1):
        print(f"\n[{idx}/{tong_so}] Đang chạy chủ đề: '{chu_de}'...")
        
        # 1. Gửi request tạo giáo trình
        try:
            res = requests.post(f"{API_URL}/tao", json={
                "tieu_de": chu_de,
                "quy_mo": "chuyen_sau", # Chạy bản chuyên sâu như trong báo cáo
                "ngon_ngu": "vi"
            })
            res_data = res.json()
            ma_cv = res_data.get("ma_cv")
            
            if not ma_cv:
                print(f"❌ Lỗi khởi tạo: {res_data}")
                continue
                
            print(f"   + Job ID: {ma_cv[:8]}... Đang chờ xử lý (có thể mất 5-10 phút)")
            
        except Exception as e:
            print(f"❌ Lỗi kết nối đến server: {e}")
            continue

        # 2. Polling chờ kết quả (Tối đa 15 phút)
        MAX_WAIT_SECONDS = 900  # 15 phút timeout
        start_time = time.time()
        while True:
            try:
                # Watchdog: Bỏ qua nếu chờ quá lâu
                elapsed = time.time() - start_time
                if elapsed > MAX_WAIT_SECONDS:
                    print(f"   ⏰ TIMEOUT sau {elapsed:.0f}s! Bỏ qua chủ đề này và chuyển sang chủ đề tiếp theo.")
                    break

                status_res = requests.get(f"{API_URL}/trang_thai/{ma_cv}")
                status_data = status_res.json()
                trang_thai = status_data.get("trang_thai")
                
                if trang_thai == "hoan_thanh":
                    thanh_cong += 1
                    elapsed = time.time() - start_time
                    
                    # Lấy KQS (Trung bình cộng của Quality Score trong top 30 links)
                    links = status_data.get("top_30_links", [])
                    if links:
                        kqs = sum(l.get("quality_score", 0) for l in links) / len(links)
                        danh_sach_kqs.append(kqs)
                    else:
                        kqs = 0.0
                    
                    # Rút trích Dàn ý (TOC)
                    toc_text = rut_trich_dan_y(ma_cv)
                    
                    # Lưu file kết quả cho chủ đề này
                    safe_name = "".join([c for c in chu_de if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                    out_file = os.path.join(THU_MUC_THUC_NGHIEM, f"{safe_name}.txt")
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(f"CHỦ ĐỀ: {chu_de}\n")
                        f.write(f"KQS: {kqs:.3f}\n")
                        f.write(f"THỜI GIAN: {elapsed:.1f}s\n")
                        f.write(f"{'='*30}\n")
                        f.write("DÀN Ý (TOC):\n")
                        f.write(toc_text)
                    
                    print(f"   ✅ Hoàn thành! Thời gian: {elapsed:.1f}s | KQS: {kqs:.3f}")
                    print(f"   💾 Đã lưu kết quả tại: {out_file}")
                    break
                    
                elif trang_thai == "that_bai":
                    loi = status_data.get("loi", "Không rõ lỗi")
                    print(f"   ❌ Thất bại! Lỗi: {loi}")
                    break
                    
                # Chờ 15 giây rồi gọi lại (giảm tải polling)
                time.sleep(15)
                
            except requests.exceptions.RequestException:
                print("   ⚠️ Mất kết nối đến server, thử lại sau 5s...")
                time.sleep(5)
        
        # Nghỉ 2 phút (120s) giữa các chủ đề để API phục hồi (trừ chủ đề cuối cùng)
        if idx < tong_so:
            print(f"⏳ Đang nghỉ 2 phút để làm mát hệ thống/API...")
            time.sleep(120)
                
    # =========================================================================
    # TỔNG KẾT
    # =========================================================================
    psr = (thanh_cong / tong_so) * 100 if tong_so > 0 else 0
    avg_kqs = sum(danh_sach_kqs) / len(danh_sach_kqs) if danh_sach_kqs else 0
    
    print("\n" + "=" * 50)
    print("📊 TỔNG KẾT THỰC NGHIỆM")
    print("=" * 50)
    print(f"Tổng số chủ đề: {tong_so}")
    print(f"Số lần thành công: {thanh_cong}")
    print(f"Tỷ lệ hoàn thành (PSR): {psr:.1f}%")
    print(f"Điểm chất lượng tri thức trung bình (KQS): {avg_kqs:.3f}")
    print("=" * 50)
    print(f"Tất cả file Dàn ý đã được lưu trong thư mục: '{THU_MUC_THUC_NGHIEM}'")
    print("Bây giờ bạn có thể lấy các file Dàn ý này để cho GPT-4o chấm điểm (LLM-as-a-Judge)!")

if __name__ == "__main__":
    chay_thuc_nghiem()
