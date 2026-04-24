# Hướng dẫn Cài đặt và Sử dụng

## 1. Yêu cầu hệ thống
- Python 3.8 trở lên
- Đã cài đặt `pip`

## 2. Cài đặt

1.  Clone hoặc tải code về máy.
2.  Mở terminal tại thư mục dự án.
3.  Tạo môi trường ảo (khuyến nghị):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```
4.  Cài đặt các thư viện phụ thuộc:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Cấu hình

1.  Tạo file `.env` từ file `.env.example` (nếu có) hoặc tạo mới với nội dung sau:
    ```env
    FLASK_SECRET_KEY=khoa-bi-mat-cua-ban
    GEMINI_API_KEY=AIzaSy... (API Key của bạn)
    GEMINI_MODEL=gemini-2.5-flash
    
    # Cấu hình crawling (tùy chọn)
    SO_TRANG_HAT_GIONG=6
    SO_TRANG_LIEN_KET=20
    SO_DOAN_GUI_GEMINI=120
    ```
2.  **Lưu ý quan trọng**: Bạn cần có API Key từ Google AI Studio (Gemini).

## 4. Chạy ứng dụng

1.  Chạy lệnh:
    ```bash
    python ung_dung.py
    ```
2.  Mở trình duyệt truy cập: `http://localhost:5000`

## 5. Sử dụng

1.  Nhập **Tiêu đề giáo trình** (ví dụ: "Nhập môn Trí tuệ Nhân tạo").
2.  Điều chỉnh tham số (số đoạn, số liên kết) nếu cần.
3.  Nhấn **Tạo giáo trình**.
4.  Chờ hệ thống xử lý (có thể mất 1-3 phút tùy mạng và dữ liệu).
5.  Sau khi hoàn thành, bạn có thể:
    -   Xem Online (HTML/JSON)
    -   Tải về PDF
    -   Tải về Word (DOCX)
