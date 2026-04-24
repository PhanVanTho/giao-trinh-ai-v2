# -*- coding: utf-8 -*-
import os

class CauHinh:
    # Flask
    KHOA_BI_MAT = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")



    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Gemini
    GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()]
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")           # Model mạnh — dùng cho việc cần độ chính xác cao
    GEMINI_MODEL_LITE = os.getenv("GEMINI_MODEL_LITE", "gemini-3.1-flash-lite-preview")  # Model nhẹ — dùng cho hầu hết Supervisor tasks

    # Supervisor Architecture Config
    WRITER_MODEL = os.getenv("WRITER_MODEL", "gpt-4o-mini")  # Mặc định dùng OpenAI
    SUPERVISOR_MODEL_LITE = os.getenv("SUPERVISOR_MODEL_LITE", "gemini-3.1-flash-lite-preview")
    SUPERVISOR_MODEL_PRO = os.getenv("SUPERVISOR_MODEL_PRO", "gemini-2.5-flash")

    GEMINI_AUDIT_RATE = float(os.getenv("GEMINI_AUDIT_RATE", "1.0"))       # 100% audit (vì dùng GPT-4o-mini làm writer)
    MAX_REWRITE_ATTEMPTS = int(os.getenv("MAX_REWRITE_ATTEMPTS", "2"))     # Số lần rewrite tối đa / chương
    SUPERVISOR_OUTLINE_CHECK = os.getenv("SUPERVISOR_OUTLINE_CHECK", "true").lower() == "true"  # Có kiểm tra outline không



    # Wikipedia crawling limits
    SO_TRANG_HAT_GIONG = int(os.getenv("SO_TRANG_HAT_GIONG", "10"))       # seed pages (tổng vi+en)
    SO_TRANG_LIEN_KET = int(os.getenv("SO_TRANG_LIEN_KET", "30"))        # linked pages (depth=1)
    SO_DOAN_THAM_KHAO = int(os.getenv("SO_DOAN_THAM_KHAO", "120"))      # top passages gửi cho AI (token-saving)

    # Passage chunking
    DO_DAI_DOAN_MIN = int(os.getenv("DO_DAI_DOAN_MIN", "800"))
    DO_DAI_DOAN_MAX = int(os.getenv("DO_DAI_DOAN_MAX", "1500"))
    CAT_TREN_MOI_TRANG = int(os.getenv("CAT_TREN_MOI_TRANG", "12000"))   # cắt tối đa ký tự mỗi trang

    # Output folders
    THU_MUC_DU_LIEU = os.getenv("THU_MUC_DU_LIEU", "du_lieu")
    THU_MUC_CACHE = os.path.join(THU_MUC_DU_LIEU, "bo_nho_dem")
    THU_MUC_DAU_RA = os.path.join(THU_MUC_DU_LIEU, "dau_ra")

    THU_MUC_JSON = os.path.join(THU_MUC_DAU_RA, "json")
    THU_MUC_PDF = os.path.join(THU_MUC_DAU_RA, "pdf")
    THU_MUC_DOCX = os.path.join(THU_MUC_DAU_RA, "docx")

    # UI defaults (Relaxed for dynamic generation)
    MAC_DINH_SO_CHUONG_MIN = int(os.getenv("MAC_DINH_SO_CHUONG_MIN", "3"))
    MAC_DINH_SO_CHUONG_MAX = int(os.getenv("MAC_DINH_SO_CHUONG_MAX", "25"))

    # Production timeouts (V23.1)
    API_TIMEOUT = 60.0
    CHAPTER_TIMEOUT = 300.0
    JOB_TIMEOUT = 1200.0  # 20 minutes hard watchdog

    # =========================================================================
    # EKRE V26.2 - ADAPTIVE THRESHOLD & SAFE DEGRADATION CONFIG
    # =========================================================================

    # --- Hard Floors (tuyệt đối không xuống thấp hơn) ---
    EKRE_MIN_SIM_FLOOR = 0.30       # Ngưỡng tương đồng tối thiểu tuyệt đối
    EKRE_MIN_QUALITY_FLOOR = 0.5     # V26.2.1: Recalibrated cho formula (sim^2)*log(len)

    # --- Adaptive Similarity Floors (theo Complexity) ---
    EKRE_SIM_FLOORS = {
        "high":   0.35,   # Chủ đề khó: Bắt đầu từ 0.35, có thể hạ xuống 0.30
        "medium": 0.40,   # Chủ đề trung bình: Bắt đầu từ 0.40
        "low":    0.45,   # Chủ đề phổ thông: Bắt đầu từ 0.45
    }

    # --- Adaptive Noise Brake (MIN_AVG_SIM theo Complexity) ---
    EKRE_MIN_AVG_SIM = {
        "high":   0.30,   # Topic khó → chấp nhận avg_sim thấp hơn
        "medium": 0.32,   # Topic bình thường
        "low":    0.34,   # Topic dễ → yêu cầu avg_sim cao hơn
    }

    # --- Adaptive Loop Config ---
    EKRE_MAX_RELAXATION_ATTEMPTS = 4   # Số lần thử giãn chuẩn tối đa
    EKRE_LOW_RATIO_BRAKE = 0.50        # Dừng nếu >50% docs là low_priority

    # --- Standard Quality Thresholds (Reweighted: sim^2 * log(len)) ---
    # Output range: ~0.5 (poor) → ~2.0 (ok) → ~5.0 (great) → ~10 (excellent)
    EKRE_QUALITY_STANDARD = 1.5        # Ngưỡng tiêu chuẩn (đã hiệu chỉnh V26.2.1)
    EKRE_QUALITY_RESCUE    = 0.8       # Ngưỡng cứu hộ (Rescue Mode)

    # --- Target Yield theo Quy mô ---
    EKRE_TARGET_YIELD = {
        "can_ban":   8,    # Cơ bản: 8 docs
        "tieu_chuan": 15,  # Tiêu chuẩn: 15 docs
        "chuyen_sau": 25,  # Chuyên sâu: 25 docs
    }

    # --- Diversity Control ---
    EKRE_MAX_CHUNKS_PER_SOURCE = 3  # Tối đa 3 chunks từ cùng 1 nguồn
