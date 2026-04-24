from cau_hinh import CauHinh

def loc_lien_ket_bang_ai(danh_sach_link, chu_de, top_k=20, model=CauHinh.OPENAI_MODEL, api_key=None):
    """
    Sử dụng AI để chọn ra các liên kết đáng giá nhất từ danh sách hàng trăm link.
    """
    if not danh_sach_link:
        return []
    
    # Chỉ lấy tối đa 200 link để filter tránh token limit
    candidates = list(set(danh_sach_link))[:300]
    
    prompt = f"""
    Bạn là trợ lý soạn giáo trình đại học.
    Chủ đề chính: "{chu_de}"
    
    Dưới đây là danh sách các thuật ngữ/liên kết từ Wikipedia:
    {json.dumps(candidates, ensure_ascii=False)}
    
    Nhiệm vụ: Hãy chọn ra đúng {top_k} liên kết QUAN TRỌNG NHẤT, LIÊN QUAN TRỰC TIẾP để bổ sung kiến thức sâu cho chủ đề này.
    Loại bỏ các liên kết rác, quá chung chung (như "Năm 2020", "Hoa Kỳ", "Tiếng Anh") hoặc không liên quan.
    
    Trả về: JSON Array chứa danh sách các chuỗi (string) đã chọn.
    Ví dụ: ["Khái niệm A", "Thuật toán B", ...]
    """

    try:
        from openai import OpenAI
        if not api_key:
            return []
        
        client = OpenAI(api_key=api_key, max_retries=0)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia lọc dữ liệu giáo dục. Chỉ trả về JSON thuần."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        
        # Hỗ trợ nhiều định dạng trả về key
        results = data.get("links") or data.get("results") or data.get("selected") or list(data.values())[0]
        
        if isinstance(results, list):
            return results[:top_k]
        return []
        
    except Exception as e:
        print(f"[WARN] Lỗi lọc linked pages bằng AI: {e}")
        # Fallback: Trả về 20 thằng đầu tiên nếu lỗi/không dùng AI
        return candidates[:top_k]
