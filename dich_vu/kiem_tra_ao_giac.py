
def kiem_tra_ao_giac(
    chu_de: str,
    chapter_content: dict,
    relevant_passages: list,
    model: str = "gpt-4o-mini",
    api_key: str = None
):
    """
    Bước 3 (Optional): Kiểm tra ảo giác (Hallucination Check).
    Nhận nội dung chương vừa tạo và corpus gốc, yêu cầu AI đối chiếu.
    Nếu phát hiện bịa đặt, AI phải sửa lại cho khớp với corpus.
    """
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY")

    # Disable SDK retry (Hotfix V5.2)
    client = OpenAI(api_key=api_key, max_retries=0)

    packed_corpus = []
    for p in relevant_passages:
        packed_corpus.append({
            "url": p.get("url"),
            "text": p.get("text")
        })

    prompt = f"""
Bạn là một CHUYÊN GIA KIỂM ĐỊNH CHẤT LƯỢNG HỌC THUẬT (Fact-checker).
Nhiệm vụ của bạn là kiểm tra xem nội dung văn bản dưới đây có bị "ảo giác" (hallucination - bịa đặt thông tin không có trong nguồn) hay không.

CHỦ ĐỀ: "{chu_de}"
CHƯƠNG: "{chapter_content.get('title')}"

NỘI DUNG CẦN KIỂM TRA (Do AI cấp dưới vừa viết):
{json.dumps(chapter_content.get('sections', []), ensure_ascii=False)}

NGUỒN DỮ LIỆU GỐC (CORPUS CHÍNH XÁC):
{json.dumps(packed_corpus, ensure_ascii=False)}

YÊU CẦU KIỂM TRA & CHỈNH SỬA:
1. Đọc kỹ từng câu trong [NỘI DUNG CẦN KIỂM TRA].
2. Đối chiếu CHẶT CHẼ với [NGUỒN DỮ LIỆU GỐC].
3. Nếu phát hiện bất kỳ thông tin nào, số liệu nào, hoặc tên riêng nào CÓ TRONG BÀI VIẾT nhưng KHÔNG HỀ TỒN TẠI trong NGUỒN DỮ LIỆU, đó là ẢO GIÁC (Bịa đặt).
4. Hành động sửa chữa:
   - XÓA BỎ các câu bịa đặt.
   - VIẾT LẠI đoạn văn đó dựa TRỰC TIẾP và DUY NHẤT vào [NGUỒN DỮ LIỆU GỐC].
   - Giữ nguyên văn phong học thuật, cách chia đoạn, nhưng ĐẢM BẢO CHÍNH XÁC 100% so với nguồn.
   - Vẫn phải giữ nguyên định dạng JSON đầu ra.
   
ĐẦU RA BẮT BUỘC (JSON format):
Phải giữ đúng cấu trúc JSON ban đầu của chapter.
{{
  "title": "{chapter_content.get('title')}",
  "sections": [
    {{
      "title": "Tên mục (giữ nguyên)",
      "content": "Nội dung ĐÃ ĐƯỢC KIỂM TRA VÀ SỬA LỖI ẢO GIÁC...",
      "citations": ["url1", "url2"] 
    }}
  ]
}}
Tuyệt đối chỉ trả về JSON, không giải thích gì thêm.
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.2, # Temperature thấp để fact-checking chính xác
                response_format={"type": "json_object"}
            )
            text = resp.choices[0].message.content or ""
            return json.loads(_tach_json(text))
        except Exception as e:
             print(f"[WARN] Fact-check error (Chapter {chapter_content.get('title')}): {e}")
             if attempt < max_retries - 1:
                time.sleep(3)
             else:
                raise e
