import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# Xem file thực nghiệm cũ (Trí tuệ nhân tạo - đã có từ trước) để so sánh
# File mới nhất đã qua Polish Layer
d = json.load(open(r'd:\tu_dong_giao_trinh\du_lieu\dau_ra\json\275a185c-0c84-43f6-baa3-2a56ce22b759.json', 'r', encoding='utf-8'))
b = d.get('book_vi', d)

# Xem metadata
print("=== JOB METADATA ===")
print(f"Title: {b.get('title', '?')}")
print(f"KQS: {d.get('kqs', '?')}")

chs = b.get('chapters', [])
print(f"Chapters: {len(chs)}")
print()
print("=== CHAPTER LIST ===")
for i, c in enumerate(chs):
    secs = c.get('sections', [])
    print(f"  Ch{i+1}: {c.get('title', '?')} ({len(secs)} sections)")
