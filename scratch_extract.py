import json
import os

filepath = r'd:\tu_dong_giao_trinh\ThucNghiem_KetQua\Tâm lý học nhận thức.txt'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header = []
json_str = ''
in_json = False

for line in lines:
    if line.strip() == '{' and not in_json:
        in_json = True
    if not in_json:
        header.append(line)
    if in_json:
        json_str += line

try:
    data = json.loads(json_str)
    toc = []
    for chap in data.get('book_vi', {}).get('chapters', []):
        toc.append(f"Chương: {chap.get('title')}")
        for sec in chap.get('sections', []):
            toc.append(f"  - {sec.get('title')}")
            
    with open(filepath, 'w', encoding='utf-8') as f:
        for h in header:
            f.write(h)
        if not toc:
            f.write("Không tìm thấy dàn ý trong JSON.")
        else:
            f.write('\n'.join(toc))
    print('Đã trích xuất Dàn ý thành công!')
except Exception as e:
    print('Lỗi:', e)
