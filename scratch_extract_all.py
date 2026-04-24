import json
import os
import glob
import sys

# Ensure stdout uses utf-8
sys.stdout.reconfigure(encoding='utf-8')

folder_path = r'd:\tu_dong_giao_trinh\ThucNghiem_KetQua'
txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

for filepath in txt_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header = []
        json_str = ''
        in_json = False
        has_json = False

        for line in lines:
            if line.strip() == '{' and not in_json:
                in_json = True
                has_json = True
            if not in_json:
                header.append(line)
            if in_json:
                json_str += line

        if has_json:
            data = json.loads(json_str)
            toc = []
            
            # Check both possible paths
            chapters = data.get('book_vi', {}).get('chapters', [])
            if not chapters:
                chapters = data.get('chapters', [])

            for chap in chapters:
                toc.append(f"Chương: {chap.get('title')}")
                for sec in chap.get('sections', []):
                    toc.append(f"  - {sec.get('title')}")
                    
            with open(filepath, 'w', encoding='utf-8') as f:
                for h in header:
                    f.write(h)
                if not toc:
                    f.write("Không tìm thấy dàn ý trong JSON.\n")
                else:
                    f.write('\n'.join(toc) + '\n')
            print(f'Done: {os.path.basename(filepath)}')
        else:
            print(f'No JSON found (already processed or failed): {os.path.basename(filepath)}')
            
    except Exception as e:
        print(f'Error processing {os.path.basename(filepath)}: {e}')
