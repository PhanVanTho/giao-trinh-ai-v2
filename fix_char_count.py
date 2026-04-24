# -*- coding: utf-8 -*-
"""
fix_char_count.py - Backfill do_dai_ky_tu for old history records (V32 Fix)
"""
import os
import sys
import json

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()
from cau_hinh import CauHinh
from ung_dung import app
from mo_hinh import db, LichSuGiaoTrinh

def backfill():
    with app.app_context():
        records = LichSuGiaoTrinh.query.filter(
            (LichSuGiaoTrinh.do_dai_ky_tu == 0) | (LichSuGiaoTrinh.do_dai_ky_tu == None)
        ).all()
        
        print(f"Found {len(records)} records to update.")
        updated = 0
        
        for r in records:
            ma_cv = r.ma_cv
            if not ma_cv:
                continue
            
            p_json = os.path.join(CauHinh.THU_MUC_JSON, f"{ma_cv}.json")
            if not os.path.exists(p_json):
                print(f"  [SKIP] {r.chu_de} - No JSON file ({ma_cv})")
                continue
            
            try:
                with open(p_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                tong_ky_tu = 0
                for chap in data.get('book_vi', {}).get('chapters', []):
                    for sec in chap.get('sections', []):
                        tong_ky_tu += len(sec.get('content', ''))
                
                if tong_ky_tu > 0:
                    r.do_dai_ky_tu = tong_ky_tu
                    updated += 1
                    print(f"  [OK] {r.chu_de} -> {tong_ky_tu:,} chars")
                else:
                    print(f"  [WARN] {r.chu_de} - Empty content in JSON")
            except Exception as e:
                print(f"  [ERR] {r.chu_de} - {e}")
        
        if updated > 0:
            db.session.commit()
            print(f"\nDone! Updated {updated}/{len(records)} records.")
        else:
            print("\nNo records updated.")

if __name__ == "__main__":
    backfill()
