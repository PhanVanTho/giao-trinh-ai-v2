# -*- coding: utf-8 -*-
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

conn = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASS", ""),
    database=os.getenv("DB_NAME", "giao_trinh_ai"),
    port=int(os.getenv("DB_PORT", "3306"))
)

cur = conn.cursor()

try:
    cur.execute("SHOW COLUMNS FROM nguoi_dung LIKE 'google_id'")
    if cur.fetchone():
        print("Column google_id already exists. Skipping.")
    else:
        cur.execute("""
            ALTER TABLE nguoi_dung 
            ADD COLUMN google_id VARCHAR(255) NULL DEFAULT NULL AFTER email,
            ADD COLUMN ho_ten VARCHAR(255) NULL DEFAULT NULL AFTER google_id,
            ADD COLUMN anh_dai_dien VARCHAR(500) NULL DEFAULT NULL AFTER ho_ten,
            ADD UNIQUE INDEX uq_google_id (google_id)
        """)
        print("OK: Added google_id, ho_ten, anh_dai_dien columns.")

    cur.execute("""
        ALTER TABLE nguoi_dung 
        MODIFY COLUMN mat_khau_hash VARCHAR(255) NULL DEFAULT NULL
    """)
    print("OK: mat_khau_hash is now nullable.")

    conn.commit()
    print("Migration done!")

except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()
