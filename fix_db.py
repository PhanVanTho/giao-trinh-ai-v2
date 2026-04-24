
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "giao_trinh_ai")
DB_PORT = int(os.getenv("DB_PORT", 3306))

def fix_database():
    print(f"Connecting to database {DB_NAME} at {DB_HOST}:{DB_PORT}...")
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()
        
        # Kiểm tra xem bảng nguoi_dung đã tồn tại chưa
        cursor.execute("SHOW TABLES LIKE 'nguoi_dung'")
        if not cursor.fetchone():
            print("Table 'nguoi_dung' does not exist. It will be created by the app.")
            return

        # Kiểm tra các cột trong bảng nguoi_dung
        cursor.execute("DESCRIBE nguoi_dung")
        columns = [row['Field'] for row in cursor.fetchall()]
        print(f"Current columns in 'nguoi_dung': {columns}")

        if 'mat_khau_hash' not in columns and 'mat_khau' not in columns:
            print("Missing password column. Adding 'mat_khau_hash'...")
            sql = "ALTER TABLE nguoi_dung ADD COLUMN mat_khau_hash VARCHAR(255) NOT NULL DEFAULT '';"
            cursor.execute(sql)
            conn.commit()
            print("Successfully added 'mat_khau_hash' column.")
        else:
            print("Password column exists.")

        if 'la_admin' not in columns:
            print("Missing column 'la_admin'. Adding it...")
            sql = "ALTER TABLE nguoi_dung ADD COLUMN la_admin BOOLEAN DEFAULT FALSE;"
            cursor.execute(sql)
            conn.commit()
            print("Successfully added 'la_admin' column.")
        else:
            print("Admin column 'la_admin' exists.")

        if 'token' not in columns:
            print("Missing column 'token'. Adding it...")
            sql = "ALTER TABLE nguoi_dung ADD COLUMN token INT DEFAULT 10;"
            cursor.execute(sql)
            conn.commit()
            print("Successfully added 'token' column.")
        else:
            print("Column 'token' already exists.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

if __name__ == "__main__":
    fix_database()
