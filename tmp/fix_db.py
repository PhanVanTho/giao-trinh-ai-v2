from ung_dung import app, db
from sqlalchemy import text

with app.app_context():
    try:
        # Check current column status (Optional but good for logging)
        # Fix: Alter the table to allow NULL for nguoi_dung_id
        db.session.execute(text("ALTER TABLE lich_su_giao_trinh MODIFY nguoi_dung_id INT NULL;"))
        db.session.commit()
        print("Successfully altered lich_su_giao_trinh table. nguoi_dung_id is now nullable.")
    except Exception as e:
        db.session.rollback()
        print(f"Error altering table: {e}")
