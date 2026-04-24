from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class NguoiDung(UserMixin, db.Model):
    __tablename__ = 'nguoi_dung'
    id = db.Column(db.Integer, primary_key=True)
    ten_dang_nhap = db.Column(db.String(150), unique=True, nullable=False)
    mat_khau = db.Column("mat_khau_hash", db.String(255), nullable=True)
    la_admin = db.Column(db.Boolean, default=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    ho_ten = db.Column(db.String(255), nullable=True)
    anh_dai_dien = db.Column(db.String(500), nullable=True)
    token = db.Column(db.Integer, default=10)

    # Quan hệ với bảng lịch sử
    lich_su = db.relationship('LichSuGiaoTrinh', backref='nguoi_dung', lazy=True)

class LichSuGiaoTrinh(db.Model):
    __tablename__ = 'lich_su_giao_trinh'
    id = db.Column(db.Integer, primary_key=True)
    nguoi_dung_id = db.Column(db.Integer, db.ForeignKey('nguoi_dung.id'), nullable=True)
    chu_de = db.Column(db.String(255), nullable=False)
    noi_dung_html = db.Column(db.Text) # LongText in MySQL
    duong_dan_file = db.Column(db.String(255))
    do_dai_ky_tu = db.Column(db.Integer, default=0)
    ngay_tao = db.Column(db.DateTime, default=datetime.utcnow)
    da_xuat_file = db.Column(db.Boolean, default=False)

    @property
    def ma_cv(self):
        if self.duong_dan_file:
            # Assumes format: .../uuid.pdf
            import os
            basename = os.path.basename(self.duong_dan_file)
            return os.path.splitext(basename)[0]
        return None

    def __repr__(self):
        return f'<GiaoTrinh {self.chu_de}>'
