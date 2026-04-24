# -*- coding: utf-8 -*-
import os
import re
import logging
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    Table, TableStyle, PageTemplate, Frame
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- IMPORT V21.5 ENTERPRISE CLEANER ---
from dich_vu.xuat_tai_lieu.bo_loc_html import clean_for_reportlab as _clean_text

logger = logging.getLogger(__name__)

# --- 1. FONTS SETUP ---
# Ưu tiên Times New Roman để chuẩn giáo trình
FONT_NAME = "Helvetica" # Fallback
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"

try:
    # Windows paths
    times_path = "C:\\Windows\\Fonts\\times.ttf"
    times_bd_path = "C:\\Windows\\Fonts\\timesbd.ttf"
    times_it_path = "C:\\Windows\\Fonts\\timesi.ttf"
    
    # Linux paths (example)
    if not os.path.exists(times_path):
         times_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
         times_bd_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
         times_it_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"

    if os.path.exists(times_path) and os.path.exists(times_bd_path) and os.path.exists(times_it_path):
        pdfmetrics.registerFont(TTFont('Times-Roman', times_path))
        pdfmetrics.registerFont(TTFont('Times-Bold', times_bd_path))
        pdfmetrics.registerFont(TTFont('Times-Italic', times_it_path))
        FONT_NAME = 'Times-Roman'
        FONT_BOLD = 'Times-Bold'
        FONT_ITALIC = 'Times-Italic'
    else:
         # Fallback to Arial if Times not found
         arial_path = "C:\\Windows\\Fonts\\arial.ttf"
         if os.path.exists(arial_path):
             pdfmetrics.registerFont(TTFont('Arial', arial_path))
             FONT_NAME = 'Arial'
             FONT_BOLD = 'Arial' # Simple fallback
             FONT_ITALIC = 'Arial'
except Exception as e:
    logger.warning(f"Font setup failed: {e}")

# --- 2. STYLES ---
styles = getSampleStyleSheet()

# Title Style
style_Title = ParagraphStyle(
    'CustomTitle',
    parent=styles['Title'],
    fontName=FONT_BOLD,
    fontSize=24,
    leading=30,
    alignment=TA_CENTER,
    spaceAfter=20,
    textColor=colors.black
)

# Chapter Heading (Heading 1)
style_H1 = ParagraphStyle(
    'CustomH1',
    parent=styles['Heading1'],
    fontName=FONT_BOLD,
    fontSize=16,
    leading=20,
    alignment=TA_CENTER,
    spaceAfter=12,
    textColor=colors.black # Màu đen chuẩn
)

# Section Heading (Heading 2)
style_H2 = ParagraphStyle(
    'CustomH2',
    parent=styles['Heading2'],
    fontName=FONT_BOLD,
    fontSize=14,
    leading=18,
    alignment=TA_LEFT,
    spaceBefore=12,
    spaceAfter=6,
    textColor=colors.black
)

# Normal Text (Justified, Indented)
style_Normal = ParagraphStyle(
    'CustomNormal',
    parent=styles['Normal'],
    fontName=FONT_NAME,
    fontSize=12,
    leading=15, # Line spacing 1.3 approx
    alignment=TA_JUSTIFY, # Căn đều 2 bên
    firstLineIndent=0.5 * inch, # Thụt đầu dòng
    spaceAfter=6
)

# List Items
style_List = ParagraphStyle(
    'CustomList',
    parent=styles['Normal'],
    fontName=FONT_NAME,
    fontSize=12,
    leading=15,
    alignment=TA_LEFT,
    leftIndent=0.5 * inch,
    spaceAfter=2
)

# Citation/Italic
style_Italic = ParagraphStyle(
    'CustomItalic',
    parent=styles['Normal'],
    fontName=FONT_ITALIC,
    fontSize=10,
    leading=12,
    alignment=TA_LEFT,
    textColor=colors.grey,
    spaceAfter=6
)

# V33: Chapter Summary
style_Summary = ParagraphStyle(
    'ChapterSummary',
    parent=styles['Normal'],
    fontName=FONT_ITALIC,
    fontSize=11,
    leading=14,
    alignment=TA_JUSTIFY,
    leftIndent=0.4 * inch,
    rightIndent=0.4 * inch,
    spaceBefore=6,
    spaceAfter=12,
    textColor=colors.HexColor('#333333'),
    borderPadding=6,
)

# V33: Glossary Term
style_GlossaryTerm = ParagraphStyle(
    'GlossaryTerm',
    parent=styles['Normal'],
    fontName=FONT_NAME,
    fontSize=12,
    leading=15,
    alignment=TA_LEFT,
    spaceAfter=4
)

# (Hàm _clean_text cũ đã được thay thế bằng import từ bo_loc_html phía trên)

# --- 3. PAGE TEMPLATE (Header/Footer) ---
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = "Trang %d" % page_num
    canvas.saveState()
    canvas.setFont(FONT_NAME, 9)
    # Footer Center
    canvas.drawCentredString(A4[0] / 2.0, 1.5 * cm, text)
    # Header Line
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, A4[1]-2*cm, A4[0]-2*cm, A4[1]-2*cm)
    canvas.restoreState()

def xuat_pdf(ket_qua: dict, duong_dan_pdf: str):
    book = ket_qua.get("book_vi", {})
    doc = SimpleDocTemplate(
        duong_dan_pdf,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2.5*cm, # Lề trái rộng hơn chút để đóng gáy
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )
    
    story = []

    # 1. TITLE PAGE
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(book.get("title", "GIÁO TRÌNH").upper(), style_Title))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Tài liệu biên soạn tự động bởi hệ thống AI", 
                           ParagraphStyle('SubTitle', parent=style_Normal, alignment=TA_CENTER, fontName=FONT_ITALIC)))
    story.append(PageBreak())

    # 2. TABLE OF CONTENTS (Static)
    story.append(Paragraph("MỤC LỤC", style_Title))
    story.append(Spacer(1, 0.5*cm))
    
    chapters = book.get("chapters", [])
    
    # Using Table for cleaner TOC
    toc_data = []
    for idx, ch in enumerate(chapters, 1):
        # Chapter row
        toc_data.append([f"CHƯƠNG {idx}: {ch.get('title','').upper()}", ""])
        # Sections
        for jdx, sec in enumerate(ch.get("sections", []), 1):
            toc_data.append([f"   {idx}.{jdx}. {sec.get('title','')}", ""]) # Indent visualization
    
    if toc_data:
        t = Table(toc_data, colWidths=[15*cm, 1*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), FONT_NAME),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(t)
    
    story.append(PageBreak())

    # 2.5 GLOSSARY (V33)
    glossary = ket_qua.get("glossary", [])
    if glossary:
        story.append(Paragraph("BẢNG THUẬT NGỮ", style_H1))
        story.append(Spacer(1, 0.3*cm))
        for item in glossary:
            term = _clean_text(item.get('term', ''))
            definition = _clean_text(item.get('definition', ''))
            story.append(Paragraph(f"<b>{term}</b>: {definition}", style_GlossaryTerm))
        story.append(PageBreak())

    # 3. CONTENT
    for idx, ch in enumerate(chapters, 1):
        # Chapter Title
        story.append(Paragraph(f"CHƯƠNG {idx}: {ch.get('title','').upper()}", style_H1))
        
        # V33: Chapter Summary
        summary = ch.get("summary", "")
        if summary:
            clean_summary = _clean_text(summary)
            story.append(Paragraph(f"<i>Tóm tắt chương: {clean_summary}</i>", style_Summary))

        
        for jdx, sec in enumerate(ch.get("sections", []), 1):
            # Section Title
            story.append(Paragraph(f"{idx}.{jdx}. {sec.get('title','')}", style_H2))
            
            # Content separation
            raw_content = sec.get("content", "")
            # Split by newlines to handle paragraphs
            paras = raw_content.split('\n')
            for p_text in paras:
                clean_p = _clean_text(p_text.strip())
                if not clean_p: continue
                
                # Kiểm tra các phần tử dạng danh sách
                if clean_p.startswith("- ") or clean_p.startswith("* "):
                    clean_p = clean_p[2:].strip()
                    story.append(Paragraph(f"• {clean_p}", style_List))
                elif re.match(r'^\d+\.\s', clean_p):
                    story.append(Paragraph(clean_p, style_List))
                elif clean_p == "---":
                    # Xử lý đường kẻ ngang (V15.0)
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Table([[""]], colWidths=[15*cm], style=TableStyle([('LINEABOVE', (0,0), (-1,-1), 1, colors.grey)])))
                    story.append(Spacer(1, 0.1*inch))
                else:
                    story.append(Paragraph(clean_p, style_Normal))

        story.append(PageBreak())

    # 4. REFERENCES
    story.append(Paragraph("TÀI LIỆU THAM KHẢO", style_H1))
    refs = ket_qua.get("references", [])
    try:
        refs = sorted(refs, key=lambda x: int(x.get("id", 0)) if isinstance(x, dict) else 0)
    except:
        pass
        
    for u in refs:
         if isinstance(u, dict):
             ref_str = f"[{u.get('id', '')}] Bách khoa toàn thư mở Wikipedia. \"{u.get('title', 'Nguồn')}\". [Trực tuyến]. Nguồn: <u>{u.get('url', '')}</u>. (Truy cập năm 2026)."
             story.append(Paragraph(ref_str, style_Normal))
         else:
             story.append(Paragraph(f"• {u}", style_List))

    # Build PDF
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
