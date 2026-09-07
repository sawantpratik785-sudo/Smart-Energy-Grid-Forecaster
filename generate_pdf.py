"""
Generate Smart_Energy_Grid_Project.pdf from Capstone Final Presentation Guide
using ReportLab with clean typography, tables, and code formatting.
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def clean_xml_text(s):
    # Fix self-closing br
    s = re.sub(r'<br\s*>', r'<br/>', s, flags=re.IGNORECASE)
    # Fix ampersands not part of entity
    s = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', r'&amp;', s)
    # Clean up double math signs
    s = s.replace('$$', '')
    return s

def create_capstone_pdf():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, 'CAPSTONE_FINAL_PRESENTATION.md')
    pdf_path = os.path.join(current_dir, 'Smart_Energy_Grid_Project.pdf')
    
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    quote_style = ParagraphStyle(
        'Quote_Custom',
        parent=body_style,
        fontName='Helvetica-Oblique',
        leftIndent=18,
        rightIndent=18,
        spaceBefore=4,
        spaceAfter=8,
        textColor=colors.HexColor('#1e293b')
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        fontName='Courier',
        fontSize=6.8,
        leading=8.2,
        textColor=colors.HexColor('#0f172a')
    )

    story = []
    lines = text.split('\n')
    i = 0
    in_code_block = False
    code_lines = []

    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                code_text = '\n'.join(code_lines)
                box_table = Table([[Preformatted(code_text, code_style)]], colWidths=[7.0 * inch])
                box_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                    ('TOPPADDING', (0,0), (-1,-1), 3),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                    ('LEFTPADDING', (0,0), (-1,-1), 5),
                    ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ]))
                story.append(box_table)
                story.append(Spacer(1, 4))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Horizontal rule
        if stripped in ['---', '***']:
            story.append(Spacer(1, 3))
            line_table = Table([['']], colWidths=[7.0 * inch])
            line_table.setStyle(TableStyle([
                ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8'))
            ]))
            story.append(line_table)
            story.append(Spacer(1, 3))
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # Document Title
        if line.startswith('# '):
            clean_title = line[2:].replace('⚡', '').strip()
            story.append(Paragraph(f"<b>⚡ {clean_title}</b>", title_style))
            i += 1
            continue

        # Subtitle
        if line.startswith('### 🏆') or line.startswith('## Project:') or line.startswith('**Topic**:'):
            clean_sub = line.lstrip('#').strip()
            clean_sub = clean_xml_text(clean_sub)
            clean_sub = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_sub)
            story.append(Paragraph(clean_sub, subtitle_style))
            i += 1
            continue

        # Section Headings (##)
        if line.startswith('## '):
            clean_h = line[3:].strip()
            clean_h = clean_xml_text(clean_h)
            story.append(Paragraph(clean_h, h1_style))
            i += 1
            continue

        # Subsections (###)
        if line.startswith('### '):
            clean_h = line[4:].strip()
            clean_h = clean_xml_text(clean_h)
            story.append(Paragraph(clean_h, h2_style))
            i += 1
            continue

        # Quotes / Pitches
        if line.startswith('> '):
            clean_quote = line[2:].strip()
            clean_quote = clean_xml_text(clean_quote)
            clean_quote = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_quote)
            clean_quote = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_quote)
            story.append(Paragraph(f"<i>\"{clean_quote}\"</i>", quote_style))
            i += 1
            continue

        # Bullets
        if stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped):
            b_text = re.sub(r'^(?:[-*]|\d+\.)\s*', '', stripped)
            b_text = clean_xml_text(b_text)
            b_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', b_text)
            b_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', b_text)
            b_text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', b_text)
            story.append(Paragraph(f"• {b_text}", bullet_style))
            i += 1
            continue

        # Markdown tables
        if '|' in line and i + 1 < len(lines) and '|' in lines[i+1] and '---' in lines[i+1]:
            table_rows = []
            while i < len(lines) and '|' in lines[i]:
                t_line = lines[i].strip()
                if '---' in t_line:
                    i += 1
                    continue
                cells = [c.strip() for c in t_line.strip('|').split('|')]
                row_cells = []
                for cell in cells:
                    fmt_cell = clean_xml_text(cell)
                    fmt_cell = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', fmt_cell)
                    fmt_cell = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', fmt_cell)
                    row_cells.append(Paragraph(fmt_cell, body_style))
                table_rows.append(row_cells)
                i += 1
            
            if table_rows:
                num_cols = len(table_rows[0])
                col_w = (7.0 * inch) / max(num_cols, 1)
                t_obj = Table(table_rows, colWidths=[col_w] * num_cols)
                t_obj.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                    ('TOPPADDING', (0,0), (-1,-1), 2.5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
                ]))
                story.append(t_obj)
                story.append(Spacer(1, 4))
            continue

        # Normal text
        p_text = stripped
        p_text = clean_xml_text(p_text)
        p_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p_text)
        p_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', p_text)
        p_text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', p_text)
        story.append(Paragraph(p_text, body_style))
        i += 1

    doc.build(story)
    print(f"[SUCCESS] Generated PDF: {pdf_path}")

if __name__ == '__main__':
    create_capstone_pdf()
