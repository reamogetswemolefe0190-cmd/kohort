"""Export the existing Nova Site editorial brief without changing its findings."""
from pathlib import Path
import re
import html
import xml.etree.ElementTree as ET

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'deliverables/novasite/2026-10-10-positioning-brief/client-brief.html'
OUTPUT = ROOT / 'output/pdf/nova-site-positioning-report.pdf'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
for name, filename in [('Body', 'arial.ttf'), ('BodyBold', 'arialbd.ttf'), ('BodyItalic', 'ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + filename))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')
ink, purple, muted = map(colors.HexColor, ['#16181d', '#4b3df5', '#5f6672'])
styles = {
    'p': ParagraphStyle('Body', fontName='Body', fontSize=10.2, leading=14.5, textColor=ink, spaceAfter=9),
    'h1': ParagraphStyle('Title', fontName='BodyBold', fontSize=27, leading=32, textColor=ink, spaceAfter=18),
    'h2': ParagraphStyle('Section', fontName='BodyBold', fontSize=17, leading=22, textColor=ink, spaceBefore=13, spaceAfter=12, keepWithNext=True),
    'h3': ParagraphStyle('Subheading', fontName='BodyBold', fontSize=12, leading=16, textColor=ink, spaceBefore=10, spaceAfter=8, keepWithNext=True),
    'quote': ParagraphStyle('Quote', fontName='Body', fontSize=11, leading=16, textColor=purple, leftIndent=12, borderPadding=8, spaceAfter=10),
    'cell': ParagraphStyle('Cell', fontName='Body', fontSize=9, leading=12, textColor=ink),
    'brand': ParagraphStyle('Brand', fontName='BodyBold', fontSize=10, leading=14, textColor=purple, spaceAfter=16),
}

def normalize(value):
    return value.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')

def inline(node):
    result = html.escape(normalize(node.text or ''))
    for child in node:
        tag = {'strong': 'b', 'em': 'i', 'code': 'font'}.get(child.tag)
        content = inline(child)
        result += (f'<{tag}>{content}</{tag}>' if tag and tag != 'font' else content)
        result += html.escape(normalize(child.tail or ''))
    return result

def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor('#e6e8ec'))
    canvas.line(42, 36, width-42, 36)
    canvas.setFont('Body', 8)
    canvas.setFillColor(muted)
    canvas.drawString(42, 23, 'Kohort | Nova Site | Positioning recommendation')
    canvas.drawRightString(width-42, 23, str(doc.page))
    canvas.restoreState()

main = ET.fromstring(re.search(r'<main>.*?</main>', SOURCE.read_text(encoding='utf-8'), re.S).group())
story = []
page_starts = {'The position to communicate', 'Two clear ways to buy', 'What to do next', 'Appendix: evidence behind the advice'}
width = A4[0] - 84
for node in main:
    if node.tag == 'div':
        story.append(Paragraph('K O H O R T', styles['brand']))
    elif node.tag in ('h1', 'h2', 'h3', 'p'):
        if node.tag == 'h2' and ''.join(node.itertext()) in page_starts:
            story.append(PageBreak())
        story.append(Paragraph(inline(node), styles[node.tag]))
    elif node.tag == 'blockquote':
        for paragraph in node:
            story.append(Paragraph(inline(paragraph), styles['quote']))
    elif node.tag in ('ul', 'ol'):
        for index, item in enumerate(node, 1):
            bullet = f'{index}.' if node.tag == 'ol' else '\u2022'
            story.append(Paragraph(inline(item), ParagraphStyle('List', parent=styles['p'], leftIndent=12, firstLineIndent=0), bulletText=bullet))
    elif node.tag == 'table':
        rows = []
        for row in node.findall('.//tr'):
            rows.append([Paragraph(('<b>'+inline(cell)+'</b>') if cell.tag == 'th' else inline(cell), styles['cell']) for cell in row])
        n = len(rows[0])
        fractions = {2: [.35, .65], 3: [.09, .51, .40], 5: [.40, .15, .15, .15, .15]}[n]
        table = Table(rows, colWidths=[width*x for x in fractions], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eeecff')),
            ('LINEBELOW', (0,0), (-1,-1), .4, colors.HexColor('#e6e8ec')),
            ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
            ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ]))
        story.extend([table, Spacer(1, 12)])

doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=48,
                        title='Nova Site - positioning recommendation', author='Kohort')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(OUTPUT)
text = '\n'.join(page.extract_text() for page in reader.pages)
for phrase in ['Our recommendation', 'Two clear ways to buy', 'What to do next', 'Source records and confidence', '200 AI role-play responses']:
    assert phrase in text, phrase
print(f'{OUTPUT}\nPages: {len(reader.pages)}')
for i, page in enumerate(reader.pages, 1):
    print(f'Page {i}: {len(page.extract_text())} text characters')
