"""Build a compact 1-2 page summary for the supervisor.

The summary reuses the current v2.1 report tables so that the numbers cannot
drift between documents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

REPORT = Path(r"C:/Users/eray/Desktop/RAPOR_AKADEMIK_REGISTRATION_RADIOMICS_v2.1.docx")
OUT = Path(r"C:/Users/eray/Desktop/OZET_BOBREK_4ORGAN_1MM.docx")
PRIMARY = Path(r"C:/Users/eray/Desktop/BOBREK_1MM_PRIMARY.xlsx")
RECON = Path(r"C:/Users/eray/Desktop/BOBREK_1MM_VS_3MM.xlsx")
SITE = "https://erayakgunn.github.io/registration-overlay/"


def cell(cell, text, bold=False, size=8.5):
    cell.text = "" if text is None else str(text)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.name = "Arial"
            if bold:
                run.bold = True


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Shading Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell(t.rows[0].cells[i], h, bold=True)
    for values in rows:
        cells = t.add_row().cells
        for i, value in enumerate(values):
            cell(cells[i], value)
    if widths:
        for row in t.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph()
    return t


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        for run in p.runs:
            run.font.size = Pt(9)


def main() -> None:
    report = Document(REPORT)
    cross_table = report.tables[2]
    test_table = report.tables[4]

    def rows_of(table_doc, indexes):
        result = []
        for row in table_doc.rows[1:]:
            values = [c.text for c in row.cells]
            result.append([values[i] for i in indexes])
        return result

    cross_rows = rows_of(cross_table, [0, 1, 2, 4, 7, 8, 9])
    test_rows = rows_of(test_table, [0, 1, 2, 4, 6, 7, 8])

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(9)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Böbrek Dahil Dört Organ Registration Değerlendirmesi")
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(31, 78, 121)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Kısa Özet | 1 mm primary böbrek analizi | 17 Kasım 2026").italic = True
    doc.add_paragraph()

    p = doc.add_paragraph(
        "Alper hocam, tüm görüntülerde sağ ve sol böbreklere registration "
        "uyguladım; dört organı (aorta, dalak, sağ böbrek, sol böbrek) siteye ve "
        "all-pairs PCCT/SEP Excel dosyalarına ekledim. Böbrek için primary "
        "analizi 1 mm ile yaptım; 3 mm’yi ayrı duyarlılık olarak bıraktım. "
        "Aşağıdaki sayılar v2.1 raporundan doğrudan alınmıştır."
    )
    p.paragraph_format.space_after = Pt(6)

    doc.add_paragraph("Cross-date registration (önce = identity, sonra = rigid)").runs[0].bold = True
    table(
        doc,
        ["Veri", "Organ", "n", "Dice sonra", "ASSD (mm)", "MSSD (mm)", "HD95 (mm)"],
        cross_rows,
        widths=[0.9, 1.1, 0.4, 0.7, 0.75, 0.75, 0.75],
    )

    doc.add_paragraph("Zurich test-retest — 1 mm primary").runs[0].bold = True
    table(
        doc,
        ["Veri", "Organ", "n", "Dice sonra", "ASSD (mm)", "MSSD (mm)", "HD95 (mm)"],
        test_rows,
        widths=[1.25, 1.1, 0.4, 0.7, 0.75, 0.75, 0.75],
    )

    doc.add_paragraph("Öne çıkan bulgular").runs[0].bold = True
    bullets(doc, [
        "Cross-date sonrası medyan Dice: PCCT 0.931 (aorta), 0.964 (dalak), 0.945 (sağ böbrek), 0.956 (sol böbrek); SEP 0.918, 0.959, 0.942, 0.949.",
        "1 mm primary böbrek: ASSD/MSSD 0.33–0.46 mm, HD95 1.0–1.2 mm; böbrekler dalak/aorta ile aynı doğruluk liginde.",
        "528 böbrek maskesinin 18’inde (%3.4) izole adacık var: 10 cross-date PCCT + 8 test-retest 3 mm.",
        "Adacıklarda Dice 0.94 kalırken ASSD/MSSD 10–35 mm’ye, HD95 299 mm’ye çıkabiliyor; largest-component ile ASSD 0.3–0.5 mm, HD95 1.0 mm’ye iniyor. Sorun registration değil, segmentasyon.",
        "SEP 20260731_s80 tek bileşenli gerçek segmentasyon/rekonstrüksiyon hatası; registration çözmüyor.",
        "3 mm: 8 adacık; sağ ortalama ASSD 7.76 mm, HD95 108.7 mm. 3 mm sol medyan ASSD 0.226 mm daha düşük görünse de smoothing/partial volume etkisi; primary değil.",
        "All-pairs PCCT (3.916 çift) ve SEP (2.278 çift) workbook’larına böbrek sütunları, özet satırları ve 8 böbrek matrisi eklendi.",
    ])

    doc.add_paragraph("Sonuç ve öneri").runs[0].bold = True
    doc.add_paragraph(
        "Böbrek primary analizi 1 mm ile yapılmalı; 3 mm ayrı raporlanmalı ve "
        "pooled edilmemelidir. Böbrek yorumu Dice tek başına değil, "
        "ASSD/MSSD/HD95, medyan/IQR ve aykırı vaka listesiyle birlikte "
        "verilmelidir. s80 tek vaka olarak ayrı QC’de tutulmalı; tüm pipeline’ı "
        "değiştirmeye gerek yok."
    )

    doc.add_paragraph("Dosyalar").runs[0].bold = True
    bullets(doc, [
        "Rapor: RAPOR_AKADEMIK_REGISTRATION_RADIOMICS_v2.1.docx",
        "Primary böbrek: BOBREK_1MM_PRIMARY.xlsx",
        "1 mm vs 3 mm: BOBREK_1MM_VS_3MM.xlsx",
        "Cross-date böbrek: CROSSDATE_BOBREK.xlsx",
        "Site: " + SITE,
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
