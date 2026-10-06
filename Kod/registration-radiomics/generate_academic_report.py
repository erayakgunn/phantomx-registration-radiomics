"""Build a Turkish academic DOCX report from the completed V6 analysis."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


FEATURE_FAMILIES = {
    "morph": "Shape",
    "loc": "Local intensity peaks",
    "stat": "First-order statistics",
    "ih": "Intensity histogram",
    "ivh": "Intensity-volume histogram",
    "cm": "GLCM",
    "rlm": "GLRLM",
    "szm": "GLSZM",
    "dzm": "GLDZM",
    "ngt": "NGTDM",
    "ngl": "NGLDM",
}
ROOT = Path(__file__).resolve().parents[2]


def add_table(doc: Document, headers: list[str], rows: list[list], widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = str(header)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cells[i].text = str(value)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph()
    return table


def add_caption(doc: Document, text: str):
    paragraph = doc.add_paragraph(text, style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def family_summary(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["family"] = data["feature"].map(
        lambda name: FEATURE_FAMILIES.get(name.split("_", 1)[0], "Other")
    )
    return data.groupby("family").agg(
        features=("n_features", "sum"),
        cv_improved=("cv_improved", "sum"),
        pairwise_improved=("pairwise_difference_improved", "sum"),
        both_improved=("both_improved", "sum"),
        fdr_significant=("fdr_significant_tests", "sum"),
        fdr_improved=("fdr_significant_improved", "sum"),
        fdr_worsened=("fdr_significant_worsened", "sum"),
    ).reset_index()


def make_figures(reg_root: Path, analysis: Path, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    metrics = []
    methods = Counter()
    for qc_path in reg_root.glob("*/registration_qc.json"):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        methods[qc.get("registration_method", "unknown")] += 1
    for csv_path in reg_root.glob("*/metrics.csv"):
        values = {row["label"]: row for row in csv.DictReader(csv_path.open(encoding="utf-8"))}
        metrics.append(values)
    organs = (("1", "Dalak"), ("52", "Aorta"))
    before = [statistics.mean(float(row[label]["before_dice"]) for row in metrics) for label, _ in organs]
    after = [statistics.mean(float(row[label]["after_dice"]) for row in metrics) for label, _ in organs]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(organs))
    ax.bar(x - .18, before, .36, label="Registration öncesi", color="#9aa6b2")
    ax.bar(x + .18, after, .36, label="V6 sonrası", color="#2878B5")
    ax.set_xticks(x, [name for _, name in organs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ortalama Dice")
    ax.set_title("V6 registration öncesi/sonrası organ overlap")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    dice_fig = figures / "v6_dice_summary.png"
    fig.savefig(dice_fig, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = list(methods)
    ax.bar(labels, [methods[k] for k in labels], color="#4C956C")
    ax.set_ylabel("Pair sayısı")
    ax.set_title("V6’da kullanılan dönüşüm yöntemi")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    method_fig = figures / "v6_method_counts.png"
    fig.savefig(method_fig, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, mode in zip(axes, ("3d", "2d")):
        summary = pd.read_csv(analysis / f"feature_family_summary_{mode}.csv")
        summary["pct"] = 100 * summary["both_improved"] / summary["n_features"]
        summary = summary.sort_values("pct", ascending=True)
        ax.barh(summary["family"], summary["pct"], color="#D48A3A")
        ax.set_title(f"{mode.upper()} features")
        ax.set_xlabel("CV ve pairwise farkı birlikte iyileşen (%)")
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    family_fig = figures / "feature_family_improvements.png"
    fig.savefig(family_fig, dpi=180)
    plt.close(fig)
    return dice_fig, method_fig, family_fig, methods, before, after


def main(args: argparse.Namespace):
    reg_root = args.registration_root
    analysis = reg_root / "analysis"
    figure_dir = analysis / "report_figures"
    dice_fig, method_fig, family_fig, methods, dice_before, dice_after = make_figures(
        reg_root, analysis, figure_dir
    )

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(.7)
    section.bottom_margin = Inches(.7)
    section.left_margin = Inches(.8)
    section.right_margin = Inches(.8)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10)
    styles["Title"].font.name = "Arial"
    styles["Title"].font.size = Pt(22)
    styles["Heading 1"].font.color.rgb = RGBColor(31, 78, 121)
    styles["Heading 2"].font.color.rgb = RGBColor(47, 84, 150)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Longitudinal PhantomX CT’de Organ Radyomik Ölçümlerinin\nRegistration-Based Spatial Standardization’ı")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("V6 analiz raporu | 6 Ekim 2026").italic = True
    doc.add_paragraph()

    doc.add_heading("Özet", level=1)
    doc.add_paragraph(
        "Bu çalışmada PhantomX longitudinal PCCT görüntülerinde spleen ve aorta "
        "maskelerinin fiziksel koordinat farklılıkları ve registration’ın radyomik "
        "tekrarlanabilirliğe etkisi incelendi. Toplam 88 moving seri, 20260805 take1 "
        "referansına karşı değerlendirildi. Registration sonrası ortalama Dice "
        f"spleen için {dice_before[0]:.3f}’ten {dice_after[0]:.3f}’e, aorta için "
        f"{dice_before[1]:.3f}’ten {dice_after[1]:.3f}’e yükseldi. En güvenilir "
        "fallback, vertebra/kaburga/sternum centroid’lerinden türetilen landmark "
        "rigid dönüşümü oldu. Bununla birlikte, radiomic feature tekrarlanabilirliği "
        "feature ailesine göre değişti; registration bütün feature’ları iyileştirmedi."
    )

    doc.add_heading("Araştırma Sorusu", level=1)
    doc.add_paragraph(
        "Longitudinal CT’de explicit rigid registration, registration-free DICOM "
        "world-space yaklaşımına kıyasla organ maskesi hizasını ve organ-level "
        "radiomic feature tekrarlanabilirliğini iyileştiriyor mu?"
    )

    doc.add_heading("Veri ve Değerlendirme Tasarımı", level=1)
    add_table(doc, ["Öğe", "Açıklama"], [
        ["Veri", "22 PhantomX oturumu; 89 seri"],
        ["Analiz pair’i", "1 fixed referans (20260805 take1) + 88 moving seri"],
        ["Protokol", "NAEOTOM Alpha PCCT, Q3/Qr40f, 120 kV, 100 mAs, 300 mm FOV, 1 mm"],
        ["Hedef organlar", "Spleen (label 1), aorta (label 52)"],
        ["Geometrik metrikler", "Dice, centroid displacement, HD95, ASSD"],
        ["Radiomics", "MIRP 2.5.0, IBSI ayarları; ayrı 2D ve 3D"],
    ])

    doc.add_heading("Kullanılan Yöntem", level=1)
    doc.add_paragraph(
        "DICOM geometri bilgisi korunarak CT ve multi-label maskeler fiziksel/world "
        "koordinatlarda karşılaştırıldı. Hedef organ centroid displacement’ı 5 mm "
        "veya altındaysa identity/world-space sonuç korundu. Büyük farklarda önce "
        "correlation tabanlı rigid registration denendi. Registration QC’si, hedef "
        "organ Dice’ına göre değil bağımsız vertebra, kaburga ve sternum maskeleri "
        "üzerinden yapıldı. Rigid sonuç QC’yi geçmediğinde landmark centroid’lerinden "
        "Kabsch/SVD rigid dönüşümü; bu da geçmezse translation fallback denendi. "
        "Hiçbir aday QC’yi geçmezse identity dönüşümü korundu. CT linear, label maskeler "
        "nearest-neighbor interpolation ile fixed grid’e taşındı."
    )
    doc.add_paragraph(
        "Aday dönüşümün kabulünde stabil Dice’ın artması ve centroid displacement, "
        "HD95, ASSD değerlerinin tanımlı toleranslarda kalması şart koşuldu. Böylece "
        "hedef organ maskeleri registration’ı seçmek için kullanılmadı."
    )

    doc.add_heading("Dice Eşiği ve Yorum", level=1)
    ceiling_path = ROOT / "PhantomX" / "REGISTRATION" / "segmentation_ceiling.csv"
    ceiling = list(csv.DictReader(ceiling_path.open(encoding="utf-8")))
    ceilings = {}
    for organ in ("spleen", "aorta"):
        values = [float(row["dice"]) for row in ceiling if row["organ"] == organ]
        ceilings[organ] = (statistics.median(values), max(values))
    add_table(doc, ["Organ", "Aynı oturum median Dice", "Gözlenen maksimum Dice", "V6 ortalama final Dice"], [
        ["Spleen", f"{ceilings['spleen'][0]:.4f}", f"{ceilings['spleen'][1]:.4f}", f"{dice_after[0]:.4f}"],
        ["Aorta", f"{ceilings['aorta'][0]:.4f}", f"{ceilings['aorta'][1]:.4f}", f"{dice_after[1]:.4f}"],
    ])
    doc.add_paragraph(
        "Dice = 2|A∩B|/(|A|+|B|), yani iki binary maskenin voxel overlap ölçüsüdür. "
        "Dice 1’e yaklaştıkça overlap artar; tek başına ground-truth segmentation "
        "doğruluğunu veya radiomic tekrarlanabilirliği kanıtlamaz. Aynı oturumdaki "
        "TotalSegmentator maskelerinde aorta için gözlenen maksimum Dice 0.99’un "
        "altında olduğundan, tüm pair’lerde Dice≥0.99 şartı bu veriyle gerçekçi bir "
        "kabul kriteri değildir."
    )

    doc.add_heading("Registration Sonuçları", level=1)
    add_table(doc, ["Ölçüm", "Spleen", "Aorta"], [
        ["Ortalama Dice, öncesi", f"{dice_before[0]:.4f}", f"{dice_before[1]:.4f}"],
        ["Ortalama Dice, V6 sonrası", f"{dice_after[0]:.4f}", f"{dice_after[1]:.4f}"],
        ["Dice ≥ 0.90 pair sayısı", "85 / 88", "84 / 88"],
        ["Dice = 0 pair sayısı", "0", "0"],
    ])
    add_table(doc, ["V6 dönüşüm yöntemi", "Pair sayısı"], [
        [method, count] for method, count in methods.items()
    ])
    doc.add_picture(str(dice_fig), width=Inches(6.4))
    add_caption(doc, "Şekil 1. V6 registration öncesi ve sonrası ortalama organ Dice değerleri.")
    doc.add_picture(str(method_fig), width=Inches(6.4))
    add_caption(doc, "Şekil 2. V6’da pair başına kullanılan dönüşüm yöntemi.")

    doc.add_heading("Düşük Dice Pair’lerinin Görsel QC’si", level=1)
    metrics = []
    for f in reg_root.glob("*/metrics.csv"):
        pair = f.parent.name
        data = {row["label"]: row for row in csv.DictReader(f.open(encoding="utf-8"))}
        spleen_dice = float(data["1"]["after_dice"])
        aorta_dice = float(data["52"]["after_dice"])
        if spleen_dice < 0.90 or aorta_dice < 0.90:
            metrics.append((pair, spleen_dice, aorta_dice))
    add_table(doc, ["Pair", "Spleen Dice", "Aorta Dice"], [
        [pair.replace("__", " / "), f"{s:.3f}", f"{a:.3f}"] for pair, s, a in metrics
    ])
    qc_dir = reg_root / "visual_qc"
    for pair, spleen_dice, aorta_dice in metrics:
        png = qc_dir / f"{pair}.png"
        if png.exists():
            doc.add_paragraph(f"{pair.replace('__', ' / ')} — spleen {spleen_dice:.3f}, aorta {aorta_dice:.3f}")
            doc.add_picture(str(png), width=Inches(6.4))
            add_caption(doc, "Şekil. Düşük overlap pair için identity ve registered maskelerin görsel karşılaştırması.")

    doc.add_heading("Radiomic Feature Tekrarlanabilirliği", level=1)
    doc.add_paragraph(
        "Repeatability, her acquisition session içindeki tekrarlar kullanılarak "
        "değerlendirildi. CV ve median pairwise relative difference hesaplandı; "
        "baseline ve registered değerler session bazında eşleştirildi. Feature başına "
        "Wilcoxon signed-rank testi, çoklu karşılaştırmalar için Benjamini–Hochberg "
        "FDR düzeltmesi uygulandı. Numeric DICOM/MIRP metadata kolonları feature "
        "analizinden çıkarıldı."
    )
    rep_rows = []
    for mode in ("3d", "2d"):
        path = analysis / f"repeatability_comparison_{mode}.csv"
        frame = pd.read_csv(path)
        sig = pd.read_csv(analysis / f"significance_{mode}.csv")
        sig = sig[sig["significant_fdr_05"]]
        rep_rows.append([
            mode.upper(), len(frame), int(frame["cv_improved"].sum()),
            int(frame["pairwise_difference_improved"].sum()),
            int((frame["cv_improved"] & frame["pairwise_difference_improved"]).sum()),
            len(sig), int((sig["direction"] == "improved").sum()),
            int((sig["direction"] != "improved").sum()),
        ])
    add_table(doc, ["Mode", "Feature-organ", "CV iyileşti", "Pairwise iyileşti", "İkisi", "FDR anlamlı", "Anlamlı iyileşti", "Anlamlı kötüleşti"], rep_rows)

    doc.add_heading("Feature Aileleri", level=1)
    family_rows = []
    for mode in ("3d", "2d"):
        fam = pd.read_csv(analysis / f"feature_family_summary_{mode}.csv")
        totals = fam.groupby("family").sum(numeric_only=True)
        for family_name, row in totals.iterrows():
            if row["n_features"] == 0:
                continue
            family_rows.append([
                mode.upper(), family_name, int(row["n_features"]),
                f"{100 * row['both_improved'] / row['n_features']:.1f}%",
                int(row["fdr_significant_tests"]),
                int(row["fdr_significant_improved"]),
                int(row["fdr_significant_worsened"]),
            ])
    add_table(doc, ["Mode", "Feature family", "N", "CV+pairwise birlikte iyileşen", "FDR anlamlı", "İyileşme", "Kötüleşme"], family_rows)
    doc.add_picture(str(family_fig), width=Inches(6.5))
    add_caption(doc, "Şekil 3. Feature ailesi bazında CV ve pairwise farkın birlikte iyileştiği feature oranı.")

    doc.add_heading("Resampling Kontrolü: World-space ve Registration", level=1)
    world_rows = []
    for mode in ("3d", "2d"):
        frame = pd.read_csv(analysis / f"worldspace_vs_registered_{mode}.csv")
        world_rows.append([
            mode.upper(), frame["feature"].nunique(), len(frame),
            f"{frame['mean_relative_difference'].median():.4f}",
        ])
    add_table(doc, ["Mode", "Feature sayısı", "Organ-feature satırı", "Median relative fark"], world_rows)
    doc.add_paragraph(
        "Identity/world-space kontrolündeki üç 20260402 pair’de spleen/aorta ROI’si "
        "fixed grid’e taşınmadığı için MIRP feature’ı üretilemedi; bu pair’ler B→C "
        "kontrolünden dışlandı. Bu nedenle bu aşamadaki world-space/registered "
        "karşılaştırma 85 ortak pair’i kapsar."
    )

    doc.add_heading("Tartışma", level=1)
    doc.add_paragraph(
        "V6, organ maskelerinin spatial overlap’ını kuvvetle artırdı. Aynı zamanda "
        "session-aware analiz, radyomik repeatability etkisinin feature-dependent "
        "olduğunu gösterdi. Registered ve identity/world-space arasındaki feature "
        "farkları resampling/interpolation etkisini de içerir. Dolayısıyla geometrik "
        "Dice iyileşmesi doğrudan radiomic reproducibility iyileşmesi olarak "
        "yorumlanmamalıdır. Özellikle GLCM/GLRLM gibi texture ailelerinde 2D/3D "
        "tanımları ve pooling yöntemleri ayrı değerlendirilmelidir."
    )

    doc.add_heading("Sınırlılıklar ve Sonraki İşler", level=1)
    for item in [
        "Reference tek bir seri (20260805 take1) olduğundan farklı reference/medoid stratejileri sensitivity analysis olarak test edilmelidir.",
        "Düşük Dice kalan pair’lerde görsel QC gereklidir; Dice segmentasyon değişkenliği ve FOV’dan da etkilenir.",
        "Aorta için aynı-oturum TotalSegmentator maksimum Dice 0.9892 olduğundan 0.99 katı eşiği ölçülebilir segmentation tavanıyla uyumsuzdur.",
        "Feature aileleri ve organlar ayrı raporlanmalı; near-zero/kararsız feature’lar relative fark analizinden çıkarılmalı veya özel işaretlenmelidir.",
        "Companion organ-coordinate yaklaşımıyla aynı pair’ler ve aynı MIRP ayarlarında karşılaştırma yapılmalıdır.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Sonuç", level=1)
    doc.add_paragraph(
        "V6’da rigid ve landmark/centroid tabanlı spatial standardization, "
        "longitudinal PhantomX organ maskelerinin geometrik uyumunu belirgin "
        "biçimde artırdı. Ancak radiomic repeatability’de kazanım tüm feature’lara "
        "yayılmadı; bazı feature ailelerinde iyileşme, bazılarında kötüleşme görüldü. "
        "Bu nedenle registration’ın gerekliliği organ ve feature ailesi bazında "
        "değerlendirilmeli; identity/world-space resampling kontrolü ve companion "
        "yöntem karşılaştırması tamamlanmadan tüm radiomics için genel üstünlük "
        "iddia edilmemelidir."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    print(f"Word report: {args.output}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate V6 academic DOCX report")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
