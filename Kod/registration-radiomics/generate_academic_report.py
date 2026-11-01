"""Build a Turkish academic DOCX report from the completed v1.0.0 analysis."""

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


def fmt_mm(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if not np.isfinite(number):
        return "NaN"
    return f"{number:.3f}"


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
    ax.bar(x + .18, after, .36, label="v1.0.0 sonrası", color="#2878B5")
    ax.set_xticks(x, [name for _, name in organs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ortalama Dice")
    ax.set_title("v1.0.0 registration öncesi/sonrası organ overlap")
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
    ax.set_title("v1.0.0’da kullanılan dönüşüm yöntemi")
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


def make_method_flowchart(path: Path) -> Path:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.2, 11.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EEF3FA", ec="#2F5496", fs=8.4):
        patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.02,rounding_size=0.18",
                               fc=fc, ec=ec, lw=1.4)
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def arrow(x1, y1, x2, y2, color="#2F5496"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                    shrinkA=2, shrinkB=2))

    cx, w = 3.7, 6.4
    box(cx, 11.2, w, 1.0, "Fixed referans (20260805 take1)  +  Moving seri")
    box(cx, 9.9, w, 1.2, "Stabil kemik maskesi\nomurga 27–43 · kaburga 92–116 · sternum 116\n(hedef organlar transform seçiminde kullanılmaz)")
    box(cx, 8.3, w, 0.9, "Organ centroid farkı ≤ skip eşiği ?", fc="#FFF4E5", ec="#D48A3A")
    box(8.4, 8.3, 2.6, 0.9, "Identity\n(hiç dokunma)", fc="#E8F5EC", ec="#3A9A5B")
    box(cx, 6.9, w, 0.9, "Başlangıç: kemik centroid ötelemesi (sadece öteleme)")
    box(cx, 5.5, w, 1.2, "Adaylar\n• Correlation rigid (6-DOF Euler, CT, kemik maskesi içinde)\n• Landmark rigid (Kabsch/SVD): vertebra-öncelikli + tam set")
    box(cx, 4.0, w, 0.9, "QC: stabil kemik Dice artmalı\ncentroid +0.5 mm · HD95 +1.0 mm · ASSD +0.25 mm toleransı")
    box(cx, 2.6, w, 1.1, "QC'yi geçen aday var mı?\nEvet → en yüksek kemik Dice'lı aday\nHayır → centroid öteleme → (o da geçmezse) identity", fc="#FFF4E5", ec="#D48A3A")
    box(cx, 1.0, w, 1.0, "Seçilen tek transformu uygula\nCT: lineer · maskeler: nearest-neighbor\n→ registered CT/seg + Dice/HD95/ASSD")

    arrow(cx, 10.7, cx, 10.5)
    arrow(cx, 9.3, cx, 8.75)
    arrow(cx, 7.85, cx, 7.35)
    arrow(cx, 6.45, cx, 6.1)
    arrow(cx, 4.9, cx, 4.45)
    arrow(cx, 3.55, cx, 3.15)
    arrow(cx, 2.05, cx, 1.5)
    arrow(cx + w / 2, 8.3, 7.1, 8.3, color="#3A9A5B")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


ORGAN_LABELS = {1: "spleen", 3: "spleen", 52: "aorta", 6: "aorta"}


def _metrics_frame(base: Path) -> pd.DataFrame:
    rows = []
    for f in base.glob("*/*/metrics.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["group"] = f.parent.parent.name
        rows.append(df[["group", "label", "before_dice", "after_dice"]])
    if not rows:
        return pd.DataFrame(columns=["group", "label", "before_dice", "after_dice", "organ"])
    df = pd.concat(rows, ignore_index=True)
    df["organ"] = df["label"].map(ORGAN_LABELS)
    return df[df["organ"].notna()]


def _pool(df: pd.DataFrame, organ: str):
    d = df[df["organ"] == organ]
    if d.empty:
        return float("nan"), float("nan"), 0
    return float(d["before_dice"].median()), float(d["after_dice"].median()), len(d)


def _cross_summary(base: Path) -> dict:
    rows = []
    for f in base.glob("*/metrics.csv"):
        try:
            rows.append(pd.read_csv(f))
        except Exception:
            continue
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    df["organ"] = df["label"].map(ORGAN_LABELS)
    df = df[df["organ"].notna()]
    out = {}
    for organ, g in df.groupby("organ"):
        out[organ] = (float(g["before_dice"].median()), float(g["after_dice"].median()), len(g))
    return out


def add_zurich_validation(doc: Document, pcct_work: Path, sep_work: Path, voi_csv: Path,
                          pcct_long: Path, sep_long: Path):
    pcct0 = _metrics_frame(pcct_work / "REGISTRATION")
    pcct5 = _metrics_frame(pcct_work / "REGISTRATION_skip5")
    sep0 = _metrics_frame(sep_work / "REGISTRATION")
    sep5 = _metrics_frame(sep_work / "REGISTRATION_skip5")

    doc.add_heading("Zurich Test-Retest Doğrulaması: Zorlamasız Registration, VOI Şekli ve Değerlendirici Etkisi", level=1)
    doc.add_paragraph(
        "Zurich test-retest verisinde (PCCT 8 protokol × 10 tekrar; SEP 5 protokol × 10–20 tekrar) "
        "organ centroid farkı tüm pair'lerde ≤0.94 mm idi. Bu nedenle 5 mm skip eşiği ile "
        "registration'ın identity'de kalması beklenir. İki koşu karşılaştırıldı: skip0 "
        "(registration zorlandı) ve skip5 (identity)."
    )
    rows = []
    for label, d0, d5 in (("PCCT", pcct0, pcct5), ("SEP", sep0, sep5)):
        for organ in ("spleen", "aorta"):
            b0, a0, _ = _pool(d0, organ)
            b5, a5, n5 = _pool(d5, organ)
            rows.append([
                label, organ.capitalize(),
                f"{b0:.4f}", f"{a0:.4f}", f"{a0 - b0:+.4f}",
                f"{b5:.4f}", f"{a5:.4f}", f"{a5 - b5:+.4f}", f"{n5}",
            ])
    add_table(doc, ["Veri seti", "Organ", "skip0 önce", "skip0 sonra", "skip0 Δ",
                    "skip5 önce", "skip5 sonra", "skip5 Δ", "N"], rows)
    doc.add_paragraph(
        "skip5 koşusunda tüm pair'ler identity'ye düştü ve before == after (Δ = 0.0000) oldu. "
        "skip0'da PCCT'te 25/72 pair landmark/centroid ile zorla register edilmişti; SEP'te "
        "yalnız 1/55 pair. Bu nedenle test-retest Dice düşüşü registration'ın değil, gereksiz "
        "resampling'in sonucudur."
    )
    doc.add_paragraph(
        "Öğrenci değerlendiricisinde Ref maskesi kırpılmış crop'a, Target ise tam fixed grid'e "
        "resample ediliyor; identity'de bile ~0.001–0.003 yapay fark oluşabiliyor. Ek olarak "
        "45 all-pairs medyanı ile 9 test-retest medyanı farklıdır. Aynı tam grid ve aynı pair "
        "seti kullanıldığında düşüş kaybolur; nitekim pilot metrics.csv değerleri skip5'te "
        "birebir eşittir."
    )

    voi_rows = []
    if voi_csv.exists():
        v = pd.read_csv(voi_csv)
        v["dataset"] = v["dataset"].astype(str).str.split(":").str[0]
        for (ds, organ, shape), g in v.groupby(["dataset", "organ", "shape"]):
            if ds not in ("zurich_pcct", "zurich_sep"):
                continue
            voi_rows.append([ds, organ, shape, f"{g['voi_dice'].median():.4f}",
                             f"{g['full_dice'].median():.4f}", len(g)])
    add_table(doc, ["Veri seti", "Organ", "VOI şekli", "VOI Dice", "Tam-organ Dice", "N"], voi_rows)
    doc.add_paragraph(
        "VOI organın içinde tanımlı olduğundan VOI Dice 0.99–1.00 aralığında doyuyor. Özellikle "
        "10 mm aort küresi tamamen organ içinde kaldığı için maskenin tamamı 1 oluyor ve Dice "
        "trivially 1.0 çıkıyor. Bu nedenle VOI Dice registration QC metriği değildir; VOI feature "
        "(HU/radiomics) stabilitesi için kullanılmalı, registration QC için tam-organ Dice "
        "kullanılmalıdır. Küre küpe göre daha homojen HU verir (SD daha düşük) ama Dice ayırt "
        "ediciliği yoktur."
    )

    pc = _cross_summary(pcct_long)
    se = _cross_summary(sep_long)
    rows = []
    for label, data in (("PCCT", pc), ("SEP", se)):
        for organ in ("spleen", "aorta"):
            if organ in data:
                b, a, n = data[organ]
                rows.append([label, organ.capitalize(), f"{b:.4f}", f"{a:.4f}",
                             f"{a - b:+.4f}", str(n)])
    add_table(doc, ["Veri seti", "Organ", "Önce (identity)", "Sonra (registered)", "Δ", "N"], rows)
    doc.add_paragraph(
        "Cross-date'te registration'ın faydası belirgindir: PCCT'te spleen 0.035→0.964, aorta "
        "0.126→0.931; SEP'te 0.000→0.962 ve 0.000→0.927. Test-retest'te registration gereksiz, "
        "cross-date'te zorunludur."
    )


def main(args: argparse.Namespace):
    reg_root = args.registration_root
    analysis = reg_root / "analysis"
    figure_dir = analysis / "report_figures"
    dice_fig, method_fig, family_fig, methods, dice_before, dice_after = make_figures(
        reg_root, analysis, figure_dir
    )

    n_pairs = 0
    ge90 = {"1": 0, "52": 0}
    for f in reg_root.glob("*/metrics.csv"):
        n_pairs += 1
        data = {row["label"]: row for row in csv.DictReader(f.open(encoding="utf-8"))}
        for lab in ("1", "52"):
            if float(data[lab]["after_dice"]) >= 0.90:
                ge90[lab] += 1

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
    subtitle.add_run("v1.1.0 analiz raporu | 2 Kasım 2026").italic = True
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
        "feature ailesine göre değişti; registration bütün feature’ları iyileştirmedi. "
        "Zurich test-retest doğrulamasında 5 mm skip eşiği ile tüm pair’ler identity’de "
        "kaldı ve Dice değişmedi; cross-date’te ise registration Dice’ı 0.03–0.13’ten "
        "0.93–0.96’ya çıkardı."
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
        "Aday dönüşümün kabulünde stabil Dice'ın artması ve centroid displacement, "
        "HD95, ASSD değerlerinin tanımlı toleranslarda kalması şart koşuldu. Böylece "
        "hedef organ maskeleri registration'ı seçmek için kullanılmadı."
    )
    flowchart_fig = make_method_flowchart(figure_dir / "method_flowchart.png")
    doc.add_picture(str(flowchart_fig), width=Inches(6.2))
    add_caption(
        doc,
        "Şekil. Registration yöntem akışı: stabil kemik maskesi, centroid ötelemesi "
        "(başlangıç), correlation/landmark rigid adayları, kemik-Dice QC ve en iyi aday seçimi.",
    )
    doc.add_paragraph(
        "Akış özeti: (1) stabil kemik maskesi çıkarılır; (2) organ centroid farkı eşiğin "
        "altındaysa identity; (3) aksi halde kemik centroid ötelemesi başlangıç alınır; "
        "(4) correlation rigid ve landmark rigid (Kabsch/SVD) adayları üretilir; "
        "(5) her aday kemik Dice ve centroid/HD95/ASSD toleransıyla QC edilir; "
        "(6) geçen adaylar arasından en yüksek kemik Dice'lısı seçilir; "
        "(7) hiçbiri geçmezse centroid öteleme, o da geçmezse identity uygulanır."
    )

    doc.add_heading("Dice Eşiği ve Yorum", level=1)
    registration_base = reg_root.parent
    ceiling_path = registration_base / "segmentation_ceiling.csv"
    if ceiling_path.exists():
        ceiling = list(csv.DictReader(ceiling_path.open(encoding="utf-8")))
        ceilings = {}
        for organ in ("spleen", "aorta"):
            values = [float(row["dice"]) for row in ceiling if row["organ"] == organ]
            ceilings[organ] = (statistics.median(values), max(values))
        add_table(doc, ["Organ", "Aynı oturum median Dice", "Gözlenen maksimum Dice", "v1.0.0 ortalama final Dice"], [
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
    else:
        doc.add_paragraph(
            "Bu veri seti için segmentasyon tavanı (segmentation_ceiling.csv) henüz "
            "üretilmedi; Dice eşiği yorumu yalnız gözlenen registration sonuçlarına dayanır."
        )

    doc.add_heading("Registration Sonuçları", level=1)
    add_table(doc, ["Ölçüm", "Spleen", "Aorta"], [
        ["Ortalama Dice, öncesi", f"{dice_before[0]:.4f}", f"{dice_before[1]:.4f}"],
        ["Ortalama Dice, v1.0.0 sonrası", f"{dice_after[0]:.4f}", f"{dice_after[1]:.4f}"],
        ["Dice ≥ 0.90 pair sayısı", f"{ge90['1']} / {n_pairs}", f"{ge90['52']} / {n_pairs}"],
        ["Dice = 0 pair sayısı", "0", "0"],
    ])
    add_table(doc, ["v1.0.0 dönüşüm yöntemi", "Pair sayısı"], [
        [method, count] for method, count in methods.items()
    ])
    doc.add_picture(str(dice_fig), width=Inches(6.4))
    add_caption(doc, "Şekil 1. v1.0.0 registration öncesi ve sonrası ortalama organ Dice değerleri.")
    doc.add_picture(str(method_fig), width=Inches(6.4))
    add_caption(doc, "Şekil 2. v1.0.0’da pair başına kullanılan dönüşüm yöntemi.")

    doc.add_heading("Dönüşüm Bileşenlerinin Ayrıştırılması: Identity, Translation, Rotation ve Registration", level=1)
    doc.add_paragraph(
        "Organ kaymasının hangi geometrik bileşenden kaynaklandığını ayırmak için, "
        "kohortun en büyük kaymasına sahip pair’lerinden biri olan 20260304 (yaklaşık "
        "70 mm masa ötelemesi) üzerinde izole dönüşüm deneyi yapıldı. Aynı moving CT ve "
        "maskesi sırasıyla identity (dönüşümsüz), yalnız translasyon (centroid), yalnız "
        "rotasyon ve geliştirilen landmark rigid (Kabsch/SVD) registration ile fixed "
        "grid’e taşındı; her adımda Dice, centroid displacement, HD95 ve ASSD ölçüldü."
    )
    transform_metrics_path = registration_base / "transform_experiment_20260304" / "metrics.csv"
    landmark_metrics_path = registration_base / "experiment_20260304_landmark" / "metrics.csv"
    if transform_metrics_path.exists() and landmark_metrics_path.exists():
        tdata = pd.read_csv(transform_metrics_path)
        ldata = pd.read_csv(landmark_metrics_path)
        order = ["identity", "translation", "rotation_only"]
        labels_tr = {
            "identity": "Identity",
            "translation": "Translation-only",
            "rotation_only": "Rotation-only",
            "landmark": "Landmark rigid",
        }
        table_rows = []
        for name in order:
            for organ_label, organ_name in ((1, "Spleen"), (52, "Aorta")):
                row = tdata[(tdata["transform"] == name) & (tdata["label"] == organ_label)]
                if row.empty:
                    continue
                r = row.iloc[0]
                table_rows.append([
                    labels_tr[name], organ_name, f"{r['dice']:.4f}",
                    fmt_mm(r["centroid_displacement_mm"]),
                    fmt_mm(r["hd95_mm"]), fmt_mm(r["assd_mm"]),
                ])
        for organ_label, organ_name in ((1, "Spleen"), (52, "Aorta")):
            row = ldata[ldata["label"] == organ_label]
            if row.empty:
                continue
            r = row.iloc[0]
            table_rows.append([
                labels_tr["landmark"], organ_name, f"{r['after_dice']:.4f}",
                fmt_mm(r["after_centroid_displacement_mm"]),
                fmt_mm(r["after_hd95_mm"]), fmt_mm(r["after_assd_mm"]),
            ])
        add_table(
            doc,
            ["Dönüşüm", "Organ", "Dice", "Centroid (mm)", "HD95 (mm)", "ASSD (mm)"],
            table_rows,
        )

        organ_names = ["Spleen", "Aorta"]
        organ_colors = ["#2878B5", "#D48A3A"]
        positions = np.arange(len(order) + 1)
        bar_width = 0.38
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for j, organ_label in enumerate((1, 52)):
            values = []
            for name in order:
                row = tdata[(tdata["transform"] == name) & (tdata["label"] == organ_label)]
                values.append(float(row.iloc[0]["dice"]) if not row.empty else 0.0)
            lrow = ldata[ldata["label"] == organ_label]
            values.append(float(lrow.iloc[0]["after_dice"]) if not lrow.empty else 0.0)
            ax.bar(positions + (j - 0.5) * bar_width, values, bar_width,
                   label=organ_names[j], color=organ_colors[j])
        ax.set_xticks(positions, ["Identity", "Translation-only", "Rotation-only", "Landmark rigid"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Dice")
        ax.set_title("20260304 pair’inde dönüşüm bileşenlerine göre organ Dice")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        transform_fig = figure_dir / "transform_components_dice.png"
        fig.savefig(transform_fig, dpi=180)
        plt.close(fig)
        doc.add_picture(str(transform_fig), width=Inches(6.4))
        add_caption(doc, "Şekil 3. Identity, translation-only, rotation-only ve landmark rigid registration sonrası spleen/aorta Dice değerleri.")

        doc.add_paragraph(
            "Sonuçlar nedenselliği netleştirir: identity durumunda organlar tamamen "
            "ayrık kalır (Dice 0.00). Yalnız rotasyon uygulamak, öteleme telafi "
            "edilmediğinden kaldıraç kolu etkisiyle centroid kaymasını dalakta "
            "69.8 mm’den 163.5 mm’ye çıkarır ve Dice’ı 0.00’da bırakır. Yalnız "
            "translasyon ise Dice’ı dalakta 0.930, aortada 0.922 seviyesine taşır; bu, "
            "longitudinal konum sapmasının büyük kısmının saf öteleme kaynaklı olduğunu "
            "gösterir. Klasik yoğunluk tabanlı rigid optimizer bu büyüklükteki kaymada "
            "ıraksayarak geçersiz sonuç (NaN) üretir. Geliştirilen landmark rigid "
            "registration ise rezidüel rotasyonu da düzelterek dalak Dice’ını 0.964’e, "
            "aorta Dice’ını 0.925’e yükseltir ve centroid kaymasını 0.19 mm (dalak) ve "
            "0.41 mm (aorta) düzeyine, yani milimetre altına indirir."
        )
    else:
        doc.add_paragraph(
            "Dönüşüm bileşeni deney çıktıları bulunamadı; bu bölüm atlandı."
        )

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

    ci_path = analysis / "repeatability_confidence_intervals_v6.csv"
    if ci_path.exists():
        ci_data = pd.read_csv(ci_path)
        ci_rows = []
        for mode in ("3d", "2d"):
            for metric, metric_label in (("coefficient_of_variation", "CV"),
                                         ("median_pairwise_relative_difference", "Pairwise relative difference")):
                subset = ci_data[(ci_data["mode"] == mode) & (ci_data["metric"] == metric)]
                counts = subset["direction"].value_counts()
                ci_rows.append([
                    mode.upper(), metric_label, len(subset),
                    int(counts.get("improved", 0)),
                    int(counts.get("worsened", 0)),
                    int(counts.get("uncertain", 0)),
                ])
        doc.add_paragraph(
            "Belirsizlik için oturum eşleşmeleri 5,000 tekrarlı paired bootstrap ile "
            "örneklendi. ‘Improved’ 95% CI’nin sıfırın tamamen altında; ‘worsened’ "
            "tamamen üstünde; ‘uncertain’ ise sıfırı içerdiği anlamına gelir."
        )
        add_table(doc, ["Mode", "Ölçü", "Test", "CI iyileşme", "CI kötüleşme", "Belirsiz"], ci_rows)

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

    doc.add_heading("Tüm-Çift Pairwise Raporu ve Analitik Sayfa", level=1)
    doc.add_paragraph(
        "Organ maskelerinin yalnızca tek bir fixed referansa göre değil, tüm seri çiftleri "
        "arasında da karşılaştırılabilmesi için SEP ve PCCT veri setleri için all-pairs "
        "pairwise raporları üretildi. Aynı oturum çiftleri kendi oturum referans grid'inde "
        "identity ile (zaten hizalı oldukları için), farklı oturum çiftleri ise fixed-referans "
        "registration sonrası karşılaştırıldı. Her çift için Dice, ASSD, HD95 ve clDice "
        "hesaplandı; sonuçlar Summary, Pairs, Improvements, Date_Summary ve Dice_Diagnostics "
        "sayfalarıyla Excel'e yazıldı."
    )
    add_table(doc, ["Veri seti", "Seri", "Pair", "Aynı oturum", "Farklı oturum"], [
        ["SEP", "68", "2278", "389", "1889"],
        ["PCCT", "89", "3916", "186", "3730"],
    ])
    doc.add_paragraph(
        "Aynı oturum içindeki tekrarlar yüksek Dice verdi (identity); farklı oturumlarda "
        "rigid/landmark registration Dice'ı belirgin biçimde yükseltti. Düşük Dice'lı seriler "
        "dışlanmadı, gerekçeleriyle notlandı: SEP'te 20260731_t20 (zayıf rigid hizalama, "
        "HD95 4–5 mm) ve 20260128_t1 (identity; segmentasyon/şekil), PCCT'te 20260601_t1 "
        "(HD95 2.0 mm, aorta sınırlayıcı) ve 20260622_t3 (ince kemik kabuğu/şekil farkı). "
        "20260622 _s1005 ise _s2087 ile aynı ışınlamadır; Dice'ı iyi olduğu için kalite "
        "nedeniyle değil, yalnızca çift sayımı kaygısıyla ayrıca belirtilmiştir."
    )
    doc.add_paragraph(
        "Etkileşimli analitik sayfa; tüm-çift Dice matrisini (mutlak Dice ve kazanç modları), "
        "Dice dağılımını, tarih-çifti kazanç sıralamasını ve düşük Dice tanılamasını içerir: "
        "https://erayakgunn.github.io/registration-overlay/analytics.html"
    )

    add_zurich_validation(doc, args.zurich_pcct, args.zurich_sep, args.voi_shapes,
                          reg_root, args.sep_longitudinal)

    doc.add_heading("Tartışma", level=1)
    doc.add_paragraph(
        "v1.0.0, organ maskelerinin spatial overlap’ını kuvvetle artırdı. Aynı zamanda "
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
        "Companion organ-coordinate yönteminin kodu/çıktıları bu paket içinde bulunmadığından, aynı pair ve MIRP ayarlarında doğrudan karşılaştırma henüz yapılamadı.",
        "Test-retest’te registration zorlanmamalıdır (5 mm skip, identity); VOI Dice registration QC metriği değildir, registration QC için tam-organ Dice kullanılmalıdır.",
        "Değerlendirici karşılaştırmalarında Ref ve Target aynı tam fixed grid üzerinde hesaplanmalı; crop/full-grid karışımı identity’de bile yapay fark üretir.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Sonuç", level=1)
    doc.add_paragraph(
        "v1.0.0’da rigid ve landmark/centroid tabanlı spatial standardization, "
        "longitudinal PhantomX organ maskelerinin geometrik uyumunu belirgin "
        "biçimde artırdı. Ancak radiomic repeatability’de kazanım tüm feature’lara "
        "yayılmadı; bazı feature ailelerinde iyileşme, bazılarında kötüleşme görüldü. "
        "Bu nedenle registration’ın gerekliliği organ ve feature ailesi bazında "
        "değerlendirilmeli; identity/world-space resampling kontrolü ve companion "
        "yöntem karşılaştırması tamamlanmadan tüm radiomics için genel üstünlük "
        "iddia edilmemelidir. Zurich test-retest doğrulaması bu ayrımı destekler: "
        "zaten hizalı veride identity en iyi sonucu verirken, cross-date’te registration "
        "zorunludur; VOI ise feature stabilitesi için tanımlıdır, Dice/registration QC "
        "metriği değildir."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    print(f"Word report: {args.output}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate v1.1.0 academic DOCX report")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zurich-pcct", type=Path,
                    default=Path(r"C:/Users/eray/Desktop/zurich_pcct_work"))
    ap.add_argument("--zurich-sep", type=Path,
                    default=Path(r"C:/Users/eray/Desktop/zurich_sep_work"))
    ap.add_argument("--voi-shapes", type=Path,
                    default=Path(r"C:/Users/eray/Desktop/zurich_pcct_work/VOI_DICE_SUMMARY_shapes.csv"))
    ap.add_argument("--sep-longitudinal", type=Path,
                    default=Path(r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/PhantomX/REGISTRATION/batch_to_20260805_v6"))
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
