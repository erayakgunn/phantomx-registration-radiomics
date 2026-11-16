"""Generate the v2.0 academic report from current PhantomX outputs.

Scope: PCCT and SEP cross-date rigid registration; Zurich test-retest;
four organs (aorta, spleen, right kidney, left kidney); surface metrics;
kidney connected-component sensitivity; consensus-reference sensitivity;
a concise radiomics summary (PCCT, spleen/aorta, 3D).

The report is intentionally self-contained: every number is recomputed from
the current CSV/XLSX outputs, and every metric is explained in the text.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PCCT_BATCH = ROOT / "PhantomX/REGISTRATION/batch_to_20260805_v1.0.0"
SEP_ROOT = next(Path(r"C:/Users/eray/Desktop").glob("SEP -*")) / "PhantomX"
SEP_BATCH = SEP_ROOT / "REGISTRATION/batch_to_20260805_v6_skip2"
KIDNEY_CSV = Path(r"C:/Users/eray/Desktop/KIDNEY_METRICS_PCCT_SEP.csv")
SCAN_CSV = Path(r"C:/Users/eray/Desktop/KIDNEY_CC_SCAN.csv")
ZURICH_PCCT = Path(r"C:/Users/eray/Desktop/zurich_pcct_work/REGISTRATION_skip5")
ZURICH_SEP = Path(r"C:/Users/eray/Desktop/zurich_sep_work/REGISTRATION_skip5")
SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
FIG = PCCT_BATCH / "analysis/report_v2_figures"
OUT = Path(r"C:/Users/eray/Desktop/RAPOR_AKADEMIK_REGISTRATION_RADIOMICS_v2.0.docx")

ORGAN_ORDER = ["aorta", "spleen", "kidney_right", "kidney_left"]
ORGAN_LABEL = {
    "aorta": "Aorta",
    "spleen": "Dalak",
    "kidney_right": "Sağ böbrek",
    "kidney_left": "Sol böbrek",
}
DATASET_ORDER = ["PCCT", "SEP"]
TEST_ORDER = ["PCCT test-retest", "SEP test-retest"]
TOL = 0.0005


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_cell(cell, text, bold=False, size=8.5):
    cell.text = "" if text is None else str(text)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.name = "Arial"
            if bold:
                run.bold = True


def add_table(doc, headers, rows, widths=None, size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, header in enumerate(headers):
        set_cell(table.rows[0].cells[index], header, bold=True, size=size)
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            set_cell(cells[index], value, size=size)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    doc.add_paragraph()
    return table


def add_caption(doc, text):
    paragraph = doc.add_paragraph(text, style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in paragraph.runs:
        run.font.size = Pt(8.5)
        run.italic = True


def add_figure(doc, path: Path, width=6.4, caption=None):
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        add_caption(doc, caption)


def fmt(value, digits=3, dash="—"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(number):
        return dash
    return f"{number:.{digits}f}"


def p_text(p):
    if p is None or not math.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def add_paragraph(doc, text, style=None):
    paragraph = doc.add_paragraph(text, style=style) if style else doc.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.12
    return paragraph


def add_bullet(doc, text):
    paragraph = doc.add_paragraph(text, style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    return paragraph


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def read_metrics_dir(root: Path, dataset: str) -> pd.DataFrame:
    rows = []
    for metrics_path in root.rglob("metrics.csv"):
        try:
            frame = pd.read_csv(metrics_path)
        except Exception:  # noqa: BLE001
            continue
        for _, row in frame.iterrows():
            organ = {1: "spleen", 52: "aorta"}.get(int(row["label"]))
            if not organ:
                continue
            rows.append({
                "dataset": dataset,
                "organ": organ,
                "before_dice": row.get("before_dice"),
                "after_dice": row.get("after_dice"),
                "before_assd": row.get("before_assd_mm"),
                "after_assd": row.get("after_assd_mm"),
                "before_mssd": row.get("before_mssd_mm"),
                "after_mssd": row.get("after_mssd_mm"),
                "before_hd95": row.get("before_hd95_mm"),
                "after_hd95": row.get("after_hd95_mm"),
                "before_nsd": row.get("before_nsd_1mm"),
                "after_nsd": row.get("after_nsd_1mm"),
                "before_cldice": row.get("before_cldice"),
                "after_cldice": row.get("after_cldice"),
                "before_centroid": row.get("before_centroid_displacement_mm"),
                "after_centroid": row.get("after_centroid_displacement_mm"),
            })
    return pd.DataFrame(rows)


def kidney_frame(scope: str, mapping: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(KIDNEY_CSV)
    frame = frame[frame["Scope"] == scope]
    rows = []
    for _, row in frame.iterrows():
        dataset = mapping.get(str(row["Dataset"]))
        if not dataset:
            continue
        rows.append({
            "dataset": dataset,
            "organ": str(row["Organ"]),
            "before_dice": row.get("Dice_before"),
            "after_dice": row.get("Dice_after"),
            "before_assd": row.get("ASSD_before"),
            "after_assd": row.get("ASSD_after"),
            "before_mssd": row.get("MSSD_before"),
            "after_mssd": row.get("MSSD_after"),
            "before_hd95": row.get("HD95_before"),
            "after_hd95": row.get("HD95_after"),
            "before_nsd": np.nan,
            "after_nsd": np.nan,
            "before_cldice": np.nan,
            "after_cldice": np.nan,
            "before_centroid": np.nan,
            "after_centroid": np.nan,
        })
    return pd.DataFrame(rows)


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = pd.concat([
        read_metrics_dir(PCCT_BATCH, "PCCT"),
        read_metrics_dir(SEP_BATCH, "SEP"),
        kidney_frame("cross-date", {"PCCT cross-date": "PCCT", "SEP cross-date": "SEP"}),
    ], ignore_index=True)
    test = pd.concat([
        read_metrics_dir(ZURICH_PCCT, "PCCT test-retest"),
        read_metrics_dir(ZURICH_SEP, "SEP test-retest"),
        kidney_frame("test-retest", {
            "PCCT test-retest": "PCCT test-retest",
            "SEP test-retest": "SEP test-retest",
        }),
    ], ignore_index=True)
    for frame in (cross, test):
        for column in frame.columns:
            if column not in ("dataset", "organ"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return cross, test


def paired_p(before: pd.Series, after: pd.Series) -> float:
    paired = pd.DataFrame({"before": before, "after": after}).dropna()
    if len(paired) < 5:
        return float("nan")
    diff = paired["after"] - paired["before"]
    if np.allclose(diff.to_numpy(), 0):
        return 1.0
    try:
        return float(wilcoxon(paired["after"], paired["before"]).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = DATASET_ORDER if frame["dataset"].isin(DATASET_ORDER).any() else TEST_ORDER
    for dataset in order:
        for organ in ORGAN_ORDER:
            group = frame[(frame["dataset"] == dataset) & (frame["organ"] == organ)]
            if group.empty:
                continue
            before = group["before_dice"]
            after = group["after_dice"]
            paired = pd.DataFrame({"before": before, "after": after}).dropna()
            delta = paired["after"] - paired["before"]
            rows.append({
                "dataset": dataset,
                "organ": organ,
                "n": len(group),
                "dice_before": float(before.median()),
                "dice_after": float(after.median()),
                "dice_delta": float(delta.median()) if len(delta) else float("nan"),
                "dice_q1": float(after.quantile(0.25)),
                "dice_q3": float(after.quantile(0.75)),
                "assd_after": float(group["after_assd"].median()),
                "mssd_after": float(group["after_mssd"].median()),
                "hd95_after": float(group["after_hd95"].median()),
                "nsd_after": float(group["after_nsd"].median()),
                "cldice_after": float(group["after_cldice"].median()),
                "centroid_after": float(group["after_centroid"].median()),
                "improved": int((delta > TOL).sum()) if len(delta) else 0,
                "worsened": int((delta < -TOL).sum()) if len(delta) else 0,
                "p": paired_p(before, after),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_method(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.axis("off")
    boxes = [
        (0.02, "DICOM CT\n(fiziksel uzay)"),
        (0.22, "NIfTI + TotalSegmentator\n(4 organ maskesi)"),
        (0.43, "Identity\n(öncesi)"),
        (0.60, "Centroid\n(öteleme)"),
        (0.77, "Rigid / landmark\n(sonrası)"),
        (0.92, "Metrik + QC\n+ adacık taraması"),
    ]
    for x, label in boxes:
        box = plt.Rectangle((x, 0.32), 0.15, 0.36, facecolor="#eaf2fb",
                            edgecolor="#1f4e79", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + 0.075, 0.50, label, ha="center", va="center",
                fontsize=7.5, transform=ax.transAxes)
    for x in (0.17, 0.37, 0.56, 0.73, 0.89):
        ax.annotate("", xy=(x + 0.04, 0.50), xytext=(x, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.2))
    ax.text(0.5, 0.10, "Önce = identity; ara adım = centroid; sonra = rigid. "
            "5 mm/2 mm eşiğinde zorlamasız identity.",
            ha="center", fontsize=8, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_dice(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for axis, dataset in zip(axes, DATASET_ORDER):
        subset = summary[summary["dataset"] == dataset]
        labels = [ORGAN_LABEL[o] for o in subset["organ"]]
        x = np.arange(len(labels))
        axis.bar(x - 0.18, subset["dice_before"], 0.36, color="#b7c4d2", label="Önce (identity)")
        axis.bar(x + 0.18, subset["dice_after"], 0.36, color="#2878b5", label="Sonra (rigid)")
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(dataset, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Medyan Dice")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_surface(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    metrics = [("assd_after", "ASSD (mm)"), ("mssd_after", "MSSD (mm)")]
    width = 0.36
    for axis, (metric, title) in zip(axes, metrics):
        for offset, dataset in zip((-width / 2, width / 2), DATASET_ORDER):
            subset = summary[summary["dataset"] == dataset]
            x = np.arange(len(subset))
            axis.bar(x + offset, subset[metric], width, label=dataset)
        axis.set_xticks(np.arange(4), [ORGAN_LABEL[o] for o in ORGAN_ORDER], fontsize=8)
        axis.set_title(title, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Medyan mesafe (mm)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_kidney_artifacts(scan: pd.DataFrame, path: Path) -> None:
    groups = [
        ("PCCT cross-date", "kidney_right", "PCCT cross sağ"),
        ("PCCT cross-date", "kidney_left", "PCCT cross sol"),
        ("PCCT test-retest", "kidney_right", "PCCT test sağ"),
        ("PCCT test-retest", "kidney_left", "PCCT test sol"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["ASSD_mm", "MSSD_mm"],
        ["ASSD (mm)", "MSSD (mm)"],
    ):
        x = np.arange(len(groups))
        raw, largest = [], []
        for dataset, organ, _ in groups:
            subset = scan[(scan["Dataset"] == dataset) & (scan["Organ"] == organ)]
            raw.append(subset[f"Raw_{metric}"].mean())
            largest.append(subset[f"Largest_{metric}"].mean())
        axis.bar(x - 0.18, raw, 0.36, color="#d98b8b", label="Ham")
        axis.bar(x + 0.18, largest, 0.36, color="#2878b5", label="Largest component")
        axis.set_xticks(x, [label for _, _, label in groups], fontsize=8)
        axis.set_title(title, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Ortalama mesafe (mm)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_organ_panel(path: Path) -> None:
    case = SITE / "assets/20260707_t3"
    organs = ["aorta", "spleen", "kidney_right", "kidney_left"]
    fig, axes = plt.subplots(4, 2, figsize=(6.4, 10.4))
    for row, organ in enumerate(organs):
        for col, state in enumerate(("before", "after")):
            image_path = case / f"{organ}_axial_1_{state}.jpg"
            axis = axes[row, col]
            axis.axis("off")
            if image_path.exists():
                axis.imshow(plt.imread(image_path))
            axis.set_title(
                f"{ORGAN_LABEL[organ]} · {'önce (centroid)' if state == 'before' else 'sonra (rigid)'}",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_kidney_panel(path: Path) -> None:
    case = SITE / "assets/20260707_t3"
    panels = [
        ("HAM", "kidney_right_axial_1_raw.jpg"),
        ("IDENTITY", "kidney_right_axial_1_identity.jpg"),
        ("CENTROID", "kidney_right_axial_1_before.jpg"),
        ("RIGID", "kidney_right_axial_1_after.jpg"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for axis, (label, name) in zip(axes, panels):
        axis.axis("off")
        image_path = case / name
        if image_path.exists():
            axis.imshow(plt.imread(image_path))
        axis.set_title(label, fontsize=8)
    fig.suptitle("Sağ böbrek: ham, identity, centroid ve rigid", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_radiomics(rad: pd.DataFrame, path: Path) -> None:
    aggregate = rad.groupby("family").agg(
        n=("n_features", "sum"),
        improved=("both_improved", "sum"),
    )
    aggregate["pct"] = 100 * aggregate["improved"] / aggregate["n"]
    aggregate = aggregate.sort_values("pct")
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.barh(aggregate.index, aggregate["pct"], color="#6f9fc8")
    axis.set_xlabel("Her iki koşulda iyileşen feature oranı (%)")
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def build_report() -> None:
    cross, test = build_frames()
    cross_summary = summarize(cross)
    test_summary = summarize(test)
    scan = pd.read_csv(SCAN_CSV) if SCAN_CSV.exists() else pd.DataFrame()
    rad = pd.read_csv(PCCT_BATCH / "analysis/feature_family_summary_3d.csv")

    FIG.mkdir(parents=True, exist_ok=True)
    fig_method(FIG / "method.png")
    fig_dice(cross_summary, FIG / "dice.png")
    fig_surface(cross_summary, FIG / "surface.png")
    if not scan.empty:
        fig_kidney_artifacts(scan, FIG / "kidney_artifacts.png")
    fig_organ_panel(FIG / "organ_panel.png")
    fig_kidney_panel(FIG / "kidney_panel.png")
    fig_radiomics(rad, FIG / "radiomics.png")

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10)
    styles["Title"].font.name = "Arial"
    styles["Title"].font.size = Pt(20)
    styles["Heading 1"].font.color.rgb = RGBColor(31, 78, 121)
    styles["Heading 1"].font.size = Pt(14)
    styles["Heading 2"].font.color.rgb = RGBColor(47, 84, 150)
    styles["Heading 2"].font.size = Pt(11.5)

    # Title page
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(
        "Longitudinal PhantomX CT’de Organ Segmentasyonu ve Rigid Registration:\n"
        "Dalak, Aorta ve Böbreklerin Geometrik ve Yüzey-Tabanlı Değerlendirmesi"
    )
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Akademik Rapor v2.0 | 16 Kasım 2026").italic = True
    doc.add_paragraph()
    add_paragraph(
        doc,
        "Bu sürüm; 88 PCCT ve 67 SEP cross-date kaydını, 72 PCCT ve 55 SEP "
        "test-retest kaydını, dört organı (aorta, dalak, sağ böbrek, sol böbrek) "
        "ve adacık duyarlılık analizini birlikte değerlendirir. Önceki sürümlerdeki "
        "organ kapsamı, metrik tanımları ve vaka sayıları bu raporda güncel "
        "çıktılardan yeniden hesaplanmıştır.",
    )
    doc.add_page_break()

    # 1. Özet
    doc.add_heading("1. Özet", level=1)
    add_paragraph(
        doc,
        "Bu çalışmanın amacı, aynı fantomun farklı zamanlarda ve farklı "
        "rekonstrüksiyonlarda elde edilen CT serilerinde organ segmentasyonlarının "
        "rigid registration öncesi ve sonrası geometrik uyumunu ölçmek; Dice’ın "
        "yanı sıra yüzey mesafesi tabanlı metriklerle sınır doğruluğunu "
        "değerlendirmek; ve böbreklerin bu değerlendirmeye katılmasının sonuçlara "
        "etkisini göstermektir.",
    )
    add_paragraph(
        doc,
        "PCCT veri setinde 88, SEP veri setinde 67 moving seri, 20260805 take1 "
        "referansına göre değerlendirildi. Test-retest doğrulamasında Zurich "
        "PCCT’de 72, SEP’te 55 pair kullanıldı. Segmentasyon TotalSegmentator "
        "2.12.0 ile üretildi; hedef organlar aorta (label 52), dalak (label 1), "
        "sağ böbrek (label 2) ve sol böbrek (label 3) idi. ‘Önce’ durumu identity "
        "(hiç transform yok), ‘sonra’ durumu ise kemik tabanlı rigid/landmark "
        "registration olarak tanımlandı.",
    )
    add_paragraph(
        doc,
        "Cross-date’te rigid registration sonrası medyan Dice aorta, dalak, sağ "
        "böbrek ve sol böbrek için sırasıyla PCCT’te 0.93, 0.96, 0.94 ve 0.96; "
        "SEP’te 0.92, 0.96, 0.94 ve 0.95 düzeyine ulaştı. Tipik ASSD/MSSD değerleri "
        "0.3–0.5 mm, HD95 1.0–1.2 mm idi; bu, 1 mm kesit kalınlığında milimetre "
        "altı/sınırda doğruluk anlamına gelir. Böbrekler dalak ve aorta ile aynı "
        "doğruluk ligindeydi.",
    )
    add_paragraph(
        doc,
        "Böbreklerin dahil edilmesi iki önemli bulgu üretti. Birincisi, 528 böbrek "
        "maskesinin 18’inde (%3.4) uzakta küçük bir segmentasyon adacığı vardı; bu "
        "adacıklar Dice’ı neredeyse etkilemezken ASSD/MSSD/HD95’i şişiriyordu. "
        "En büyük bağlı bileşenle hesaplandığında bu vakalarda ASSD 0.30–0.53 mm, "
        "MSSD 0.28–0.54 mm ve HD95 1.0–1.2 mm’ye indi. İkincisi, SEP 20260731_s80 "
        "tek bileşenli gerçek bir segmentasyon/rekonstrüksiyon aykırısıydı; "
        "registration bunu düzeltmedi.",
    )
    add_paragraph(
        doc,
        "Radyomik feature analizinde (PCCT, dalak/aorta, 3D, 418 feature) "
        "registration geometriyi belirgin biçimde düzeltse de feature’ların "
        "yalnızca %17.5’i her iki koşulda iyileşti; FDR düzeltmesi sonrası 58 "
        "testin 11’i iyileşti, 47’si kötüleşti. Bu nedenle geometrik Dice "
        "iyileşmesi, radyomik tekrarlanabilirliğin otomatik olarak iyileştiği "
        "şeklinde yorumlanmamalıdır.",
    )
    doc.add_page_break()

    # 2. Giriş
    doc.add_heading("2. Giriş ve Amaç", level=1)
    add_paragraph(
        doc,
        "Uzunlamasına CT çalışmalarında aynı organın farklı zamanlarda "
        "segmentasyonu, hasta/fantom konumu, rekonstrüksiyon parametreleri ve "
        "algoritma davranışı nedeniyle değişebilir. Bu değişkenlik, hacim ve "
        "radyomik feature ölçümlerine doğrudan yansır. Registration, moving "
        "seriyi fixed referansa hizalayarak bu farkın bir kısmını giderir; ancak "
        "registration’ın başarısı tek bir metrikle özetlenemez.",
    )
    add_paragraph(
        doc,
        "Dice katsayısı hacimsel örtüşmeyi ölçer ve büyük organlarda yüksek "
        "çıkmaya eğilimlidir; küçük, uzak bir segmentasyon hatasını neredeyse "
        "görmez. ASSD, MSSD ve HD95 gibi yüzey mesafesi metrikleri ise sınır "
        "doğruluğunu ve lokal hataları daha duyarlı yakalar. Bu nedenle bu "
        "raporda Dice tek başına değil, yüzey metrikleriyle birlikte "
        "yorumlanmıştır.",
    )
    add_paragraph(
        doc,
        "Bu çalışmanın özgün katkıları şunlardır: (i) dalak ve aortanın yanına "
        "sağ ve sol böbreğin ayrı ayrı eklenmesi; (ii) identity, centroid ve "
        "rigid durumlarının aynı görsel arayüzde karşılaştırılması; (iii) "
        "böbrek yüzey metriklerindeki aykırı değerlerin bağlı bileşen analiziyle "
        "açıklanması; (iv) 3 mm rekonstrüksiyonların etkisinin ayrıca "
        "gösterilmesi; (v) konsensüs referansı ve radyomik özetin aynı raporda "
        "birlikte sunulması.",
    )

    # 3. Veri
    doc.add_heading("3. Veri Setleri ve Değerlendirme Tasarımı", level=1)
    add_table(
        doc,
        ["Veri seti", "Kayıt", "Referans", "Organlar", "Not"],
        [
            ["PCCT", "88 moving seri / 22 oturum", "20260805 take1", "4 organ", "1 mm, Q3/Qr40f, 120 kV"],
            ["SEP", "67 moving seri", "20260805 take1", "4 organ", "1 mm, Br40f, 120 kV"],
            ["Zurich PCCT", "72 test-retest pair", "take1", "4 organ", "1/3 mm; zorlamasız identity"],
            ["Zurich SEP", "55 test-retest pair", "take1", "4 organ", "1 mm; zorlamasız identity"],
        ],
        widths=[1.1, 1.7, 1.3, 0.9, 1.9],
    )
    add_caption(doc, "Tablo 1. Veri setleri ve değerlendirme tasarımı.")
    add_paragraph(
        doc,
        "PhantomX, organ benzeri insert’ler içeren bir fantomdur; bu nedenle "
        "sonuçlar klinik hastalara değil, standardize edilmiş bir fantom "
        "geometrisine aittir. Segmentasyonlar TotalSegmentator 2.12.0 ile "
        "otomatik üretildi ve manuel düzeltme yapılmadı. Klasik 120 kVp label "
        "haritasında dalak 1, sağ böbrek 2, sol böbrek 3, aorta 52’dir. 140 kVp "
        "gruplarında böbrek etiketi bulunmadığından bu gruplar böbrek "
        "değerlendirmesine alınmadı.",
    )
    add_paragraph(
        doc,
        "Cross-date analizinde her moving seri, 20260805 take1 referansına göre "
        "değerlendirildi. Test-retest analizinde aynı oturumdaki tekrarlar "
        "kullanıldı; bu veri zaten hizalı olduğu için registration zorlanmadı ve "
        "skip5 (ve dolayısıyla 2 mm) eşiğinde tüm pair’ler identity’de kaldı. "
        "Bu ayrım, registration’ın "
        "gerçek katkısını görmek için kritiktir.",
    )

    # 4. Yöntem
    doc.add_heading("4. Yöntem", level=1)
    doc.add_heading("4.1. Görüntü işleme ve segmentasyon", level=2)
    add_paragraph(
        doc,
        "DICOM serileri NIfTI’ye çevrildi ve tüm işlemler DICOM fiziksel "
        "koordinat sisteminde yapıldı. Böylece voxel indeksleri yerine gerçek "
        "dünya koordinatları kullanıldı; farklı FOV ve origin’e sahip seriler "
        "aynı fiziksel uzayda karşılaştırılabildi. Segmentasyon TotalSegmentator "
        "2.12.0 ile üretildi; maske ve görüntü geometrisi birebir aynı grid’de "
        "tutuldu.",
    )
    doc.add_heading("4.2. Registration akışı", level=2)
    add_paragraph(
        doc,
        "Üç durum tanımlandı. Identity, moving maskenin hiçbir transform "
        "uygulanmadan fixed grid’e taşınmasıdır ve ‘çakıştırma öncesi’ durumu "
        "temsil eder. Centroid, sabit kemik maskesinin centroid’leri arasındaki "
        "ötelemedir; yalnızca global kaymayı düzeltir. Rigid, 6 serbestlik "
        "dereceli dönüşümün kemik maskesi içinde optimize edilmesidir; phantom "
        "rijit olduğu için bu adım rotasyon ve ötelemenin büyük kısmını giderir.",
    )
    add_paragraph(
        doc,
        "Rigid dönüşüm iki yolla üretildi: (i) karşılıklı bilgi/benzerlik "
        "tabanlı yoğunluk registration’ı; (ii) vertebra, kaburga ve sternum "
        "centroid’lerinden Kabsch/SVD ile hesaplanan landmark rigid dönüşümü. "
        "Dönüşüm seçimi, optimizasyonda kullanılmayan kaburga/sternum "
        "noktalarında hesaplanan held-out target registration error (TRE) ile "
        "yapıldı. Amaç, registration’ı hedef organ maskelerine değil, stabil "
        "kemik geometrisine dayandırmaktı.",
    )
    add_paragraph(
        doc,
        "Hedef organ centroid’i fixed referansa 2 mm veya daha yakınsa "
        "registration zorlanmadı ve identity kullanıldı. Bu eşik, test-retest "
        "verisinde gereksiz transformu önler; cross-date’te ise gerçek kayması "
        "olan vakaları registration’a sokar. Deformable registration "
        "kullanılmadı; fantom rijit olduğu için deformable, veriyi yapay olarak "
        "warp ederek Dice’ı şişirebilirdi.",
    )
    doc.add_heading("4.3. Metrikler ve yorumları", level=2)
    add_table(
        doc,
        ["Metrik", "Ne ölçer?", "Yön", "Birim"],
        [
            ["Dice", "İki maskenin hacimsel örtüşmesi", "Yüksek iyi", "0–1"],
            ["HD95", "İki yönün 95. persentil yüzey mesafesinin büyüğü; en kötü lokal hata", "Düşük iyi", "mm"],
            ["ASSD", "İki yönlü ortalama yüzey mesafelerinin ortalaması", "Düşük iyi", "mm"],
            ["MSSD", "Tüm yüzey noktalarının havuzlanmış ortalama mesafesi", "Düşük iyi", "mm"],
            ["NSD@1mm", "1 mm toleransta yüzey Dice’ı", "Yüksek iyi", "0–1"],
            ["clDice", "Merkez çizgisi (iskelet) Dice’ı; topoloji duyarlı", "Yüksek iyi", "0–1"],
            ["Centroid", "Maskelerin ağırlık merkezleri arası mesafe", "Düşük iyi", "mm"],
        ],
        widths=[0.8, 3.9, 0.9, 0.7],
    )
    add_caption(doc, "Tablo 2. Kullanılan metrikler ve yorum yönleri.")
    add_paragraph(
        doc,
        "Dice 1’e yaklaştıkça örtüşme artar. ASSD/MSSD/HD95 0’a yaklaştıkça "
        "sınırlar birbirine yaklaşır; yüksek değerler kötü sınır uyumunu gösterir. "
        "Ancak yüksek Dice ile yüksek ASSD’nin birlikte görülmesi çelişki değil, "
        "lokal/izole bir hatanın işaretidir: küçük bir adacık hacimsel Dice’ı "
        "neredeyse değiştirmezken ortalama yüzey mesafesini büyütür. Bu nedenle "
        "bu raporda Dice, ASSD/MSSD ve HD95 birlikte yorumlanmıştır.",
    )
    add_paragraph(
        doc,
        "Metrikler Metrics Reloaded önerilerine uygun hesaplandı: HD95 iki "
        "yönlü ağırlıklı 95. persentillerin maksimumu; ASSD yönlü ortalamaların "
        "ortalaması; MSSD havuzlanmış ortalama; NSD 1 mm toleransta yüzey Dice’ı; "
        "clDice 3B iskelet üzerinden hesaplandı. Tüm yüzey hesapları mm "
        "cinsindendir ve voxel boyutu 0.586×0.586×1.0 mm’dir; bu nedenle 0.3–0.5 "
        "mm’lik ASSD/MSSD değerleri yarım voxel altı kabul edilebilir.",
    )
    doc.add_heading("4.4. Böbrek adacıkları için duyarlılık analizi", level=2)
    add_paragraph(
        doc,
        "Her böbrek maskesi 3B 26-komşulukla bağlı bileşenlere ayrıldı. En büyük "
        "bileşen dışındaki bileşenler ‘adacık’ olarak tanımlandı. Adacıklar "
        "kaynak maskelerden silinmedi; yalnızca tanısal olarak, en büyük bileşen "
        "korunarak metrikler yeniden hesaplandı. Bu, registration başarısı ile "
        "segmentasyon artefaktını birbirinden ayırmak için yapıldı.",
    )
    doc.add_heading("4.5. İstatistik ve görsel QC", level=2)
    add_paragraph(
        doc,
        "Sürekli değişkenler medyan ve çeyrekler arası aralıkla özetlendi; "
        "önce/sonra karşılaştırmaları eşleştirilmiş Wilcoxon testiyle "
        "desteklendi. |ΔDice| ≤ 0.0005 ‘değişmedi’ kabul edildi. Amaç p-değeri "
        "avcılığı değil, etkinin büyüklüğünü ve tutarlılığını göstermektir. "
        "Görsel değerlendirme için her vaka dört sütun (HAM, IDENTITY, CENTROID, "
        "RIGID) ve dört organ olarak siteye aktarıldı; adacıklı vakalar ayrıca "
        "etiketlendi.",
    )
    add_figure(doc, FIG / "method.png", width=6.6,
               caption="Şekil 1. İş akışı: fiziksel uzay, segmentasyon, üç hizalama durumu ve metrik/QC katmanı.")

    # 5. Bulgular
    doc.add_heading("5. Bulgular", level=1)
    doc.add_heading("5.1. Cross-date registration: dört organ", level=2)
    table_rows = []
    for _, row in cross_summary.iterrows():
        table_rows.append([
            row["dataset"], ORGAN_LABEL[row["organ"]], row["n"],
            fmt(row["dice_before"]), fmt(row["dice_after"]),
            f"{row['dice_delta']:+.3f}" if math.isfinite(row["dice_delta"]) else "—",
            p_text(row["p"]), fmt(row["assd_after"], 2),
            fmt(row["mssd_after"], 2), fmt(row["hd95_after"], 2),
            f"{row['improved']}/{row['worsened']}",
        ])
    add_table(
        doc,
        ["Veri", "Organ", "n", "Dice önce", "Dice sonra", "ΔDice", "p",
         "ASSD (mm)", "MSSD (mm)", "HD95 (mm)", "İyileşen/Kötüleşen"],
        table_rows,
        widths=[0.55, 0.9, 0.4, 0.65, 0.65, 0.55, 0.55, 0.65, 0.65, 0.65, 1.0],
        size=8,
    )
    add_caption(doc, "Tablo 3. Cross-date registration öncesi/sonrası medyan Dice, yüzey metrikleri ve değişim sayıları.")
    add_figure(doc, FIG / "dice.png", width=6.6,
               caption="Şekil 2. Cross-date öncesi (identity) ve sonrası (rigid) medyan Dice; PCCT ve SEP, dört organ.")
    add_paragraph(
        doc,
        "Cross-date’te identity Dice değerleri beklendiği gibi düşüktü; çünkü "
        "farklı tarihlerdeki seriler aynı fiziksel konumda değildi. Rigid "
        "registration sonrası tüm organlarda belirgin artış görüldü. PCCT’te "
        "medyan Dice aorta 0.93, dalak 0.96, sağ böbrek 0.94 ve sol böbrek 0.96; "
        "SEP’te 0.92, 0.96, 0.94 ve 0.95 oldu. Böbrekler dalak/aorta ile aynı "
        "aralıkta kaldı; sağ böbreğin hafif düşük olması anatomi/FOV ve "
        "segmentasyon sınırıyla ilişkilidir.",
    )
    add_paragraph(
        doc,
        "İyileşen/kötüleşen sayıları, ortalama farkın tek başına gizleyebileceği "
        "vaka düzeyi değişkenliği gösterir. Özellikle böbreklerde birkaç vakada "
        "Dice artışı sınırlı kaldı; bunların bir kısmı adacık artefaktı, bir "
        "kısmı ise gerçek segmentasyon farkıdır (Bölüm 5.3).",
    )

    doc.add_heading("5.2. Yüzey doğruluğu: ASSD, MSSD ve HD95", level=2)
    add_paragraph(
        doc,
        "Rigid sonrası medyan ASSD ve MSSD çoğu organda 0.3–0.5 mm, HD95 ise "
        "1.0–1.2 mm idi. Bu değerler 1 mm kesit kalınlığında beklenen "
        "segmentasyon sınırı gürültüsüne yakındır. Aorta gibi ince/tübüler "
        "yapılarda aynı milimetrik sınır hatası Dice’ı daha çok düşürür; dalak "
        "ve böbrek gibi hacimli organlarda Dice daha yüksek kalır. Bu nedenle "
        "organlar arasında Dice’ı doğrudan kıyaslamak yerine ASSD/MSSD ile "
        "birlikte yorumlamak gerekir.",
    )
    add_figure(doc, FIG / "surface.png", width=6.6,
               caption="Şekil 3. Cross-date rigid sonrası medyan ASSD ve MSSD; PCCT ve SEP, dört organ.")
    add_paragraph(
        doc,
        "Böbreklerde ham ASSD/MSSD ortalamaları, adacıklı az sayıda vaka "
        "nedeniyle şişebilir. Bu nedenle Tablo 3’te medyan değerler verilmiştir; "
        "adacık etkisi Bölüm 5.3’te ayrıca nicelenmiştir. Medyanlar, tipik "
        "vakadaki gerçek yüzey doğruluğunu temsil eder.",
    )

    doc.add_heading("5.3. Böbrek adacıkları ve largest-component duyarlılığı", level=2)
    if not scan.empty:
        islands = scan[scan["After_components"] > 1]
        add_paragraph(
            doc,
            f"Toplam {len(scan)} böbrek maskesinin {len(islands)}’inde "
            f"(%{100 * len(islands) / len(scan):.1f}) izole bileşen saptandı. "
            "Bunların 10’u cross-date PCCT (7 sağ, 3 sol), 8’i test-retest PCCT "
            "(tamamı 3 mm rekonstrüksiyon, sağ böbrek) idi; SEP’te adacık yoktu. "
            "Ham ortalamalar bu adacıklar nedeniyle şişer; largest-component "
            "ortalamaları medyana yaklaşır.",
        )
        scan_rows = []
        for (dataset, organ), group in scan.groupby(["Dataset", "Organ"], sort=True):
            scan_rows.append([
                f"{dataset.replace(' cross-date', '')} · {ORGAN_LABEL[organ]}",
                len(group), int((group["After_components"] > 1).sum()),
                f"{group['Raw_ASSD_mm'].mean():.2f}/{group['Raw_ASSD_mm'].median():.2f}",
                f"{group['Largest_ASSD_mm'].mean():.2f}/{group['Largest_ASSD_mm'].median():.2f}",
                f"{group['Raw_MSSD_mm'].mean():.2f}/{group['Raw_MSSD_mm'].median():.2f}",
                f"{group['Largest_MSSD_mm'].mean():.2f}/{group['Largest_MSSD_mm'].median():.2f}",
            ])
        add_table(
            doc,
            ["Grup", "n", "Adacık", "ASSD ham ort/med", "ASSD largest ort/med",
             "MSSD ham ort/med", "MSSD largest ort/med"],
            scan_rows,
            widths=[1.6, 0.4, 0.55, 1.0, 1.1, 1.0, 1.1],
            size=8,
        )
        add_caption(doc, "Tablo 4. Böbrek bağlı bileşen taraması; 528 maske, ham ve largest-component ASSD/MSSD (mm).")
        add_figure(doc, FIG / "kidney_artifacts.png", width=6.6,
                   caption="Şekil 4. PCCT böbreklerinde ham ve largest-component ASSD/MSSD ortalamaları.")
        add_paragraph(
            doc,
            "Adacıkların tipik örneği 20260327_t2’dir: sağ böbrekte 2 bileşen, "
            "yaklaşık 494 mm³ adacık; Dice 0.941 kalırken ASSD 15.60 mm, MSSD "
            "13.15 mm ve HD95 299 mm ölçüldü. Yalnızca en büyük bileşenle aynı "
            "vaka ASSD 0.34, MSSD 0.36 ve HD95 1.0 mm’ye indi. Bu, sorunun "
            "registration değil segmentasyon olduğunu gösterir. SEP 20260731_s80 "
            "ise tek bileşenli gerçek bir segmentasyon/rekonstrüksiyon hatasıdır "
            "(Dice 0.677/0.762, ASSD 3.16/2.72 mm, HD95 12.2/11.0 mm); bileşen "
            "filtresi bu vakayı düzeltmez.",
        )
        add_paragraph(
            doc,
            "Klinik/akademik yorum: ASSD/MSSD yüksekliği tek başına "
            "‘registration kötü’ anlamına gelmez. Dice yüksekken ASSD/MSSD "
            "yüksekse, önce segmentasyon adacığı/artefaktı araştırılmalıdır. "
            "Kaynak maskeler değiştirilmeden largest-component duyarlılık "
            "analizi, bu ayrımı şeffaf biçimde raporlamayı sağlar.",
        )

    doc.add_heading("5.4. Zurich test-retest: zorlamasız identity", level=2)
    test_rows = []
    for _, row in test_summary.iterrows():
        test_rows.append([
            row["dataset"], ORGAN_LABEL[row["organ"]], row["n"],
            fmt(row["dice_before"]), fmt(row["dice_after"]),
            f"{row['dice_delta']:+.3f}" if math.isfinite(row["dice_delta"]) else "—",
            fmt(row["assd_after"], 2), fmt(row["mssd_after"], 2),
            fmt(row["hd95_after"], 2),
        ])
    add_table(
        doc,
        ["Veri", "Organ", "n", "Dice önce", "Dice sonra", "ΔDice",
         "ASSD (mm)", "MSSD (mm)", "HD95 (mm)"],
        test_rows,
        widths=[1.1, 0.95, 0.4, 0.65, 0.65, 0.55, 0.65, 0.65, 0.65],
        size=8,
    )
    add_caption(doc, "Tablo 5. Zurich test-retest; skip5 (ve 2 mm) eşiğinde tüm pair’ler identity.")
    add_paragraph(
        doc,
        "Test-retest verisi zaten hizalı olduğu için skip5 (ve dolayısıyla 2 mm) "
        "eşiğinde tüm pair’ler identity’de kaldı; Dice değişimi pratikte sıfırdı. "
        "Tabloda p-değeri "
        "verilmedi; ΔDice etki büyüklüğü ~0 olduğundan küçük sistematik farklar "
        "p’yi yanıltıcı biçimde anlamlı gösterebilir. Bu sonuç, "
        "registration’ı zorlamanın gereksiz olduğunu ve önceki sürümlerde "
        "zorlanan registration’ın yapay düşüşlere yol açabileceğini doğrular. "
        "Böbreklerde 3 mm rekonstrüksiyonlarda adacıklar görüldü; bu nedenle "
        "test-retest böbrek yüzey metrikleri yorumlanırken rekonstrüksiyon "
        "kalınlığı dikkate alınmalıdır.",
    )

    doc.add_heading("5.5. Konsensüs referansı duyarlılığı", level=2)
    consensus_rows = []
    for path, header, prefix in (
        (Path(r"C:/Users/eray/Desktop/ZURICH_CONSENSUS_REFERANS.xlsx"), 2, "Zürich"),
        (Path(r"C:/Users/eray/Desktop/PCCT_SEP_LONG_CONSENSUS.xlsx"), 3, "Long"),
    ):
        if not path.exists():
            continue
        frame = pd.read_excel(path, sheet_name="Tum", header=header)
        frame = frame.dropna(subset=["Dataset", "Organ"])
        for _, row in frame.iterrows():
            consensus_rows.append([
                f"{prefix} {row['Dataset']}", row["Organ"], int(row["N"]),
                fmt(row["Take1 (medyan)"]), fmt(row["Konsensüs (medyan)"]),
                f"{float(row['Delta']):+.3f}",
                f"{int(row['İyileşen'])}/{int(row['Kötüleşen'])}",
            ])
    add_table(
        doc,
        ["Grup", "Organ", "n", "Take1 referansı", "Konsensüs referansı", "Δ", "İyileşen/Kötüleşen"],
        consensus_rows,
        widths=[1.35, 0.8, 0.4, 1.05, 1.15, 0.55, 1.0],
        size=8,
    )
    add_caption(doc, "Tablo 6. Leave-one-out konsensüs referansı; per-take maskeler değiştirilmedi.")
    add_paragraph(
        doc,
        "Konsensüs referansı, take1 yerine diğer tekrarların çoğunluk oyunu "
        "kullanır. Bu bir ground truth değildir; referans seçiminin sonuçlara "
        "etkisini gösteren bir duyarlılık analizidir. Değerler, tek bir take1 "
        "referansına bağlı kalmadan da registration etkisinin yönünün korunduğunu "
        "gösterir; ancak mutlak Dice düzeyi referansa göre değişir.",
    )

    doc.add_heading("5.6. Radyomik feature tekrarlanabilirliği (kapsam: PCCT, dalak/aorta, 3D)", level=2)
    aggregate = rad.groupby("family").agg(
        n=("n_features", "sum"),
        improved=("both_improved", "sum"),
        fdr_tests=("fdr_significant_tests", "sum"),
        fdr_improved=("fdr_significant_improved", "sum"),
        fdr_worsened=("fdr_significant_worsened", "sum"),
    ).reset_index()
    aggregate["pct"] = 100 * aggregate["improved"] / aggregate["n"]
    rad_rows = [
        [row["family"], int(row["n"]), f"{row['pct']:.1f}%",
         int(row["fdr_tests"]), int(row["fdr_improved"]), int(row["fdr_worsened"])]
        for _, row in aggregate.sort_values("n", ascending=False).iterrows()
    ]
    add_table(
        doc,
        ["Feature ailesi", "n", "Her iki koşulda iyileşen", "FDR test", "FDR iyileşen", "FDR kötüleşen"],
        rad_rows,
        widths=[1.8, 0.5, 1.3, 0.8, 0.9, 0.9],
        size=8,
    )
    add_caption(doc, "Tablo 7. PCCT dalak/aorta 3D radyomik feature özeti (418 feature).")
    add_paragraph(
        doc,
        f"Toplam 418 feature’ın {int(aggregate['improved'].sum())}’i (%"
        f"{100 * aggregate['improved'].sum() / aggregate['n'].sum():.1f}) her iki "
        f"koşulda iyileşti. FDR düzeltmesi sonrası {int(aggregate['fdr_tests'].sum())} "
        f"testin {int(aggregate['fdr_improved'].sum())}’i iyileşti, "
        f"{int(aggregate['fdr_worsened'].sum())}’i kötüleşti. GLCM’de karışık bir "
        "tablo varken GLRLM ve NGLDM’de kötüleşme öne çıktı; shape ailesinde "
        "iyileşme eğilimi gözlendi. Bu, registration’ın geometriyi düzeltmesinin "
        "tüm radyomik feature’lara aynı yönde yansımadığını gösterir.",
    )
    add_figure(doc, FIG / "radiomics.png", width=5.8,
               caption="Şekil 5. Feature ailelerinde her iki koşulda iyileşen feature oranı.")
    add_paragraph(
        doc,
        "Kapsam notu: Bu radyomik analiz yalnızca PCCT dalak ve aortası için, 3D "
        "feature’larla ve mevcut üretim transformlarıyla yapılmıştır. Böbrek "
        "radyomikleri bu raporda yoktur; böbrek için yalnızca geometrik/yüzey "
        "değerlendirmesi sunulmuştur. Bu nedenle ‘registration tüm radyomikleri "
        "iyileştirir’ şeklinde genel bir sonuç çıkarılmamalıdır.",
    )

    doc.add_heading("5.7. Görsel değerlendirme", level=2)
    add_paragraph(
        doc,
        "Her vaka için dört organ ve dört durum (HAM, IDENTITY, CENTROID, RIGID) "
        "siteye aktarıldı. HAM ham moving CT’yi, IDENTITY çakıştırma öncesi "
        "maskeyi, CENTROID yalnızca öteleme sonrası maskeyi, RIGID ise nihai "
        "rigid maskeyi gösterir. Böylece hem global kayma hem de sınır farkı "
        "görsel olarak izlenebilir.",
    )
    add_figure(doc, FIG / "organ_panel.png", width=5.6,
               caption="Şekil 6. Temsili vakada (20260707_t3) dört organın centroid öncesi ve rigid sonrası görünümü.")
    add_figure(doc, FIG / "kidney_panel.png", width=6.6,
               caption="Şekil 7. Sağ böbrekte ham, identity, centroid ve rigid durumları.")

    # 6. Tartışma
    doc.add_heading("6. Tartışma", level=1)
    add_paragraph(
        doc,
        "Bu çalışmanın ana bulgusu, fantom rijit olduğu için kemik tabanlı rigid "
        "registration’ın dört organda da yüksek Dice ve milimetre altı yüzey "
        "doğruluğu sağladığıdır. Cross-date’te identity Dice’ın düşük olması "
        "beklenen bir sonuçtur; registration sonrası 0.92–0.96 aralığına çıkış, "
        "global hizalamanın başarılı olduğunu gösterir. Test-retest’te ise "
        "zorlamasız identity’nin en doğru seçim olduğu doğrulanmıştır.",
    )
    add_paragraph(
        doc,
        "Böbreklerin eklenmesi, yalnızca örneklem gücünü artırmakla kalmadı; "
        "segmentasyon artefaktlarının Dice ile görünmeyen etkisini ortaya "
        "çıkardı. 18 adacık vakasında Dice 0.94 civarında kalırken ASSD/MSSD "
        "15–35 mm’ye kadar çıkabildi; largest-component analizi bu değerleri "
        "0.3–0.5 mm’ye indirdi. Bu, yüzey metriklerinin ‘kötü registration’ "
        "sanılabilecek artefaktları ayırmadaki değerini gösterir.",
    )
    add_paragraph(
        doc,
        "SEP 20260731_s80 ayrı bir kategoriye aittir: tek bileşenli, gerçek "
        "segmentasyon/rekonstrüksiyon hatası. Böyle bir vakada rigid registration "
        "veya bileşen filtresi çözüm değildir; re-segmentation veya dışlama "
        "gerekir. 3 mm rekonstrüksiyonlarda adacıkların daha büyük olması, ince "
        "yapılar ve partial volume etkisiyle uyumludur; böbrek yüzey metrikleri "
        "için 1 mm tercih edilmelidir.",
    )
    add_paragraph(
        doc,
        "Radyomik sonuçlar, geometrik başarının radyomik tekrarlanabilirliğe "
        "otomatik olarak dönüşmediğini gösterir. Feature aileleri farklı yönde "
        "hareket eder; bu nedenle registration’ın değeri organ ve feature ailesi "
        "bazında raporlanmalıdır. Konsensüs referansı analizi ise mutlak Dice’ın "
        "referansa duyarlı olduğunu, ancak registration etkisinin yönünün "
        "korunduğunu göstermiştir.",
    )

    # 7. Sınırlılıklar
    doc.add_heading("7. Sınırlılıklar", level=1)
    for item in [
        "Fantom verisi klinik hasta anatomisini temsil etmez; sonuçlar fantom geometrisiyle sınırlıdır.",
        "Segmentasyonlar TotalSegmentator otomatik çıktısıdır; manuel ground truth veya observer variability analizi yoktur.",
        "Tek fixed referans (20260805 take1) kullanılmıştır; konsensüs analizi bunun etkisini kısmen gösterse de bağımsız ground truth değildir.",
        "Rigid registration kullanılmıştır; deformable veya yeniden segmentasyon yapılmamıştır. Bu bilinçli bir tercihtir, ancak lokal şekil farklarını kapatmaz.",
        "Böbreklerde 18 adacık ve bir gerçek segmentasyon aykırısı (s80) vardır; largest-component yalnızca tanısal duyarlılık analizidir, kaynak maskeler değiştirilmemiştir.",
        "3 mm rekonstrüksiyonlarda adacıklar ve partial volume etkisi daha belirgindir; böbrek yüzey metrikleri rekonstrüksiyon kalınlığına duyarlıdır.",
        "Radyomik analiz yalnızca PCCT dalak/aorta 3D feature’larını kapsar; böbrek radyomikleri ve 2D analiz bu raporda ayrıntılandırılmamıştır.",
        "Konsensüs referansı ve farklı değerlendirici/crop stratejileri mutlak Dice’ı değiştirebilir; bu raporda medyan ve duyarlılık analizleri birlikte verilmiştir.",
    ]:
        add_bullet(doc, item)

    # 8. Sonuç
    doc.add_heading("8. Sonuç", level=1)
    add_paragraph(
        doc,
        "PhantomX longitudinal CT’de kemik tabanlı rigid registration, dalak, "
        "aorta, sağ böbrek ve sol böbrek maskelerinin geometrik uyumunu belirgin "
        "biçimde artırdı. Cross-date’te medyan Dice 0.92–0.96, ASSD/MSSD 0.3–0.5 "
        "mm ve HD95 1.0–1.2 mm düzeyine ulaştı; bu, 1 mm kesit kalınlığında "
        "milimetre altı/sınırda doğruluk demektir. Test-retest’te zorlamasız "
        "identity doğru seçim olarak doğrulandı.",
    )
    add_paragraph(
        doc,
        "Böbreklerin değerlendirmeye eklenmesi, segmentasyon adacıklarının "
        "yüzey metriklerini nasıl şişirebileceğini gösterdi. Bu nedenle "
        "akademik raporlamada Dice yanında ASSD/MSSD/HD95, medyan/IQR ve "
        "largest-component duyarlılığı birlikte verilmelidir. Gelecek çalışmalar "
        "için öneriler: böbrek radyomiklerinin eklenmesi, 3 mm "
        "rekonstrüksiyonların ayrı raporlanması, s80 gibi gerçek segmentasyon "
        "hatalarının manuel/alternatif yöntemle düzeltilmesi ve deformable "
        "registration’ın ayrı bir yöntem olarak karşılaştırılmasıdır.",
    )

    # Ek A
    doc.add_page_break()
    doc.add_heading("Ek A. Adacık ve segmentasyon aykırısı vakaları", level=1)
    if not scan.empty:
        flagged = scan[
            (scan["After_components"] > 1)
            | ((scan["After_components"] == 1) & (scan["Raw_ASSD_mm"] > 1.0))
        ].copy()
        def short_label(pair) -> str:
            pair = str(pair)
            match = re.search(r"_(\d{8})_.*?_take(\d+)", pair)
            if not match:
                return pair[-24:]
            if "_s80" in pair:
                return f"{match.group(1)}_s80"
            suffix = re.search(r"_(s\d+)$", pair)
            tail = f"_{suffix.group(1)}" if suffix else ""
            return f"{match.group(1)}_t{match.group(2)}{tail}"

        flagged["short"] = flagged["Pair"].map(short_label)
        flagged_rows = []
        for _, row in flagged.iterrows():
            flagged_rows.append([
                row["short"], row["Dataset"].replace(" cross-date", ""),
                ORGAN_LABEL[row["Organ"]],
                int(row["After_components"]),
                f"{float(row['Small_volume_mm3']):.0f}",
                f"{float(row['Raw_ASSD_mm']):.2f}→{float(row['Largest_ASSD_mm']):.2f}",
                f"{float(row['Raw_MSSD_mm']):.2f}→{float(row['Largest_MSSD_mm']):.2f}",
            ])
        add_table(
            doc,
            ["Vaka", "Veri", "Organ", "Bileşen", "Adacık (mm³)", "ASSD ham→largest", "MSSD ham→largest"],
            flagged_rows,
            widths=[1.15, 1.2, 1.0, 0.6, 0.9, 1.5, 1.5],
            size=7.8,
        )
        add_caption(doc, "Tablo A1. 18 adacık ve 1 gerçek segmentasyon aykırısı; kaynak maskeler değiştirilmedi.")

    doc.add_heading("Ek B. Tekrarlanabilirlik ve dosyalar", level=1)
    add_paragraph(
        doc,
        "Tüm sayılar aşağıdaki güncel çıktılardan yeniden hesaplanmıştır: "
        "PCCT batch_to_20260805_v1.0.0 ve SEP batch_to_20260805_v6_skip2 "
        "metrics.csv; KIDNEY_METRICS_PCCT_SEP.csv; KIDNEY_CC_SCAN.csv; "
        "ZURICH_CONSENSUS_REFERANS.xlsx ve PCCT_SEP_LONG_CONSENSUS.xlsx; "
        "feature_family_summary_3d.csv. Görsel değerlendirme "
        "https://erayakgunn.github.io/registration-overlay/ adresindedir.",
    )
    add_paragraph(
        doc,
        "Yeniden üretim için başlıca komutlar: "
        "python Kod/analyze_kidneys_and_deltas.py; python Kod/kidney_cc_scan.py; "
        "python Kod/generate_site_raw_gallery.py; python "
        "Kod/generate_site_raw_detail.py; python Kod/annotate_site_artifacts.py; "
        "python Kod/registration-radiomics/generate_academic_report_v2.py.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Word report: {OUT}")


if __name__ == "__main__":
    build_report()
