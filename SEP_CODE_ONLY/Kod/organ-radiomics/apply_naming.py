"""İşlenmiş serilerin çıktı klasörlerini DICOM metadata'sına göre yeniden adlandırır.

Şema: <tarih>_<kurum>_<üretici>_<model>_<kernel>_<kVp>_<mAs>_<FOV>_<kesit>
      (PCCT'de sona rekon eki; test-retest'te _takeN)
Örnek: 20260629_USZ_SIEMENS_SOMATOM_Edge_Plus_Br36f_120_100_300_3

Ne yapar (KESİNLİKLE VERİ KAYBETMEDEN):
  1. Kaynak DICOM ağacını tarar (batch.discover_series) → her seri için ad üretir
     (yalnız DICOM metadata; series_naming.build_name).
  2. Çakışan adları ayrıştırır (_s<SeriesNumber>, sonra UID kuyruğu) → hiçbir seri ezilmez.
  3. data/features, data/nifti, data/segmentations altındaki <series_key> klasörlerini
     üretilen ada TAŞIR (os.rename; idempotent — zaten adlıysa atlar).
  4. NAMING_MAP.json / NAMING_MAP.csv yazar (ad ↔ series_key ↔ kaynak ↔ alanlar).
  5. all_features_combined.csv'yi 'metadata_name' sütunu ekleyerek yeniden derler.

Kullanım:
    C:\\Python314\\python.exe apply_naming.py "C:\\...\\29062026-EID" --dry-run   # önizleme
    C:\\Python314\\python.exe apply_naming.py "C:\\...\\29062026-EID"              # uygula

Not: Yeniden adlandırma resume'u BOZMAZ — series_is_complete sidecar'ı series_key
ile bulur (feature_extraction_mirp._key_index).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

import config
import batch_120kvp as batch
import feature_extraction_mirp as fx
import series_naming as sn
import sort_spatially as ss


# --------------------------------------------------------------------------- #
# Ad üretimi + çakışma çözümü
# --------------------------------------------------------------------------- #
def _build_entries(source_root: Path) -> list[dict[str, Any]]:
    """Kaynaktaki her benzersiz seri için ad adayı + izlenebilirlik meta'sı üretir."""
    records = batch.discover_series(source_root)
    unique, _dups = batch.dedup_and_takes(records)   # take ataması + gerçek-kopya tekilleme
    entries: list[dict[str, Any]] = []
    for r in unique:
        key = batch._series_key(r)
        ds = batch._read_header(r["files"][0]) if r["files"] else None
        if ds is None:
            print(f"[isim] UYARI: {r['folder'].name} ref okunamadı, atlanıyor.")
            continue
        fields = sn.parse_fields(ds, kvp=r.get("kvp"), recon=r.get("recon_type"),
                                 is_pcct=r.get("photon_counting"))
        name = sn.build_name(ds, kvp=r.get("kvp"), recon=r.get("recon_type"),
                             is_pcct=r.get("photon_counting"), take=r.get("take"))
        entries.append({
            "series_key": key,
            "name": name,
            "folder": r["folder"],
            "series_uid": r["series_uid"],
            "series_number": r.get("series_number") or "",
            "take": r.get("take"),
            "fields": fields,
        })
    return entries


def resolve_collisions(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Aynı ada düşen (şemanın ayırt edemediği) FARKLI serileri benzersizleştirir.

    Sıra: (a) SeriesNumber ile (_sN), (b) hâlâ çakışıyorsa UID kuyruğu, (c) paranoyak
    global benzersizlik. Deterministiktir (klasör+UID sırası). Döndürür: çakışma raporu.
    """
    collisions: list[dict[str, str]] = []
    by_name: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_name[e["name"]].append(e)

    for name, group in by_name.items():
        if len(group) == 1:
            continue
        group.sort(key=lambda e: (str(e["folder"]), e["series_uid"]))
        snums = [str(e["series_number"]) for e in group]
        if all(snums) and len(set(snums)) == len(group):
            for e in group:
                e["name"] = f"{name}_s{ss._sanitize(e['series_number'])}"
        else:
            for e in group:
                tail = hashlib.sha1(e["series_uid"].encode("utf-8")).hexdigest()[:6]
                e["name"] = f"{name}_{tail}"
        for e in group:
            collisions.append({"base_name": name, "resolved": e["name"],
                               "series_key": e["series_key"], "folder": e["folder"].name})

    # (c) Paranoyak: yine de eşleşen kalırsa sayısal ek.
    seen: set[str] = set()
    for e in sorted(entries, key=lambda e: (str(e["folder"]), e["series_uid"])):
        base, cand, i = e["name"], e["name"], 2
        while cand in seen:
            cand = f"{base}_{i}"
            i += 1
        e["name"] = cand
        seen.add(cand)
    return collisions


# --------------------------------------------------------------------------- #
# Yeniden adlandırma + çıktı
# --------------------------------------------------------------------------- #
_ROOTS = ("FEATURES", "NIFTI", "SEG")


def _rename_dirs(key: str, name: str, dry: bool) -> list[str]:
    """features/nifti/segmentations altındaki <key> klasörünü <name>'e taşır.

    İdempotent: <key> yoksa ve <name> varsa 'zaten adlı' sayar. Hedef doluysa
    (çakışma çözümü sonrası olmamalı) taşımaz, UYARI verir (veri kaybı önlenir).
    """
    actions: list[str] = []
    for root_attr in _ROOTS:
        base: Path = getattr(config, root_attr)
        src, dst = base / key, base / name
        if src == dst:
            continue
        if not src.exists():
            if dst.exists():
                actions.append(f"{root_attr}: zaten adlı")
            continue
        if dst.exists():
            actions.append(f"{root_attr}: HEDEF DOLU, taşınmadı ({dst.name})")
            continue
        if not dry:
            src.rename(dst)
        actions.append(f"{root_attr}: {'[kuru] ' if dry else ''}{src.name} -> {dst.name}")
    return actions


def _rebuild_combined(entries: list[dict[str, Any]], dry: bool) -> int:
    """metadata_name sütunlu birleşik CSV'yi (yeni adlı klasörlerden) yeniden derler."""
    frames = []
    for e in entries:
        csv = config.FEATURES / e["name"] / "features.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # noqa: BLE001
            print(f"[isim] okunamadı {csv}: {exc}")
            continue
        df.insert(0, "metadata_name", e["name"])
        frames.append(df)
    if not frames:
        print("[isim] Birleşik CSV için features.csv bulunamadı (henüz işlenmemiş olabilir).")
        return 0
    combined = pd.concat(frames, ignore_index=True)
    if not dry:
        batch._safe_write_csv(combined, config.COMBINED_FEATURES_CSV)
        batch._safe_write_xlsx(combined, config.COMBINED_FEATURES_CSV)   # Excel çalışma dosyası da
    print(f"[isim] {'[kuru] ' if dry else ''}Birleşik: {len(combined)} satır, "
          f"{len(combined.columns)} sütun -> {config.COMBINED_FEATURES_CSV.name} (+ .xlsx)")
    return len(combined)


def _write_map(entries: list[dict[str, Any]], dry: bool) -> None:
    """NAMING_MAP.json + .csv: ad ↔ series_key ↔ kaynak klasör ↔ tüm alanlar."""
    rows = []
    for e in entries:
        processed = (config.FEATURES / e["name"] / "features.csv").exists() or \
                    (config.FEATURES / e["series_key"] / "features.csv").exists()
        rows.append({
            "metadata_name": e["name"],
            "series_key": e["series_key"],
            "source_folder": e["folder"].name,
            "series_uid": e["series_uid"],
            "series_number": e["series_number"],
            "take": e["take"],
            "processed": processed,
            **{f"f_{k}": v for k, v in e["fields"].items()},
        })
    if dry:
        return
    config.FEATURES.mkdir(parents=True, exist_ok=True)
    fx._atomic_write_text(config.FEATURES / "NAMING_MAP.json",
                          json.dumps(rows, indent=2, ensure_ascii=False))
    batch._safe_write_csv(pd.DataFrame(rows), config.FEATURES / "NAMING_MAP.csv")


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main(source_root: Path, dry: bool = False, rename: bool = True) -> int:
    config.ensure_dirs()
    print(f"[isim] Kaynak taranıyor: {source_root}")
    entries = _build_entries(source_root)
    collisions = resolve_collisions(entries)
    print(f"[isim] {len(entries)} benzersiz seri; {len(collisions)} çakışma çözüldü.")

    if collisions:
        print("[isim] Çakışma çözümleri (şemanın ayırt edemediği farklı seriler):")
        for c in collisions:
            print(f"    {c['base_name']}  ->  {c['resolved']}   [{c['folder']}]")

    renamed = pending = 0
    for e in entries:
        acts = _rename_dirs(e["series_key"], e["name"], dry) if rename else []
        has_out = (config.FEATURES / e["series_key"]).exists() or \
                  (config.FEATURES / e["name"]).exists()
        if has_out:
            renamed += 1
            tag = " [take%s]" % e["take"] if e["take"] else ""
            print(f"  {e['series_key']}{tag}\n      -> {e['name']}")
            for a in acts:
                print(f"         {a}")
        else:
            pending += 1

    _write_map(entries, dry)
    n_rows = _rebuild_combined(entries, dry)

    print("\n" + "=" * 60)
    print(f"[isim] {'KURU ÇALIŞTIRMA (değişiklik yok)' if dry else 'TAMAM'}")
    print(f"  adlandırılan (çıktısı olan) seri: {renamed}")
    print(f"  henüz işlenmemiş (çıktısız) seri: {pending}")
    print(f"  çözülen çakışma: {len(collisions)}")
    if n_rows:
        print(f"  birleşik tablo satırı: {n_rows}")
    if not dry:
        print(f"  harita: {(config.FEATURES / 'NAMING_MAP.csv')}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Çıktı klasörlerini DICOM metadata'sına göre adlandır")
    ap.add_argument("source", help="Ham DICOM kök klasörü (adların okunacağı yer)")
    ap.add_argument("--dry-run", action="store_true", help="Sadece önizleme; hiçbir şey değiştirme")
    ap.add_argument("--no-rename", action="store_true",
                    help="Klasörleri taşıma; yalnız harita + birleşik CSV üret")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args(sys.argv[1:])
    raise SystemExit(main(Path(a.source), dry=a.dry_run, rename=not a.no_rename))
