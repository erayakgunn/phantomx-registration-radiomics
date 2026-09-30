"""Small desktop GUI for monitoring one registration pilot.

Run with the registration environment:
    ..\\.venv\\Scripts\\python.exe gui.py
"""

from __future__ import annotations

import queue
import csv
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path(__file__).with_name("pilot.py")
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


class RegistrationGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PhantomX Registration Pilot")
        self.geometry("980x680")
        self.minsize(780, 520)
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.fields: dict[str, tk.StringVar] = {}
        self.qc_window: tk.Toplevel | None = None
        self.qc_photo: ImageTk.PhotoImage | None = None
        self._build()
        self.after(100, self._poll_events)

    def _build(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(7, weight=1)

        ttk.Label(container, text="Registration Pilot", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        ttk.Label(container, text="Dosyalar fixed/moving çiftini tanımlar. Kaynak dosyalar değiştirilmez.").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )

        self.fields["dataset_root"] = tk.StringVar()
        ttk.Label(container, text="Dataset root", width=16).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(container, textvariable=self.fields["dataset_root"]).grid(
            row=2, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(container, text="Browse", command=self._browse_dataset).grid(row=2, column=2, pady=4)
        ttk.Button(container, text="Auto-fill take1/take2", command=self._autofill).grid(
            row=3, column=1, sticky="w", padx=8, pady=(0, 8)
        )

        rows = (
            ("Fixed CT", "fixed_ct"),
            ("Fixed mask", "fixed_seg"),
            ("Moving CT", "moving_ct"),
            ("Moving mask", "moving_seg"),
            ("Output folder", "output"),
        )
        for row, (label, key) in enumerate(rows, start=4):
            self.fields[key] = tk.StringVar()
            ttk.Label(container, text=label, width=16).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(container, textvariable=self.fields[key]).grid(
                row=row, column=1, sticky="ew", padx=8, pady=4
            )
            ttk.Button(container, text="Browse", command=lambda k=key: self._browse(k)).grid(
                row=row, column=2, pady=4
            )

        self.fields["labels"] = tk.StringVar(value="1 52")
        self.fields["registration_labels"] = tk.StringVar(
            value="27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116"
        )
        ttk.Label(container, text="Evaluation labels").grid(row=9, column=0, sticky="w", pady=4)
        ttk.Entry(container, textvariable=self.fields["labels"]).grid(row=9, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(container, text="örn. 1 52 = spleen, aorta").grid(row=9, column=2, sticky="w")

        ttk.Label(container, text="Registration labels").grid(row=10, column=0, sticky="w", pady=4)
        ttk.Entry(container, textvariable=self.fields["registration_labels"]).grid(
            row=10, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Label(container, text="Registration bölgesi; önce organlarla pilot yap").grid(row=10, column=2, sticky="w")

        controls = ttk.Frame(container)
        controls.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        self.run_button = ttk.Button(controls, text="Run registration", command=self._start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(controls, text="Open output", command=self._open_output).pack(side="left")
        ttk.Button(controls, text="Visual QC viewer", command=self._open_visual_qc).pack(side="left", padx=8)
        ttk.Button(controls, text="3D volume viewer", command=self._show_volume_viewer).pack(side="left")
        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=220)
        self.progress.pack(side="right")

        self.status = tk.StringVar(value="Hazır")
        ttk.Label(container, textvariable=self.status).grid(row=12, column=0, columnspan=3, sticky="w", pady=4)

        self.log = tk.Text(container, height=14, wrap="word", state="disabled")
        self.log.grid(row=13, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        container.rowconfigure(13, weight=1)

    def _browse_dataset(self) -> None:
        value = filedialog.askdirectory(title="PhantomX LABELED dataset root")
        if value:
            self.fields["dataset_root"].set(value)
            self._autofill()

    def _autofill(self) -> None:
        root = Path(self.fields["dataset_root"].get().strip())
        ct_files = sorted(root.glob("nifti/*/ct.nii.gz"))
        if not ct_files:
            messagebox.showerror(
                "Dataset bulunamadı",
                "Kök klasörde nifti\\<seri>\\ct.nii.gz bulunamadı.",
            )
            return
        take1 = next((p for p in ct_files if "_take1" in p.parent.name), None)
        moving = None
        if take1 is not None:
            moving_name = take1.parent.name.replace("_take1", "_take2", 1)
            moving_candidate = root / "nifti" / moving_name / "ct.nii.gz"
            if moving_candidate.exists():
                moving = moving_candidate
        if take1 is None or moving is None:
            take1, moving = ct_files[0], ct_files[1] if len(ct_files) > 1 else None
        if moving is None:
            messagebox.showerror("Seri eksik", "En az iki seri bulunamadı.")
            return
        self.fields["fixed_ct"].set(str(take1))
        self.fields["moving_ct"].set(str(moving))
        self.fields["fixed_seg"].set(str(root / "segmentation" / take1.parent.name / "seg.nii.gz"))
        self.fields["moving_seg"].set(str(root / "segmentation" / moving.parent.name / "seg.nii.gz"))
        if not self.fields["output"].get().strip():
            self.fields["output"].set(str(ROOT / "PhantomX" / "REGISTRATION" / "gui_take1_take2"))
        self._write_log(f"Otomatik seçildi: {take1.parent.name} -> {moving.parent.name}")

    def _browse(self, key: str) -> None:
        if key == "output":
            value = filedialog.askdirectory(title="Output folder")
        else:
            value = filedialog.askopenfilename(title=f"Select {key}")
        if value:
            self.fields[key].set(value)

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if self.process is not None:
            return
        required = ("fixed_ct", "fixed_seg", "moving_ct", "moving_seg")
        missing = [key for key in required if not self.fields[key].get().strip()]
        if missing:
            messagebox.showerror("Eksik dosya", "Şu alanlar eksik: " + ", ".join(missing))
            return
        output = self.fields["output"].get().strip()
        if not output:
            output = str(ROOT / "PhantomX" / "REGISTRATION" / (
                "gui_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            ))
            self.fields["output"].set(output)

        interpreter = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        args = [
            interpreter,
            str(PILOT),
            "--fixed-ct", self.fields["fixed_ct"].get(),
            "--fixed-seg", self.fields["fixed_seg"].get(),
            "--moving-ct", self.fields["moving_ct"].get(),
            "--moving-seg", self.fields["moving_seg"].get(),
            "--output", output,
            "--labels", *self.fields["labels"].get().split(),
            "--registration-labels", *self.fields["registration_labels"].get().split(),
        ]
        self._write_log("Komut başlatılıyor...")
        self._write_log("Python: " + interpreter)
        self._write_log("Çıktı: " + output)
        self.status.set("Çalışıyor...")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(10)
        thread = threading.Thread(target=self._run_process, args=(args,), daemon=True)
        thread.start()

    def _run_process(self, args: list[str]) -> None:
        try:
            process = subprocess.Popen(
                args,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self.process = process
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put(("log", line.rstrip()))
            code = process.wait()
            self.events.put(("done", str(code)))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", repr(exc)))

    def _stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self._write_log("İşlem kullanıcı tarafından durduruldu.")
        self._finish("Durduruldu")

    def _finish(self, status: str) -> None:
        self.process = None
        self.status.set(status)
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress.stop()

    def _open_output(self) -> None:
        output = Path(self.fields["output"].get())
        if output.is_dir():
            subprocess.Popen(["explorer", str(output)])
        else:
            messagebox.showinfo("Output", "Henüz geçerli bir output klasörü yok.")

    def _open_visual_qc(self) -> None:
        output = Path(self.fields["output"].get())
        candidates = [
            output / "visual_qc",
            output.parent / "visual_qc",
            ROOT / "PhantomX" / "REGISTRATION" / "batch_to_20260805_v5" / "visual_qc",
        ]
        for folder in candidates:
            if folder.is_dir():
                self._show_qc_viewer(folder)
                return
        messagebox.showinfo("Visual QC", "Visual QC klasörü henüz bulunamadı.")

    def _show_qc_viewer(self, folder: Path) -> None:
        metrics_path = folder / "visual_qc_metrics.csv"
        pngs = sorted(folder.glob("*.png"))
        if not pngs:
            messagebox.showinfo("Visual QC", "Bu klasörde PNG raporu bulunamadı.")
            return
        rows_by_pair: dict[str, list[dict[str, str]]] = {}
        if metrics_path.exists():
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows_by_pair.setdefault(row["pair"], []).append(row)

        if self.qc_window is not None and self.qc_window.winfo_exists():
            self.qc_window.destroy()
        self.qc_window = tk.Toplevel(self)
        self.qc_window.title("PhantomX Visual QC")
        self.qc_window.geometry("1350x900")
        self.qc_window.columnconfigure(0, weight=1)
        self.qc_window.rowconfigure(2, weight=1)

        pair_names = [path.stem for path in pngs]
        selected = tk.StringVar(value=pair_names[0])
        ttk.Label(self.qc_window, text="Pair seç:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        selector = ttk.Combobox(
            self.qc_window, textvariable=selected, values=pair_names,
            state="readonly", width=120,
        )
        selector.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        image_label = ttk.Label(self.qc_window)
        image_label.grid(row=2, column=0, sticky="nsew", padx=10)
        info = tk.Text(self.qc_window, height=7, wrap="word", state="disabled")
        info.grid(row=3, column=0, sticky="ew", padx=10, pady=8)

        def render(*_event) -> None:
            pair = selected.get()
            path = folder / f"{pair}.png"
            if not path.exists():
                return
            image = Image.open(path).convert("RGB")
            image.thumbnail((1280, 700), Image.Resampling.LANCZOS)
            self.qc_photo = ImageTk.PhotoImage(image)
            image_label.configure(image=self.qc_photo)
            lines = [f"Pair: {pair}", ""]
            for row in rows_by_pair.get(pair, []):
                lines.append(
                    f"{row['organ']}: Dice {float(row['before_dice']):.4f} -> "
                    f"{float(row['after_dice']):.4f} | "
                    f"HD95 {float(row['before_hd95_mm']):.2f} -> {float(row['after_hd95_mm']):.2f} mm"
                )
            info.configure(state="normal")
            info.delete("1.0", "end")
            info.insert("end", "\n".join(lines))
            info.configure(state="disabled")

        selector.bind("<<ComboboxSelected>>", render)
        render()

    def _show_volume_viewer(self) -> None:
        root = ROOT / "PhantomX" / "REGISTRATION" / "batch_to_20260805_v5"
        pair_dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and "__" in path.name
            and (path / "seg_registered.nii.gz").exists()
        ) if root.is_dir() else []
        if not pair_dirs:
            messagebox.showinfo("3D viewer", "Registered pair bulunamadı.")
            return

        fixed_ct_path = Path(self.fields.get("fixed_ct", tk.StringVar()).get())
        fixed_seg_path = Path(self.fields.get("fixed_seg", tk.StringVar()).get())
        if not fixed_ct_path.exists() or not fixed_seg_path.exists():
            messagebox.showinfo("3D viewer", "Önce Dataset root seçip Auto-fill yapmalısın.")
            return

        if self.qc_window is not None and self.qc_window.winfo_exists():
            self.qc_window.destroy()
        window = tk.Toplevel(self)
        self.qc_window = window
        window.title("PhantomX 3D Volume Viewer")
        window.geometry("1400x900")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)

        names = [path.name for path in pair_dirs]
        selected = tk.StringVar(value=names[0])
        ttk.Label(window, text="Registered pair").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        selector = ttk.Combobox(window, textvariable=selected, values=names, state="readonly", width=120)
        selector.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
        panels = ttk.Frame(window)
        panels.grid(row=2, column=0, sticky="nsew", padx=10)
        for column in range(3):
            panels.columnconfigure(column, weight=1)
        image_labels = [ttk.Label(panels) for _ in range(3)]
        for column, label in enumerate(image_labels):
            label.grid(row=0, column=column, sticky="nsew", padx=5)
        scales = []
        scale_frame = ttk.Frame(window)
        scale_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=8)
        for column, text in enumerate(("Axial", "Coronal", "Sagittal")):
            ttk.Label(scale_frame, text=text).grid(row=0, column=column, sticky="w")
            scale = tk.Scale(scale_frame, orient="horizontal", from_=0, to=1, showvalue=True,
                             resolution=1, command=lambda _value: render())
            scale.grid(row=1, column=column, sticky="ew", padx=5)
            scale_frame.columnconfigure(column, weight=1)
            scales.append(scale)

        loaded: dict[str, object] = {}

        def load_pair() -> None:
            pair = next(path for path in pair_dirs if path.name == selected.get())
            session, series = pair.name.split("__", 1)
            dataset_root = Path(self.fields["dataset_root"].get())
            moving_seg_path = dataset_root / ".." / session / "segmentation" / series / "seg.nii.gz"
            if not moving_seg_path.exists():
                moving_seg_path = ROOT / "PhantomX" / "LABELED" / session / "segmentation" / series / "seg.nii.gz"
            loaded["ct"] = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_ct_path), sitk.sitkFloat32))
            loaded["fixed"] = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_seg_path), sitk.sitkUInt16))
            loaded["registered"] = sitk.GetArrayFromImage(
                sitk.ReadImage(str(pair / "seg_registered.nii.gz"), sitk.sitkUInt16)
            )
            shape = loaded["ct"].shape
            scales[0].configure(to=shape[0] - 1)
            scales[1].configure(to=shape[1] - 1)
            scales[2].configure(to=shape[2] - 1)
            scales[0].set(shape[0] // 2)
            scales[1].set(shape[1] // 2)
            scales[2].set(shape[2] // 2)
            render()

        def render() -> None:
            if "ct" not in loaded:
                return
            ct = loaded["ct"]
            fixed = loaded["fixed"]
            registered = loaded["registered"]
            indices = [int(scale.get()) for scale in scales]
            slices = [
                (ct[indices[0]], fixed[indices[0]], registered[indices[0]]),
                (ct[:, indices[1], :], fixed[:, indices[1], :], registered[:, indices[1], :]),
                (ct[:, :, indices[2]], fixed[:, :, indices[2]], registered[:, :, indices[2]]),
            ]
            for label_widget, (image, fixed_mask, registered_mask) in zip(image_labels, slices):
                image = np.rot90(image)
                fixed_mask = np.rot90(fixed_mask)
                registered_mask = np.rot90(registered_mask)
                low, high = 40.0 - 200.0, 40.0 + 200.0
                base = np.clip((image - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
                rgb = np.repeat(base[:, :, None], 3, axis=2)
                fixed_area = (fixed_mask == 1) | (fixed_mask == 52)
                registered_area = (registered_mask == 1) | (registered_mask == 52)
                rgb[fixed_area] = [0, 255, 0]
                rgb[registered_area] = [255, 0, 0]
                rgb[fixed_area & registered_area] = [255, 255, 0]
                pil = Image.fromarray(rgb).convert("RGB")
                pil.thumbnail((430, 620), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                label_widget.configure(image=photo)
                label_widget.image = photo

        selector.bind("<<ComboboxSelected>>", lambda _event: load_pair())
        load_pair()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._write_log(value)
                elif kind == "done":
                    self._finish("Tamamlandı" if value == "0" else f"Hata ile bitti (kod {value})")
                else:
                    self._write_log("HATA: " + value)
                    self._finish("Hata")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)


if __name__ == "__main__":
    RegistrationGui().mainloop()
