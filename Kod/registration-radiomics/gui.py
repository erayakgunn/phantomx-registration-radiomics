"""Small desktop GUI for monitoring one registration pilot.

Run with the registration environment:
    ..\\.venv\\Scripts\\python.exe gui.py
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


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
