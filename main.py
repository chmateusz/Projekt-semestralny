import re
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


VALID_DNA = set("ACGTN")



# ANALIZY

def read_sequence_file(path: str):
    if not path:
        raise ValueError("Nie wybrano pliku.")
    if not Path(path).exists():
        raise ValueError("Plik nie istnieje.")

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Plik jest pusty.")

    lines = text.splitlines()
    header = ""

    if lines[0].startswith(">"):
        header = lines[0].strip()
        seq_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
    else:
        seq_lines = [ln.strip() for ln in lines if ln.strip()]

    seq = "".join(seq_lines).upper().replace(" ", "")

    bad = sorted(set(seq) - VALID_DNA)
    if bad:
        raise ValueError(f"Niepoprawne znaki w sekwencji: {bad}")

    return header, seq


def find_motif_positions(seq: str, motif: str):
    motif = motif.upper().strip()

    if not motif:
        raise ValueError("Motyw nie może być pusty.")
    if set(motif) - VALID_DNA:
        raise ValueError("Motyw może zawierać tylko A,C,G,T,N.")

    pattern = re.compile(rf"(?=({re.escape(motif)}))")
    return [m.start() for m in pattern.finditer(seq)]


def segment_counts(seq_len: int, positions0: list[int], bin_size: int):
    if bin_size <= 0:
        raise ValueError("Bin size musi być > 0.")

    n_bins = int(np.ceil(seq_len / bin_size))

    if len(positions0) == 0:
        counts = np.zeros(n_bins, dtype=int)
    else:
        bins = np.array(positions0) // bin_size
        counts = np.bincount(bins, minlength=n_bins)

    return pd.DataFrame({
        "bin_index": np.arange(n_bins),
        "count": counts
    })


# GUI

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DNA Motif Analyzer")
        self.geometry("950x650")

        self.file_var = tk.StringVar()
        self.motif_var = tk.StringVar(value="ATG")
        self.bin_var = tk.StringVar(value="100")

        self._build()

    def _build(self):
        top_frame = tk.Frame(self, padx=10, pady=10)
        top_frame.pack(fill="x")

        tk.Label(top_frame, text="Plik FASTA/TXT:").grid(row=0, column=0, sticky="w")
        tk.Entry(top_frame, textvariable=self.file_var, width=60).grid(row=0, column=1, padx=5)
        tk.Button(top_frame, text="Wybierz...", command=self.pick_file).grid(row=0, column=2)

        tk.Label(top_frame, text="Motyw:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(top_frame, textvariable=self.motif_var, width=20).grid(row=1, column=1, sticky="w", padx=5, pady=(8, 0))

        tk.Label(top_frame, text="Bin size:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        vcmd = (self.register(self._only_digits), "%P")
        tk.Entry(top_frame, textvariable=self.bin_var,
                 validate="key", validatecommand=vcmd,
                 width=10).grid(row=2, column=1, sticky="w", padx=5, pady=(8, 0))

        tk.Button(top_frame, text="Analizuj",
                  font=("Arial", 10, "bold"),
                  command=self.run).grid(row=3, column=1, sticky="w", pady=10)

        self.out = tk.Text(self, height=8)
        self.out.pack(fill="x", padx=10)

        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _only_digits(self, value):
        return value.isdigit() or value == ""

    def pick_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("FASTA/TXT", "*.fasta *.fa *.fna *.txt"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)

    def log(self, text):
        self.out.insert("end", text + "\n")
        self.out.see("end")

    def run(self):
        try:
            start = time.time()

            self.out.delete("1.0", "end")
            self.ax.clear()

            path = self.file_var.get().strip()
            motif = self.motif_var.get().strip().upper()

            if not self.bin_var.get():
                raise ValueError("Podaj bin size.")

            bin_size = int(self.bin_var.get())

            header, seq = read_sequence_file(path)
            positions0 = find_motif_positions(seq, motif)
            bins_df = segment_counts(len(seq), positions0, bin_size)

            # ===== OUTPUT DIRECTORY =====
            BASE_DIR = Path(__file__).resolve().parent
            out_dir = BASE_DIR / "output"
            out_dir.mkdir(exist_ok=True)

            # ===== EXPORT CSV =====
            hits_df = pd.DataFrame({
                "motif": motif,
                "start_1": [p + 1 for p in positions0]
            })

            hits_df.to_csv(out_dir / "hits.csv", index=False)
            bins_df.to_csv(out_dir / "bins.csv", index=False)

            #  UPDATE TEXT
            self.log(f"Długość sekwencji: {len(seq)} nt")
            self.log(f"Liczba wystąpień motywu {motif}: {len(positions0)}")
            self.log(f"Pierwsze pozycje: {[p + 1 for p in positions0[:10]]}")
            self.log(f"Pliki zapisane w:\n{out_dir.resolve()}")

            #  DRAW PLOT
            self.ax.bar(bins_df["bin_index"], bins_df["count"])
            self.ax.set_xlabel("Segment (bin)")
            self.ax.set_ylabel("Liczba wystąpień")
            self.ax.set_title(f"Motyw: {motif}")
            self.ax.grid(True)

            self.figure.tight_layout()
            self.canvas.draw()

            # zapis wykresu
            self.figure.savefig(out_dir / "motif_plot.png")

            elapsed = time.time() - start
            self.log(f"Czas analizy: {elapsed:.3f} s")

        except Exception as e:
            messagebox.showerror("Błąd", str(e))


if __name__ == "__main__":
    App().mainloop()