import re
import time
import ssl
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import certifi
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

from Bio import Entrez, SeqIO


# =========================
# STAŁE / KONFIGURACJA
# =========================

VALID_DNA = set("ACGTN")

SOURCE_FILE = "file"
SOURCE_NCBI = "ncbi"

PLOT_COMPARISON = "comparison"
PLOT_SEGMENTATION = "segmentation"
PLOT_POSITIONS = "positions"

DEFAULT_EMAIL = "twoj_mail@example.com"
DEFAULT_MOTIFS = "ATG,TATA,CGCG"
DEFAULT_BIN_SIZE = "100"
WINDOW_GEOMETRY = "1120x680"
APP_TITLE = "DNA Motif Analyzer - Extended"

FILETYPES_FASTA = [("FASTA/TXT", "*.fasta *.fa *.fna *.txt"), ("All files", "*.*")]


# =========================
# LOGIKA: SEKWENCJE
# =========================

def validate_dna(seq: str) -> str:
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\r", "")
    bad = sorted(set(seq) - VALID_DNA)
    if bad:
        raise ValueError(f"Niepoprawne znaki w sekwencji: {bad}")
    if not seq:
        raise ValueError("Sekwencja jest pusta.")
    return seq


def read_sequence_file(path: str) -> Tuple[str, str, str]:
    if not path:
        raise ValueError("Nie wybrano pliku.")

    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("Plik nie istnieje.")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Plik jest pusty.")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Plik nie zawiera danych.")

    if lines[0].startswith(">"):
        header = lines[0]
        seq_parts = []
        encountered_extra_header = False

        for ln in lines[1:]:
            if ln.startswith(">"):
                encountered_extra_header = True
                break
            seq_parts.append(ln)

        if encountered_extra_header:
            raise ValueError(
                "Plik multi-FASTA nie jest obsługiwany w tej wersji. Użyj pliku z jednym rekordem."
            )

        seq = validate_dna("".join(seq_parts))
        return file_path.name, header, seq

    seq = validate_dna("".join(lines))
    return file_path.name, file_path.name, seq


def fetch_sequence_from_ncbi(accession: str, email: str) -> Tuple[str, str, str]:
    accession = accession.strip()
    email = email.strip() or DEFAULT_EMAIL

    if not accession:
        raise ValueError("Podaj accession ID.")

    Entrez.email = email
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        from urllib.request import urlopen
        import Bio.Entrez

        Bio.Entrez.urlopen = lambda *args, **kwargs: urlopen(
            *args, context=ssl_context, **kwargs
        )

        with Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="fasta",
            retmode="text"
        ) as handle:
            fasta_text = handle.read()

    except Exception as e:
        raise ValueError(f"Nie udało się pobrać sekwencji z NCBI: {e}")

    if not fasta_text.strip():
        raise ValueError("NCBI nie zwróciło danych.")

    record = SeqIO.read(StringIO(fasta_text), "fasta")
    header = record.description
    seq = validate_dna(str(record.seq))
    source_name = f"NCBI:{accession}"

    return source_name, header, seq


def load_sequence(
    source_mode: str,
    file_path: str,
    accession: str,
    email: str,
    seq_label: str,
    required: bool = True
) -> Optional[Tuple[str, str, str]]:
    if source_mode == SOURCE_FILE:
        file_path = file_path.strip()
        if file_path:
            return read_sequence_file(file_path)
        if required:
            raise ValueError(f"Dla {seq_label} wybrano źródło 'Plik', ale nie wskazano pliku.")
        return None

    if source_mode == SOURCE_NCBI:
        accession = accession.strip()
        if accession:
            return fetch_sequence_from_ncbi(accession, email)
        if required:
            raise ValueError(f"Dla {seq_label} wybrano źródło 'NCBI', ale nie podano accession ID.")
        return None

    raise ValueError(f"Nieprawidłowe źródło dla {seq_label}.")


# =========================
# LOGIKA: MOTYWY I ANALIZA
# =========================

def normalize_motifs(motifs_text: str) -> List[str]:
    motifs = [m.strip().upper() for m in motifs_text.split(",") if m.strip()]
    if not motifs:
        raise ValueError("Podaj co najmniej jeden motyw.")

    for motif in motifs:
        bad = set(motif) - VALID_DNA
        if bad:
            raise ValueError(f"Niepoprawny motyw: {motif}. Dozwolone: A, C, G, T, N.")
    return motifs


def motif_to_regex(motif: str) -> str:
    return "".join("[ACGTN]" if ch == "N" else re.escape(ch) for ch in motif)


def find_motif_positions(seq: str, motif: str) -> List[int]:
    pattern = re.compile(rf"(?=({motif_to_regex(motif)}))")
    return [m.start() for m in pattern.finditer(seq)]


def segment_counts(seq_len: int, positions0: List[int], bin_size: int) -> pd.DataFrame:
    if bin_size <= 0:
        raise ValueError("Bin size musi być > 0.")

    n_bins = max(int(np.ceil(seq_len / bin_size)), 1)

    if not positions0:
        counts = np.zeros(n_bins, dtype=int)
    else:
        bins = np.array(positions0) // bin_size
        counts = np.bincount(bins, minlength=n_bins)

    starts = np.arange(n_bins) * bin_size + 1
    ends = np.minimum((np.arange(n_bins) + 1) * bin_size, seq_len)

    return pd.DataFrame({
        "bin_index": np.arange(n_bins),
        "start_nt": starts,
        "end_nt": ends,
        "count": counts
    })


def compute_summary(seq: str, motif: str, positions0: List[int]) -> Dict:
    seq_len = len(seq)
    count = len(positions0)
    density_per_1000 = (count / seq_len * 1000) if seq_len else 0.0

    if len(positions0) >= 2:
        gaps = np.diff(positions0)
        mean_gap = float(np.mean(gaps))
        median_gap = float(np.median(gaps))
        min_gap = int(np.min(gaps))
        max_gap = int(np.max(gaps))
    else:
        mean_gap = np.nan
        median_gap = np.nan
        min_gap = np.nan
        max_gap = np.nan

    return {
        "motif": motif,
        "sequence_length": seq_len,
        "count": count,
        "density_per_1000nt": round(density_per_1000, 4),
        "mean_gap_nt": mean_gap,
        "median_gap_nt": median_gap,
        "min_gap_nt": min_gap,
        "max_gap_nt": max_gap
    }


def analyze_sequence(
    source_name: str,
    header: str,
    seq: str,
    motifs: List[str],
    bin_size: int
) -> Dict[str, pd.DataFrame]:
    summary_rows = []
    hits_rows = []
    bins_frames = []

    for motif in motifs:
        positions0 = find_motif_positions(seq, motif)

        bins_df = segment_counts(len(seq), positions0, bin_size).copy()
        bins_df["motif"] = motif
        bins_df["source_name"] = source_name
        bins_df["header"] = header
        bins_frames.append(bins_df)

        summary = compute_summary(seq, motif, positions0)
        summary["source_name"] = source_name
        summary["header"] = header
        summary["bin_size"] = bin_size
        summary_rows.append(summary)

        for p in positions0:
            hits_rows.append({
                "source_name": source_name,
                "header": header,
                "motif": motif,
                "start_0": p,
                "start_1": p + 1,
                "end_1": p + len(motif)
            })

    return {
        "summary": pd.DataFrame(summary_rows),
        "hits": pd.DataFrame(hits_rows),
        "bins": pd.concat(bins_frames, ignore_index=True) if bins_frames else pd.DataFrame()
    }


# =========================
# RAPORT PDF
# =========================

def export_pdf_report(
    out_path: Path,
    analysis1: Dict[str, pd.DataFrame],
    source1: str,
    analysis2: Optional[Dict[str, pd.DataFrame]] = None,
    source2: Optional[str] = None
):
    def add_table_page(pdf, title: str, df: pd.DataFrame, source_label: str = ""):
        fig = Figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=16)

        if source_label:
            ax.text(0.02, 0.90, source_label, ha="left", va="top", fontsize=10)

        if df.empty:
            ax.text(0.02, 0.82, "Brak danych.", fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            fig.clear()
            return

        show_df = df.copy()
        wanted_cols = [
            "motif",
            "count",
            "density_per_1000nt",
            "mean_gap_nt",
            "median_gap_nt",
            "min_gap_nt",
            "max_gap_nt"
        ]
        show_df = show_df[[c for c in wanted_cols if c in show_df.columns]]

        for col in show_df.columns:
            if pd.api.types.is_float_dtype(show_df[col]):
                show_df[col] = show_df[col].round(3)

        show_df = show_df.rename(columns={
            "density_per_1000nt": "dens/1000nt",
            "mean_gap_nt": "mean_gap",
            "median_gap_nt": "median_gap",
            "min_gap_nt": "min_gap",
            "max_gap_nt": "max_gap"
        })

        if len(show_df) > 25:
            show_df = show_df.head(25)

        ncols = len(show_df.columns)
        col_widths = [1 / ncols] * ncols

        table = ax.table(
            cellText=show_df.values,
            colLabels=show_df.columns,
            cellLoc="center",
            colLoc="center",
            bbox=[0.05, 0.48, 0.90, 0.22],
            colWidths=col_widths
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_height(cell.get_height() * 1.15)

        pdf.savefig(fig, bbox_inches="tight")
        fig.clear()

    with PdfPages(out_path) as pdf:
        fig0 = Figure(figsize=(11.69, 8.27))
        ax0 = fig0.add_subplot(111)
        ax0.axis("off")

        lines = [
            "DNA Motif Analyzer - raport zbiorczy",
            "",
            f"Sekwencja 1: {source1}",
        ]
        if source2:
            lines.append(f"Sekwencja 2: {source2}")

        lines += [
            "",
            "Raport zawiera:",
            "- podsumowanie statystyk motywów",
            "- porównanie liczby wystąpień",
            "- segmentację dla każdego motywu",
            "- rozmieszczenie motywów na osi sekwencji"
        ]

        ax0.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=14)
        pdf.savefig(fig0)
        fig0.clear()

        src1 = str(analysis1["summary"]["source_name"].iloc[0]) if "source_name" in analysis1["summary"].columns else ""
        hdr1 = str(analysis1["summary"]["header"].iloc[0]) if "header" in analysis1["summary"].columns else ""
        add_table_page(
            pdf,
            "Podsumowanie - sekwencja 1",
            analysis1["summary"],
            source_label=f"Źródło: {src1}\nNagłówek: {hdr1[:120]}"
        )

        if analysis2 is not None:
            src2 = str(analysis2["summary"]["source_name"].iloc[0]) if "source_name" in analysis2["summary"].columns else ""
            hdr2 = str(analysis2["summary"]["header"].iloc[0]) if "header" in analysis2["summary"].columns else ""
            add_table_page(
                pdf,
                "Podsumowanie - sekwencja 2",
                analysis2["summary"],
                source_label=f"Źródło: {src2}\nNagłówek: {hdr2[:120]}"
            )

        fig1 = Figure(figsize=(11.69, 8.27))
        ax1 = fig1.add_subplot(111)

        s1 = analysis1["summary"][["motif", "count"]].copy()
        motifs = s1["motif"].tolist()
        x = np.arange(len(motifs))
        width = 0.35

        ax1.bar(x - width / 2, s1["count"], width=width, label="Sekwencja 1")

        if analysis2 is not None:
            s2 = analysis2["summary"].set_index("motif").reindex(motifs).fillna(0)
            ax1.bar(x + width / 2, s2["count"], width=width, label="Sekwencja 2")

        ax1.set_xticks(x)
        ax1.set_xticklabels(motifs)
        ax1.set_xlabel("Motyw")
        ax1.set_ylabel("Liczba wystąpień")
        ax1.set_title("Porównanie liczby wystąpień motywów")
        ax1.grid(True)
        ax1.legend()
        pdf.savefig(fig1)
        fig1.clear()

        all_motifs = analysis1["summary"]["motif"].tolist()

        for motif in all_motifs:
            fig2 = Figure(figsize=(11.69, 8.27))
            ax2 = fig2.add_subplot(111)

            d1 = analysis1["bins"][analysis1["bins"]["motif"] == motif]
            if not d1.empty:
                ax2.plot(d1["start_nt"], d1["count"], marker="o", label="Sekwencja 1")

            if analysis2 is not None:
                d2 = analysis2["bins"][analysis2["bins"]["motif"] == motif]
                if not d2.empty:
                    ax2.plot(d2["start_nt"], d2["count"], marker="o", label="Sekwencja 2")

            ax2.set_xlabel("Pozycja startowa segmentu (nt)")
            ax2.set_ylabel("Liczba trafień")
            ax2.set_title(f"Segmentacja motywu: {motif}")
            ax2.grid(True)
            ax2.legend()
            pdf.savefig(fig2)
            fig2.clear()

        for motif in all_motifs:
            fig3 = Figure(figsize=(11.69, 8.27))
            ax3 = fig3.add_subplot(111)

            d1 = analysis1["hits"][analysis1["hits"]["motif"] == motif]
            if not d1.empty:
                ax3.scatter(d1["start_1"], np.ones(len(d1)), label="Sekwencja 1")

            if analysis2 is not None:
                d2 = analysis2["hits"][analysis2["hits"]["motif"] == motif]
                if not d2.empty:
                    ax3.scatter(d2["start_1"], np.ones(len(d2)) * 2, label="Sekwencja 2")

            ax3.set_xlabel("Pozycja w sekwencji (nt)")
            ax3.set_yticks([1, 2] if analysis2 is not None else [1])
            ax3.set_yticklabels(["Sekwencja 1", "Sekwencja 2"] if analysis2 is not None else ["Sekwencja 1"])
            ax3.set_title(f"Rozmieszczenie motywu na osi sekwencji: {motif}")
            ax3.grid(True)
            ax3.legend()
            pdf.savefig(fig3)
            fig3.clear()


# =========================
# GUI
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(WINDOW_GEOMETRY)
        self.minsize(980, 620)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.file1_var = tk.StringVar()
        self.file2_var = tk.StringVar()

        self.seq1_source_var = tk.StringVar(value=SOURCE_FILE)
        self.seq2_source_var = tk.StringVar(value=SOURCE_FILE)

        self.ncbi1_var = tk.StringVar()
        self.ncbi2_var = tk.StringVar()
        self.email_var = tk.StringVar(value=DEFAULT_EMAIL)

        self.motifs_var = tk.StringVar(value=DEFAULT_MOTIFS)
        self.bin_var = tk.StringVar(value=DEFAULT_BIN_SIZE)

        self.analysis1 = None
        self.analysis2 = None

        self.current_scatter = None
        self.current_points_meta = []

        self._build()

    def _build(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.settings_tab = tk.Frame(self.notebook)
        self.results_tab = tk.Frame(self.notebook)

        self.notebook.add(self.settings_tab, text="Ustawienia")
        self.notebook.add(self.results_tab, text="Wyniki")

        self.settings_tab.grid_columnconfigure(0, weight=1)
        self.results_tab.grid_columnconfigure(0, weight=1)
        self.results_tab.grid_rowconfigure(1, weight=1)

        self._build_settings_tab()
        self._build_results_tab()

    def _build_settings_tab(self):
        header_frame = tk.Frame(self.settings_tab, padx=10, pady=10)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        tk.Label(
            header_frame,
            text="DNA Motif Analyzer",
            font=("Arial", 18, "bold"),
            anchor="center",
            justify="center"
        ).grid(row=0, column=0, sticky="ew")

        tk.Label(
            header_frame,
            text=(
                "Aplikacja do analizy motywów sekwencyjnych w DNA. "
                "Obsługuje pliki FASTA/TXT i rekordy NCBI, porównanie dwóch sekwencji, "
                "wizualizację oraz eksport zbiorczego raportu PDF."
            ),
            font=("Arial", 10),
            anchor="center",
            justify="center",
            wraplength=1100
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        form_frame = tk.Frame(self.settings_tab, padx=10, pady=5)
        form_frame.grid(row=1, column=0, sticky="ew")
        form_frame.grid_columnconfigure(1, weight=1)

        row = 0

        row = self._build_sequence_controls(
            parent=form_frame,
            row=row,
            seq_number=1,
            source_var=self.seq1_source_var,
            file_var=self.file1_var,
            ncbi_var=self.ncbi1_var,
            pick_cmd=self.pick_file1,
            fetch_cmd=self.fetch_ncbi1,
            clear_cmd=self.clear_seq1,
            pady_top=0
        )

        row = self._build_sequence_controls(
            parent=form_frame,
            row=row,
            seq_number=2,
            source_var=self.seq2_source_var,
            file_var=self.file2_var,
            ncbi_var=self.ncbi2_var,
            pick_cmd=self.pick_file2,
            fetch_cmd=self.fetch_ncbi2,
            clear_cmd=self.clear_seq2,
            pady_top=12
        )

        tk.Label(form_frame, text="E-mail do Entrez/NCBI:").grid(row=row, column=0, sticky="w", pady=(12, 0))
        tk.Entry(form_frame, textvariable=self.email_var, width=35).grid(row=row, column=1, padx=5, sticky="w", pady=(12, 0))

        row += 1
        tk.Label(form_frame, text="Motywy (po przecinku):").grid(row=row, column=0, sticky="w", pady=(12, 0))
        tk.Entry(form_frame, textvariable=self.motifs_var, width=32).grid(row=row, column=1, padx=5, sticky="w", pady=(12, 0))

        row += 1
        tk.Label(form_frame, text="Bin size:").grid(row=row, column=0, sticky="w")
        vcmd = (self.register(self._only_digits), "%P")
        tk.Entry(
            form_frame,
            textvariable=self.bin_var,
            validate="key",
            validatecommand=vcmd,
            width=10
        ).grid(row=row, column=1, padx=5, sticky="w")

        row += 1
        tk.Button(
            form_frame,
            text="Analizuj",
            font=("Arial", 10, "bold"),
            command=self.run_analysis
        ).grid(row=row, column=1, sticky="w", pady=10)

        tk.Button(
            form_frame,
            text="Wyczyść wszystko",
            command=self.clear_all_inputs
        ).grid(row=row, column=2, sticky="w", padx=(10, 0), pady=10)

    def _build_results_tab(self):
        controls_frame = tk.Frame(self.results_tab, padx=10, pady=10)
        controls_frame.grid(row=0, column=0, sticky="ew")

        tk.Label(controls_frame, text="Tryb wykresu:").grid(row=0, column=0, sticky="w")
        self.plot_mode = tk.StringVar(value=PLOT_COMPARISON)
        ttk.Combobox(
            controls_frame,
            textvariable=self.plot_mode,
            values=[PLOT_COMPARISON, PLOT_SEGMENTATION, PLOT_POSITIONS],
            state="readonly",
            width=20
        ).grid(row=0, column=1, sticky="w", padx=(5, 20))

        tk.Label(controls_frame, text="Motyw do wykresu szczegółowego:").grid(row=0, column=2, sticky="w")
        self.selected_motif_var = tk.StringVar(value="ATG")
        self.motif_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.selected_motif_var,
            values=["ATG"],
            state="readonly",
            width=20
        )
        self.motif_combo.grid(row=0, column=3, sticky="w", padx=(5, 20))

        tk.Button(controls_frame, text="Odśwież wykres", command=self.refresh_plot).grid(row=0, column=4, sticky="w")
        tk.Button(controls_frame, text="Eksport PDF", command=self.export_all).grid(row=0, column=5, sticky="w", padx=(10, 0))

        content_frame = tk.Frame(self.results_tab, padx=10, pady=5)
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)

        self.out = tk.Text(content_frame, height=10)
        self.out.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.plot_frame = tk.Frame(content_frame)
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 5.5), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas_visible = False

    def _build_sequence_controls(
        self,
        parent,
        row: int,
        seq_number: int,
        source_var: tk.StringVar,
        file_var: tk.StringVar,
        ncbi_var: tk.StringVar,
        pick_cmd,
        fetch_cmd,
        clear_cmd,
        pady_top: int = 0
    ) -> int:
        tk.Label(parent, text=f"Sekwencja {seq_number} - źródło:").grid(row=row, column=0, sticky="w", pady=(pady_top, 0))
        tk.Radiobutton(parent, text="Plik", variable=source_var, value=SOURCE_FILE).grid(row=row, column=1, sticky="w", pady=(pady_top, 0))
        tk.Radiobutton(parent, text="NCBI", variable=source_var, value=SOURCE_NCBI).grid(row=row, column=1, padx=(80, 0), sticky="w", pady=(pady_top, 0))

        row += 1
        tk.Label(parent, text=f"Sekwencja {seq_number} - plik:").grid(row=row, column=0, sticky="w")
        tk.Entry(parent, textvariable=file_var, width=50).grid(row=row, column=1, padx=5, sticky="w")
        tk.Button(parent, text="Wybierz...", command=pick_cmd).grid(row=row, column=2, sticky="w")
        tk.Button(parent, text="Clear", command=clear_cmd).grid(row=row, column=3, padx=(5, 0), sticky="w")

        row += 1
        tk.Label(parent, text=f"Sekwencja {seq_number} - accession NCBI:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        tk.Entry(parent, textvariable=ncbi_var, width=24).grid(row=row, column=1, padx=5, sticky="w", pady=(6, 0))
        tk.Button(parent, text=f"Pobierz {seq_number} z NCBI", command=fetch_cmd).grid(row=row, column=2, sticky="w", pady=(6, 0))

        return row + 1

    def _only_digits(self, value: str) -> bool:
        return value.isdigit() or value == ""

    def log(self, text: str):
        self.out.insert("end", text + "\n")
        self.out.see("end")

    def clear_seq1(self):
        self.file1_var.set("")
        self.ncbi1_var.set("")
        self.seq1_source_var.set(SOURCE_FILE)

    def clear_seq2(self):
        self.file2_var.set("")
        self.ncbi2_var.set("")
        self.seq2_source_var.set(SOURCE_FILE)

    def clear_all_inputs(self):
        self.clear_seq1()
        self.clear_seq2()
        self.email_var.set(DEFAULT_EMAIL)
        self.motifs_var.set(DEFAULT_MOTIFS)
        self.bin_var.set(DEFAULT_BIN_SIZE)

    def pick_file1(self):
        path = filedialog.askopenfilename(filetypes=FILETYPES_FASTA)
        if path:
            self.file1_var.set(path)
            self.ncbi1_var.set("")
            self.seq1_source_var.set(SOURCE_FILE)

    def pick_file2(self):
        path = filedialog.askopenfilename(filetypes=FILETYPES_FASTA)
        if path:
            self.file2_var.set(path)
            self.ncbi2_var.set("")
            self.seq2_source_var.set(SOURCE_FILE)

    def fetch_ncbi1(self):
        try:
            source, header, seq = fetch_sequence_from_ncbi(self.ncbi1_var.get(), self.email_var.get())
            self.file1_var.set("")
            self.seq1_source_var.set(SOURCE_NCBI)
            self.log(f"Pobrano sekwencję 1 z NCBI: {source}, długość = {len(seq)} nt")
        except Exception as e:
            messagebox.showerror("Błąd NCBI", str(e))

    def fetch_ncbi2(self):
        try:
            source, header, seq = fetch_sequence_from_ncbi(self.ncbi2_var.get(), self.email_var.get())
            self.file2_var.set("")
            self.seq2_source_var.set(SOURCE_NCBI)
            self.log(f"Pobrano sekwencję 2 z NCBI: {source}, długość = {len(seq)} nt")
        except Exception as e:
            messagebox.showerror("Błąd NCBI", str(e))

    def get_seq1(self):
        return load_sequence(
            source_mode=self.seq1_source_var.get(),
            file_path=self.file1_var.get(),
            accession=self.ncbi1_var.get(),
            email=self.email_var.get(),
            seq_label="sekwencji 1",
            required=True
        )

    def get_seq2(self):
        return load_sequence(
            source_mode=self.seq2_source_var.get(),
            file_path=self.file2_var.get(),
            accession=self.ncbi2_var.get(),
            email=self.email_var.get(),
            seq_label="sekwencji 2",
            required=False
        )

    def run_analysis(self):
        try:
            start = time.time()
            self.out.delete("1.0", "end")

            if not self.bin_var.get():
                raise ValueError("Podaj bin size.")

            bin_size = int(self.bin_var.get())
            motifs = normalize_motifs(self.motifs_var.get())

            seq1 = self.get_seq1()
            seq2 = self.get_seq2()

            source1, header1, dna1 = seq1
            self.analysis1 = analyze_sequence(source1, header1, dna1, motifs, bin_size)
            self.analysis2 = None

            self.log(f"Sekwencja 1: {source1}")
            self.log(f"Nagłówek 1: {header1}")
            self.log(f"Długość 1: {len(dna1)} nt")
            self.log("")

            for _, row in self.analysis1["summary"].iterrows():
                self.log(f"[SEQ1] {row['motif']}: count={row['count']}, density/1000nt={row['density_per_1000nt']}")

            if seq2 is not None:
                source2, header2, dna2 = seq2
                self.analysis2 = analyze_sequence(source2, header2, dna2, motifs, bin_size)

                self.log("")
                self.log(f"Sekwencja 2: {source2}")
                self.log(f"Nagłówek 2: {header2}")
                self.log(f"Długość 2: {len(dna2)} nt")
                self.log("")

                for _, row in self.analysis2["summary"].iterrows():
                    self.log(f"[SEQ2] {row['motif']}: count={row['count']}, density/1000nt={row['density_per_1000nt']}")

            self.motif_combo["values"] = motifs
            self.selected_motif_var.set(motifs[0])

            self.refresh_plot()
            self.notebook.select(self.results_tab)

            elapsed = time.time() - start
            self.log("")
            self.log(f"Czas analizy: {elapsed:.3f} s")

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def on_pick(self, event):
        try:
            if self.current_scatter is None:
                return
            if event.artist != self.current_scatter:
                return
            if not hasattr(event, "ind") or len(event.ind) == 0:
                return

            idx = event.ind[0]
            if idx >= len(self.current_points_meta):
                return

            seq_label, motif, pos = self.current_points_meta[idx]
            self.log(f"Kliknięto punkt: {seq_label}, motyw {motif}, pozycja {pos}")
            messagebox.showinfo(
                "Szczegóły punktu",
                f"{seq_label}\nMotyw: {motif}\nPozycja: {pos}"
            )
        except Exception as e:
            messagebox.showerror("Błąd kliknięcia", str(e))

    def refresh_plot(self):
        try:
            self.ax.clear()
            self.current_scatter = None
            self.current_points_meta = []

            if self.analysis1 is None:
                return

            mode = self.plot_mode.get()
            motif = self.selected_motif_var.get()

            if mode == PLOT_COMPARISON:
                self._plot_comparison()
            elif mode == PLOT_SEGMENTATION:
                self._plot_segmentation(motif)
            elif mode == PLOT_POSITIONS:
                self._plot_positions(motif)

            if not self.canvas_visible:
                self.canvas_widget.grid(row=0, column=0, sticky="nsew")
                self.canvas_visible = True
                self.update_idletasks()

            self.figure.tight_layout(rect=[0, 0, 1, 0.95])
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Błąd wykresu", str(e))

    def _plot_comparison(self):
        s1 = self.analysis1["summary"][["motif", "count"]].copy()
        motifs = s1["motif"].tolist()
        x = np.arange(len(motifs))
        width = 0.35

        self.ax.bar(x - width / 2, s1["count"], width=width, label="Sekwencja 1")

        if self.analysis2 is not None:
            s2 = self.analysis2["summary"].set_index("motif").reindex(motifs).fillna(0)
            self.ax.bar(x + width / 2, s2["count"], width=width, label="Sekwencja 2")

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(motifs)
        self.ax.set_xlabel("Motyw")
        self.ax.set_ylabel("Liczba wystąpień")
        self.ax.set_title("Porównanie liczby wystąpień motywów")
        self.ax.legend()
        self.ax.grid(True)

    def _plot_segmentation(self, motif: str):
        d1 = self.analysis1["bins"][self.analysis1["bins"]["motif"] == motif]
        if not d1.empty:
            self.ax.plot(d1["start_nt"], d1["count"], marker="o", label="Sekwencja 1")

        if self.analysis2 is not None:
            d2 = self.analysis2["bins"][self.analysis2["bins"]["motif"] == motif]
            if not d2.empty:
                self.ax.plot(d2["start_nt"], d2["count"], marker="o", label="Sekwencja 2")

        self.ax.set_xlabel("Pozycja startowa segmentu (nt)")
        self.ax.set_ylabel("Liczba trafień")
        self.ax.set_title(f"Segmentacja motywu: {motif}")
        self.ax.legend()
        self.ax.grid(True)

    def _plot_positions(self, motif: str):
        x_positions = []
        y_positions = []

        d1 = self.analysis1["hits"][self.analysis1["hits"]["motif"] == motif]
        if not d1.empty:
            for pos in d1["start_1"].tolist():
                x_positions.append(pos)
                y_positions.append(1)
                self.current_points_meta.append(("Sekwencja 1", motif, pos))

        if self.analysis2 is not None:
            d2 = self.analysis2["hits"][self.analysis2["hits"]["motif"] == motif]
            if not d2.empty:
                for pos in d2["start_1"].tolist():
                    x_positions.append(pos)
                    y_positions.append(2)
                    self.current_points_meta.append(("Sekwencja 2", motif, pos))

        if x_positions:
            self.current_scatter = self.ax.scatter(
                x_positions,
                y_positions,
                label="Motywy",
                picker=5
            )

        self.ax.set_xlabel("Pozycja w sekwencji (nt)")
        self.ax.set_yticks([1, 2] if self.analysis2 is not None else [1])
        self.ax.set_yticklabels(
            ["Sekwencja 1", "Sekwencja 2"] if self.analysis2 is not None else ["Sekwencja 1"]
        )
        self.ax.set_title(f"Rozmieszczenie motywu na osi sekwencji: {motif}")
        self.ax.grid(True)

    def export_all(self):
        try:
            if self.analysis1 is None:
                raise ValueError("Najpierw wykonaj analizę.")

            input_path = self.file1_var.get().strip()
            out_dir = Path(input_path).resolve().parent / "output" if input_path else Path.cwd() / "output"
            out_dir.mkdir(exist_ok=True)

            pdf_filename = filedialog.asksaveasfilename(
                title="Zapisz raport PDF",
                defaultextension=".pdf",
                initialdir=str(out_dir),
                initialfile=f"motif_report_{int(time.time())}.pdf",
                filetypes=[("PDF files", "*.pdf")]
            )

            if not pdf_filename:
                self.log("Anulowano zapis PDF.")
                return

            pdf_path = Path(pdf_filename)
            source1 = self.analysis1["summary"]["source_name"].iloc[0]
            source2 = self.analysis2["summary"]["source_name"].iloc[0] if self.analysis2 is not None else None

            export_pdf_report(
                out_path=pdf_path,
                analysis1=self.analysis1,
                source1=source1,
                analysis2=self.analysis2,
                source2=source2
            )

            self.log("")
            self.log("Zapisano jeden zbiorczy raport PDF:")
            self.log(str(pdf_path.resolve()))

            messagebox.showinfo("Eksport zakończony", f"Raport PDF zapisano tutaj:\n{pdf_path.resolve()}")

        except Exception as e:
            messagebox.showerror("Błąd eksportu", str(e))


if __name__ == "__main__":
    App().mainloop()
