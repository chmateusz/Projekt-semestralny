import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VALID_DNA = set("ACGTN")



# Wczytanie pliku

def read_sequence(path):
    text = Path(path).read_text(encoding="utf-8").strip()

    if text.startswith(">"):
        lines = text.splitlines()[1:]
    else:
        lines = text.splitlines()

    seq = "".join(lines).replace(" ", "").upper()

    if not seq:
        raise ValueError("Brak sekwencji.")

    if set(seq) - VALID_DNA:
        raise ValueError("Niepoprawne znaki w sekwencji.")

    return seq



# Wyszukiwanie motywu

def find_positions(seq, motif):
    pattern = re.compile(rf"(?=({re.escape(motif)}))")
    return [m.start() for m in pattern.finditer(seq)]

# Segmentacja

def segment(seq_len, positions, bin_size):
    n_bins = int(np.ceil(seq_len / bin_size))
    counts = np.zeros(n_bins, dtype=int)

    for p in positions:
        counts[p // bin_size] += 1

    return pd.DataFrame({
        "bin": np.arange(n_bins),
        "start": np.arange(n_bins) * bin_size + 1,
        "end": np.minimum((np.arange(n_bins) + 1) * bin_size, seq_len),
        "count": counts
    })

# Wykres

def plot_bins(df, motif):
    plt.bar(df["bin"], df["count"])
    plt.xlabel("Segment")
    plt.ylabel("Liczba motywów")
    plt.title(f"Motyw {motif}")
    plt.tight_layout()
    plt.savefig("motif_plot.png")
    plt.show()

# MAIN

def main():
    path = input("Plik FASTA/TXT: ").strip()
    motif = input("Motyw (np. ATG): ").upper().strip()
    bin_size = int(input("Bin size (np. 100): ").strip())

    seq = read_sequence(path)
    positions = find_positions(seq, motif)
    bins_df = segment(len(seq), positions, bin_size)

    # zapis
    Path("output").mkdir(exist_ok=True)
    pd.DataFrame({"position_1": [p + 1 for p in positions]}).to_csv(
        "output/hits.csv", index=False
    )
    bins_df.to_csv("output/bins.csv", index=False)

    print("\nWynik:")
    print("Długość sekwencji:", len(seq))
    print("Liczba wystąpień:", len(positions))
    print("Pierwsze pozycje:", [p + 1 for p in positions[:10]])

    plot_bins(bins_df, motif)


if __name__ == "__main__":
    main()
