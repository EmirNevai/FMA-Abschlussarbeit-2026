"""
01-Empfehlungen-berechnen.py
============================
TF-IDF-Berechnung und Export der Empfehlungen fuer den gebana-Produktkatalog.

Entspricht der Berechnungslogik aus Empfehlungssystem-Entwicklung.ipynb.
Exportiert in Ergebnisse/:
  - Produktempfehlungen.csv  (produkt_id ; empfohlen_id ; rang ; score)
  - Aehnlichkeitsmatrix.npz  (sparse TF-IDF-Matrix)
  - Produktverzeichnis.csv   (Zeilenzuordnung: artikel_id, name)

Danach 02-Empfehlungen-auswerten.py ausfuehren.
"""

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# KONFIGURATION
# ============================================================

CSV_DATEI     = "Produktbeschreibungen.csv"
CSV_TRENNER   = ";"
CSV_ENCODING  = "utf-8-sig"

USE_STOPWORDS    = True
USE_STEMMING     = True
TOP_N            = 6
MIN_TEXT_LAENGE  = 50   # Mindestlaenge des kombinierten Textes in Zeichen
PARAMETER_SUCHE  = False  # True: vergleicht 4 TF-IDF-Konfigurationen vor der Hauptberechnung

EXPORT_DIR    = Path("Ergebnisse")

# ============================================================
# SPALTENZUORDNUNG (PIM-Export)
# ============================================================

SPALTEN_MAP = {
    "Artikelname (KOM --> Übersetzung)":             "name",
    "Produkttext Shop (KOM --> Übersetzung)":         "beschreibung",
    "NAVIGATIONPATH":                                 "navigationpfad",
    "Anbau - übersetzen (KOM)":                       "anbau",
    "Nachhaltigkeit & Transparenz (KOM)":             "nachhaltigkeit",
    "Verwendung & Zubereitung (KOM --> Übersetzung)": "verwendung",
    "EXTERNALKEY":                                    "artikel_id",
    "TYPE":                                           "typ",
}

# ============================================================
# TEXTVORVERARBEITUNG
# ============================================================

STOP_WORDS = set(stopwords.words("german"))
STOP_WORDS.update({"ca", "kg", "ml", "gr", "mindestens", "haltbar", "bis", "nan"})
_stemmer = SnowballStemmer("german")


def _tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z\u00e4\u00f6\u00fc\u00df\s]", " ", text)
    return text.split()


def preprocess(text: str) -> str:
    """Tokenisierung, optionale Stopwort-Entfernung und Stemming."""
    tokens = _tokenize(text)
    if USE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if USE_STEMMING:
        tokens = [_stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]
    return " ".join(tokens)


def felder_zusammenfuehren(row) -> str:
    teile = [
        str(row.get("name", "") or ""),
        str(row.get("beschreibung", "") or ""),
        str(row.get("anbau", "") or ""),
        str(row.get("nachhaltigkeit", "") or ""),
        str(row.get("verwendung", "") or ""),
    ]
    return " ".join(t for t in teile if t.strip())


# ============================================================
# DATEN LADEN (KI)
# ============================================================

def lade_und_bereite_vor() -> pd.DataFrame:
    """Laedt den Produktkatalog, filtert und bereitet Texte vor."""
    df = pd.read_csv(CSV_DATEI, sep=CSV_TRENNER, encoding=CSV_ENCODING, dtype=str)
    df = df.rename(columns=SPALTEN_MAP)

    # Nur echte Artikel, keine Kategorieknoten
    df = df[df["typ"] == "Artikel"].copy()
    df = df[df["name"].notna() & (df["name"].str.strip() != "")].copy()
    df = df.reset_index(drop=True)

    df["text_combined"] = df.apply(felder_zusammenfuehren, axis=1)
    df["text_clean"]    = df["text_combined"].apply(preprocess)

    # Produkte mit zu kurzem kombiniertem Text ausschliessen
    df["text_laenge"] = df["text_combined"].str.len()
    df = df[df["text_laenge"] >= MIN_TEXT_LAENGE].copy().reset_index(drop=True)
    return df


# ============================================================
# TF-IDF UND KOSINUS-AEHNLICHKEIT
# ============================================================

def berechne_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-z\u00e4\u00f6\u00fc\u00df]+",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_clean"])
    cosim_matrix = cosine_similarity(tfidf_matrix)
    return tfidf_matrix, cosim_matrix, vectorizer


# ============================================================
# EMPFEHLUNGEN GENERIEREN
# ============================================================

def generiere_empfehlungen(df: pd.DataFrame, cosim_matrix: np.ndarray, top_n: int = TOP_N) -> pd.DataFrame:
    zeilen = []
    for idx, row in df.iterrows():
        artikel_id = str(row["artikel_id"]).strip()
        sim = cosim_matrix[idx]
        sortiert = [i for i in sim.argsort()[::-1] if i != idx][:top_n]
        for rang, emp_idx in enumerate(sortiert, start=1):
            zeilen.append({
                "produkt_id":   artikel_id,
                "empfohlen_id": str(df.at[emp_idx, "artikel_id"]).strip(),
                "rang":         rang,
                "score":        round(float(sim[emp_idx]), 6),
            })
    return pd.DataFrame(zeilen)


# ============================================================
# PARAMETER-SUCHE (KI)
# ============================================================

def einfache_parametersuche(df: pd.DataFrame) -> None:
    if not PARAMETER_SUCHE:
        return

    konfigurationen = [
        {"min_df": 1, "max_df": 1.00, "sublinear_tf": False, "label": "Basis       "},
        {"min_df": 2, "max_df": 0.95, "sublinear_tf": True,  "label": "Standard    "},
        {"min_df": 3, "max_df": 0.90, "sublinear_tf": True,  "label": "Strenger    "},
        {"min_df": 2, "max_df": 0.80, "sublinear_tf": True,  "label": "MaxDF=0.80  "},
    ]

    stichprobe = min(100, len(df))
    print("\n  Parameterstudie (PARAMETER_SUCHE=True):")
    print(f"  {'Konfiguration':<14}  {'Vokabular':>9}  {'Ø Top-Sim':>9}")
    print("  " + "-" * 36)
    for cfg in konfigurationen:
        vect = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-z\u00e4\u00f6\u00fc\u00df]+",
            min_df=cfg["min_df"], max_df=cfg["max_df"], sublinear_tf=cfg["sublinear_tf"],
        )
        matrix = vect.fit_transform(df["text_clean"])
        cosim  = cosine_similarity(matrix[:stichprobe], matrix)
        np.fill_diagonal(cosim[:stichprobe, :stichprobe], 0)
        top_sim = float(cosim.max(axis=1).mean())
        print(f"  {cfg['label']}  {matrix.shape[1]:>9,}  {top_sim:>9.3f}")
    print()


# ============================================================
# EXPORT
# ============================================================

def exportiere(df: pd.DataFrame, tfidf_matrix, df_empfehlungen: pd.DataFrame) -> None:
    EXPORT_DIR.mkdir(exist_ok=True)

    empf_pfad = EXPORT_DIR / "Produktempfehlungen.csv"
    df_empfehlungen.to_csv(empf_pfad, index=False, sep=";", encoding="utf-8-sig")
    print(f"  Empfehlungen:   {empf_pfad}  ({len(df_empfehlungen):,} Zeilen)")

    npz_pfad = EXPORT_DIR / "Aehnlichkeitsmatrix.npz"
    scipy.sparse.save_npz(str(npz_pfad), tfidf_matrix)
    print(f"  TF-IDF-Matrix:  {npz_pfad}  {tfidf_matrix.shape}")

    ids_pfad = EXPORT_DIR / "Produktverzeichnis.csv"
    df[["artikel_id", "name"]].to_csv(ids_pfad, index=False, sep=";", encoding="utf-8-sig")
    print(f"  Produkt-IDs:    {ids_pfad}  ({len(df):,} Produkte)")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 55)
    print("  TF-IDF BERECHNUNG  —  gebana AG")
    print("=" * 55)

    print("\n[1/3] Daten laden, vorverarbeiten und filtern ...")
    df = lade_und_bereite_vor()
    print(f"  Produkte im Modell: {len(df)} (kombinierter Text >= {MIN_TEXT_LAENGE} Zeichen)")

    einfache_parametersuche(df)

    print("\n[2/3] TF-IDF und Kosinus-Aehnlichkeit berechnen ...")
    tfidf_matrix, cosim_matrix, vectorizer = berechne_tfidf(df)
    sparsity = 1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
    print(f"  Matrix: {tfidf_matrix.shape[0]} Produkte x {tfidf_matrix.shape[1]} Terme")
    print(f"  Sparsity: {sparsity:.1%}")

    df_empfehlungen = generiere_empfehlungen(df, cosim_matrix)
    print(f"  Empfehlungen generiert: {len(df_empfehlungen):,} Zeilen")

    print("\n[3/3] Exportieren nach Ergebnisse/ ...")
    exportiere(df, tfidf_matrix, df_empfehlungen)

    print(f"\nFertig. Naechster Schritt: python 02-Empfehlungen-auswerten.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
