"""
02-Empfehlungen-auswerten.py
=============================
Offline-Evaluation des TF-IDF-Empfehlungssystems anhand von
Co-Purchase Ground Truth aus historischen Bestelldaten.

Gesamtevaluation (alle Bestellungen) + sequentieller Train/Test-Split (80/20).

Hypothese H1: Mit TF-IDF kann eine durchschnittliche Precision@6
              von mindestens 0.6 erreicht werden.
"""

import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# KONFIGURATION
# ============================================================

RANDOM_STATE = 0
TOP_N        = 6
TRAIN_ANTEIL = 0.8
MIN_TEXT_LAENGE = 50
DATUM_SPALTE = None
SCHWELLENWERTE  = [0, 25, 50, 100, 150]

HYPOTHESE_PRECISION_MEAN   = 0.6
HYPOTHESE_MINDEST_COVERAGE = 0.50

EMPFEHLUNGEN_CSV    = Path("Ergebnisse/Produktempfehlungen.csv")
EMPF_TRENNER        = ";"

BESTELL_CSV         = "Bestellhistorie-anonymisiert.csv"
BESTELL_TRENNER     = ";"
BESTELL_ENCODING    = "utf-8-sig"

UUID_SKU_CSV        = "Zuordnung-ID-Artikelnummer.csv"
ARTIKELSTRUKTUR_CSV = "Produktliste.csv"
PRODUKT_DATEI       = Path("Produktbeschreibungen.csv")

AUSGABE_DIR         = Path("Ergebnisse")
PDF_DATEI           = AUSGABE_DIR / "Ergebnisbericht.pdf"
METRIKEN_CSV        = AUSGABE_DIR / "Qualitaetskennzahlen-Gesamt.csv"
PRODUKT_CSV         = AUSGABE_DIR / "Qualitaetskennzahlen-pro-Produkt.csv"
DATENQUALITAET_CSV  = AUSGABE_DIR / "Datenqualitaet-Uebersicht.csv"
FILTER_CSV          = AUSGABE_DIR / "Vergleich-mit-ohne-Filter.csv"
SENSITIVITAET_CSV   = AUSGABE_DIR / "Schwellenwert-Analyse.csv"
IDENTISCHE_CSV      = AUSGABE_DIR / "Produkte-mit-identischem-Text.csv"
AUSGESCHLOSSENE_CSV = AUSGABE_DIR / "Ausgeschlossene-Produkte.csv"

# Farbpalette
SEABORN_STYLE   = "whitegrid"
FARBE_PRECISION = "#1f77b4"
FARBE_RECALL    = "#2ca02c"
FARBE_F1        = "#9467bd"
PALETTE_POS     = "#2ca02c"
PALETTE_NEG     = "#d62728"

# Spaltenmapping Produktdaten
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

TEXT_SPALTEN = ["name", "beschreibung", "anbau", "nachhaltigkeit", "verwendung"]


# ============================================================
# 1. EMPFEHLUNGEN LADEN
# ============================================================

def lade_empfehlungen(pfad: Path = EMPFEHLUNGEN_CSV) -> dict[str, list[str]]:
    if not pfad.exists():
        sys.exit(
            f"FEHLER: {pfad} nicht gefunden.\n"
            "Bitte zuerst 01-Empfehlungen-berechnen.py ausfuehren."
        )

    df = pd.read_csv(pfad, sep=EMPF_TRENNER, encoding="utf-8-sig", dtype=str)
    df["rang"] = pd.to_numeric(df["rang"], errors="coerce").fillna(999).astype(int)
    df = df.sort_values(["produkt_id", "rang"])

    empfehlungen: dict[str, list[str]] = {}
    for produkt_id, gruppe in df.groupby("produkt_id"):
        empfehlungen[str(produkt_id).strip()] = gruppe["empfohlen_id"].str.strip().tolist()

    return empfehlungen


# ============================================================
# 2. PRODUKTDATEN LADEN & FILTERN
# ============================================================

def lade_produktdaten(pfad: Path = PRODUKT_DATEI) -> pd.DataFrame:
    if not pfad.exists():
        sys.exit(f"FEHLER: Produktdatei nicht gefunden: {pfad}")

    df = pd.read_csv(pfad, sep=";", encoding="utf-8-sig", dtype=str)
    df = df.rename(columns={k: v for k, v in SPALTEN_MAP.items() if k in df.columns})

    if "typ" in df.columns:
        df = df[df["typ"] == "Artikel"].copy()

    for col in TEXT_SPALTEN:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()
        else:
            df[col] = ""

    if "artikel_id" not in df.columns:
        sys.exit("FEHLER: Spalte 'EXTERNALKEY' nicht in Produktdaten.")

    df["artikel_id"] = df["artikel_id"].fillna("").str.strip()
    df = df[df["artikel_id"] != ""].reset_index(drop=True)

    df["text_laenge"] = df[TEXT_SPALTEN].apply(
        lambda zeile: len(" ".join(str(v) for v in zeile if str(v).strip())), axis=1
    )

    return df


def filter_produkte_nach_textlaenge(
    df_produkte: pd.DataFrame,
    text_spalten: list[str],
    min_laenge: int = MIN_TEXT_LAENGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if "text_laenge" not in df_produkte.columns:
        df_produkte = df_produkte.copy()
        df_produkte["text_laenge"] = df_produkte[text_spalten].apply(
            lambda zeile: len(" ".join(str(v) for v in zeile if str(v).strip())), axis=1
        )

    maske = df_produkte["text_laenge"] >= min_laenge
    return (
        df_produkte[maske].reset_index(drop=True),
        df_produkte[~maske].reset_index(drop=True),
    )


# ============================================================
# 3. ID-MAPPINGS LADEN (KI)
# ============================================================

def lade_uuid_zu_sku(pfad: str = UUID_SKU_CSV) -> dict[str, str]:
    if not Path(pfad).exists():
        sys.exit(f"FEHLER: UUID-SKU-Mapping nicht gefunden: {pfad}")
    df = pd.read_csv(pfad, sep=";", encoding="utf-8", dtype=str)
    return {
        row["id"].strip(): row["Artikelnummer"].strip()
        for _, row in df.iterrows()
        if pd.notna(row["id"]) and pd.notna(row["Artikelnummer"])
    }


def lade_child_zu_parent(pfad: str = ARTIKELSTRUKTUR_CSV) -> dict[str, str]:
 
    if not Path(pfad).exists():
        sys.exit(f"FEHLER: Artikelstruktur nicht gefunden: {pfad}")
    df = pd.read_csv(pfad, sep=";", encoding="utf-8-sig", dtype=str)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        sku = str(row["ExternalKey"]).strip() if pd.notna(row["ExternalKey"]) else None
        if not sku:
            continue
        typ = str(row.get("Artikeltyp", "")).strip()
        parent_raw = row.get("ParentExternalKey", None)
        if typ == "Child" and pd.notna(parent_raw) and str(parent_raw).strip() not in ("", "NICHT_GEFUNDEN", "nan"):
            parent_str = str(parent_raw).strip()
            if parent_str.endswith(".0"):
                parent_str = parent_str[:-2]
            mapping[sku] = parent_str
        else:
            mapping[sku] = sku
    return mapping


def uebersetze_produkt_ids(
    df_bestellungen: pd.DataFrame,
    uuid_zu_sku: dict[str, str],
    child_zu_parent: dict[str, str],
) -> pd.DataFrame:

    nicht_gemappt = 0

    def uebersetze(produkte: list[str]) -> list[str]:
        nonlocal nicht_gemappt
        parent_skus: list[str] = []
        gesehen: set[str] = set()
        for uuid in produkte:
            sku = uuid_zu_sku.get(uuid)
            if sku is None:
                nicht_gemappt += 1
                continue
            parent = child_zu_parent.get(sku, sku)
            if parent not in gesehen:
                gesehen.add(parent)
                parent_skus.append(parent)
        return parent_skus

    df_bestellungen = df_bestellungen.copy()
    df_bestellungen["produkte"] = df_bestellungen["produkte"].apply(uebersetze)

    if nicht_gemappt:
        print(f"  WARNUNG: {nicht_gemappt:,} UUIDs konnten nicht ins SKU-Mapping uebersetzt werden.")

    return df_bestellungen


def berechne_mapping_qualitaet(
    df_bestellungen_roh: pd.DataFrame,
    uuid_zu_sku: dict[str, str],
    child_zu_parent: dict[str, str],
) -> dict:

    alle_uuids: set[str] = set()
    for inhalt in df_bestellungen_roh["Inhalt"].dropna():
        for segment in str(inhalt).split("|"):
            match = re.match(r"^\d+x\s+(.+)$", segment.strip())
            if match:
                alle_uuids.add(match.group(1).strip())

    n_total    = len(alle_uuids)
    n_gemappt  = sum(1 for u in alle_uuids if u in uuid_zu_sku)
    n_child    = sum(
        1 for u in alle_uuids
        if u in uuid_zu_sku
        and child_zu_parent.get(uuid_zu_sku[u], uuid_zu_sku[u]) != uuid_zu_sku[u]
    )

    return {
        "n_uuid_total":         n_total,
        "n_uuid_gemappt":       n_gemappt,
        "n_uuid_nicht_gemappt": n_total - n_gemappt,
        "mapping_rate":         n_gemappt / n_total if n_total else 0.0,
        "n_child_zu_parent":    n_child,
    }


# ============================================================
# 4. BESTELLDATEN PARSEN (KI)
# ============================================================

def parse_inhalt(inhalt: str) -> list[str]:

    if not isinstance(inhalt, str) or not inhalt.strip():
        return []
    produkte: list[str] = []
    for segment in inhalt.split("|"):
        match = re.match(r"^\d+x\s+(.+)$", segment.strip())
        if match:
            pid = match.group(1).strip()
            if pid:
                produkte.append(pid)
    return produkte


def lade_bestellungen() -> pd.DataFrame:
    if not Path(BESTELL_CSV).exists():
        sys.exit(f"FEHLER: Bestelldatei nicht gefunden: {BESTELL_CSV}")

    df = pd.read_csv(BESTELL_CSV, sep=BESTELL_TRENNER, encoding=BESTELL_ENCODING, dtype=str)
    df["produkte"] = df["Inhalt"].apply(
        lambda x: list(dict.fromkeys(parse_inhalt(str(x))))
    )
    return df


# ============================================================
# 5. GROUND TRUTH AUFBAUEN (KI)
# ============================================================

def build_ground_truth(df_bestellungen: pd.DataFrame) -> dict[str, set[str]]:

    ground_truth: dict[str, set[str]] = defaultdict(set)
    for produkte in df_bestellungen["produkte"]:
        if len(produkte) < 2:
            continue
        produkte_set = set(produkte)
        for p in produkte_set:
            ground_truth[p] |= produkte_set - {p}
    return dict(ground_truth)


def build_kategorie_ground_truth(df_produkte: pd.DataFrame) -> dict[str, set[str]]:

    if "navigationpfad" not in df_produkte.columns:
        return {}

    df = df_produkte[["artikel_id", "navigationpfad"]].copy()
    df["top_kategorie"] = df["navigationpfad"].str.split(">").str[0].str.strip()
    df = df[df["top_kategorie"].str.len() > 0]

    gt: dict[str, set[str]] = {}
    for _, gruppe in df.groupby("top_kategorie"):
        ids = set(gruppe["artikel_id"].astype(str))
        if len(ids) < 2:
            continue
        for pid in ids:
            gt[pid] = ids - {pid}
    return gt


# ============================================================
# 6. EVALUATION (Teilweise KI)
# ============================================================

def evaluate_at_k(
    ground_truth: dict[str, set[str]],
    empfehlungen: dict[str, list[str]],
    k: int = TOP_N,
) -> pd.DataFrame:

    ergebnisse = []
    for produkt_id, gt_set in ground_truth.items():
        if not gt_set or produkt_id not in empfehlungen:
            continue

        top_k = empfehlungen[produkt_id][:k]
        tp    = sum(1 for p in top_k if p in gt_set)
        fp    = len(top_k) - tp
        fn    = len(gt_set) - tp

        precision = tp / k
        recall    = tp / len(gt_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        ergebnisse.append({
            "produkt_id":  produkt_id,
            "tp":          tp,
            "fp":          fp,
            "fn":          fn,
            "precision":   precision,
            "recall":      recall,
            "f1":          f1,
            "hits":        tp,
            "gt_groesse":  len(gt_set),
            "n_empfohlen": len(top_k),
        })

    return pd.DataFrame(ergebnisse)


# ============================================================
# 7. TRAIN/TEST-SPLIT (sequentiell)
# ============================================================

def train_test_evaluation(
    df_bestellungen: pd.DataFrame,
    empfehlungen: dict[str, list[str]],
    train_anteil: float = TRAIN_ANTEIL,
    datum_spalte: str | None = DATUM_SPALTE,
) -> dict:

    df = df_bestellungen.copy()
    if datum_spalte and datum_spalte in df.columns:
        df[datum_spalte] = pd.to_datetime(df[datum_spalte], errors="coerce")
        df = df.sort_values(datum_spalte).reset_index(drop=True)

    split     = int(len(df) * train_anteil)
    df_train  = df.iloc[:split]
    df_test   = df.iloc[split:]

    gt_train = build_ground_truth(df_train)
    gt_test  = build_ground_truth(df_test)

    df_eval_train = evaluate_at_k(gt_train, empfehlungen)
    df_eval_test  = evaluate_at_k(gt_test,  empfehlungen)

    return {
        "df_train":     df_eval_train,
        "df_test":      df_eval_test,
        "n_train":      len(df_train),
        "n_test":       len(df_test),
        "train_anteil": train_anteil,
    }


# ============================================================
# 8. COVERAGE-METRIKEN
# ============================================================

def berechne_coverage(df_eval: pd.DataFrame, empfehlungen: dict, k: int = TOP_N) -> dict:

    n = len(df_eval)
    if n == 0:
        return {}

    n_genau_k       = sum(1 for pid in df_eval["produkt_id"] if len(empfehlungen.get(pid, [])) >= k)
    n_min_1_treffer = int((df_eval["precision"] > 0).sum())
    n_min_2_treffer = int((df_eval["hits"] >= 2).sum())
    n_p05           = int((df_eval["precision"] >= 0.5).sum())
    n_p06           = int((df_eval["precision"] >= 0.6).sum())

    return {
        "n_evaluierbar":        n,
        "anteil_genau_k_empf":  n_genau_k / n,
        "anteil_min_1_treffer": n_min_1_treffer / n,
        "anteil_min_2_treffer": n_min_2_treffer / n,
        "anteil_p_gte_05":      n_p05 / n,
        "anteil_p_gte_06":      n_p06 / n,
        "n_genau_k_empf":       n_genau_k,
        "n_min_1_treffer":      n_min_1_treffer,
        "n_min_2_treffer":      n_min_2_treffer,
        "n_p_gte_05":           n_p05,
        "n_p_gte_06":           n_p06,
    }


# ============================================================
# 9. VERTEILUNGSSTATISTIKEN
# ============================================================

def berechne_statistiken(df_eval: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Berechnet erweiterte Verteilungsstatistiken fuer alle Metriken."""
    stats = {}
    for col in ["precision", "recall", "f1"]:
        werte = df_eval[col]
        stats[col] = {
            "mean":         werte.mean(),
            "median":       werte.median(),
            "std":          werte.std(),
            "q25":          werte.quantile(0.25),
            "q75":          werte.quantile(0.75),
            "min":          werte.min(),
            "max":          werte.max(),
            "anteil_null":  (werte == 0).mean(),
            "anteil_gte06": (werte >= 0.6).mean(),
        }
    return stats


# ============================================================
# 10. HYPOTHESENPRÜFUNG
# ============================================================

def pruefe_hypothese(df_eval: pd.DataFrame, coverage: dict) -> dict:

    p_mean = df_eval["precision"].mean()
    cov    = coverage.get("anteil_min_1_treffer", 0.0)

    ok_mean    = p_mean >= HYPOTHESE_PRECISION_MEAN
    ok_cov     = cov    >= HYPOTHESE_MINDEST_COVERAGE
    bestaetigt = ok_mean and ok_cov

    return {
        "precision_mean":          p_mean,
        "coverage":                cov,
        "precision_mean_erreicht": ok_mean,
        "coverage_erreicht":       ok_cov,
        "ergebnis":                "bestaetigt" if bestaetigt else "nicht bestaetigt",
    }


# ============================================================
# 11. DATENQUALITÄTS-ANALYSE (KI)
# ============================================================

def analysiere_textqualitaet(df: pd.DataFrame, text_spalten: list[str]) -> dict:

    df = df.copy()
    if "text_laenge" not in df.columns:
        df["text_laenge"] = df[text_spalten].apply(
            lambda zeile: len(" ".join(str(v) for v in zeile if str(v).strip())), axis=1
        )

    befuellungsraten: dict[str, float] = {}
    for col in text_spalten:
        if col in df.columns:
            befuellungsraten[col] = float((df[col].str.len() > 0).mean())

    laengen = df["text_laenge"]
    verteilung = {
        "leer_0":        int((laengen == 0).sum()),
        "kurz_1_49":     int(((laengen >= 1)   & (laengen < 50)).sum()),
        "mittel_50_99":  int(((laengen >= 50)  & (laengen < 100)).sum()),
        "mittel_100_299":int(((laengen >= 100) & (laengen < 300)).sum()),
        "lang_300_plus": int((laengen >= 300).sum()),
    }

    return {
        "n_total":          len(df),
        "textlaengen":      laengen.values,
        "befuellungsraten": befuellungsraten,
        "verteilung":       verteilung,
        "mean_laenge":      float(laengen.mean()),
        "median_laenge":    float(laengen.median()),
    }


def analysiere_identische_texte(df: pd.DataFrame, text_spalten: list[str]) -> pd.DataFrame:

    df = df.copy()
    df["_text"] = df[text_spalten].apply(
        lambda zeile: " ".join(str(v) for v in zeile if str(v).strip()), axis=1
    )
    df_nonempty = df[df["_text"].str.len() > 0]

    zeilen = []
    gruppe_id = 1
    for text, gruppe in df_nonempty.groupby("_text"):
        if len(gruppe) < 2:
            continue
        namen = gruppe["name"].tolist() if "name" in gruppe.columns else []
        zeilen.append({
            "gruppe_id":      gruppe_id,
            "n_artikel":      len(gruppe),
            "beispiel_name_1": namen[0] if len(namen) > 0 else "",
            "beispiel_name_2": namen[1] if len(namen) > 1 else "",
            "text_laenge":    len(str(text)),
            "textauszug":     str(text)[:100],
        })
        gruppe_id += 1

    return pd.DataFrame(zeilen)


def analysiere_ausgeschlossene(df: pd.DataFrame, text_spalten: list[str]) -> pd.DataFrame:

    nicht_name_felder = [c for c in text_spalten if c != "name"]
    zeilen = []
    for _, zeile in df.iterrows():
        sku        = str(zeile.get("artikel_id", "")).strip()
        name_val   = str(zeile.get("name", "")).strip()
        text_laenge = int(zeile.get("text_laenge", 0))

        n_befuellt = sum(1 for col in text_spalten if str(zeile.get(col, "")).strip())

        nav = str(zeile.get("navigationpfad", "")).strip()
        navigationskategorie = nav.split(">")[0].strip() if ">" in nav else nav[:60]

        ist_archiv = False

        if n_befuellt == 0:
            grund = "komplett leer"
        elif name_val and all(not str(zeile.get(c, "")).strip() for c in nicht_name_felder):
            grund = "nur Name"
        else:
            grund = "Text zu kurz"

        zeilen.append({
            "sku":                 sku,
            "name":                name_val,
            "text_laenge":         text_laenge,
            "n_befuellte_felder":  n_befuellt,
            "navigationskategorie":navigationskategorie,
            "ist_archiv":          ist_archiv,
            "grund":               grund,
        })
    return pd.DataFrame(zeilen)


def vergleiche_mit_ohne_filter(
    ground_truth: dict[str, set[str]],
    empfehlungen: dict[str, list[str]],
    df_ungefiltert: pd.DataFrame,
    df_gefiltert: pd.DataFrame,
) -> pd.DataFrame:

    ids_ungefiltert = set(df_ungefiltert["artikel_id"].dropna().astype(str))
    ids_gefiltert   = set(df_gefiltert["artikel_id"].dropna().astype(str))

    gt_ohne = {k: v for k, v in ground_truth.items() if k in ids_ungefiltert}
    gt_mit  = {k: v for k, v in ground_truth.items() if k in ids_gefiltert}

    df_ohne = evaluate_at_k(gt_ohne, empfehlungen)
    df_mit  = evaluate_at_k(gt_mit,  empfehlungen)

    def mittel(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if len(df) > 0 else 0.0

    def coverage(df: pd.DataFrame) -> float:
        return float((df["precision"] > 0).mean()) if len(df) > 0 else 0.0

    zeilen = [
        {"metrik": "precision_mean", "ohne_filter": mittel(df_ohne, "precision"), "mit_filter": mittel(df_mit, "precision")},
        {"metrik": "recall_mean",    "ohne_filter": mittel(df_ohne, "recall"),    "mit_filter": mittel(df_mit, "recall")},
        {"metrik": "f1_mean",        "ohne_filter": mittel(df_ohne, "f1"),        "mit_filter": mittel(df_mit, "f1")},
        {"metrik": "coverage",       "ohne_filter": coverage(df_ohne),            "mit_filter": coverage(df_mit)},
    ]
    df_vgl = pd.DataFrame(zeilen)
    df_vgl["differenz"] = df_vgl["mit_filter"] - df_vgl["ohne_filter"]
    return df_vgl


def sensitivitaetsanalyse(
    ground_truth: dict[str, set[str]],
    empfehlungen: dict[str, list[str]],
    df_produkte: pd.DataFrame,
    text_spalten: list[str],
    schwellenwerte: list[int] = SCHWELLENWERTE,
) -> pd.DataFrame:

    zeilen = []
    for schwellenwert in schwellenwerte:
        df_gef, _ = filter_produkte_nach_textlaenge(df_produkte, text_spalten, schwellenwert)
        ids = set(df_gef["artikel_id"].dropna().astype(str))
        gt_gefiltert = {k: v for k, v in ground_truth.items() if k in ids}
        df_ev = evaluate_at_k(gt_gefiltert, empfehlungen)

        if len(df_ev) == 0:
            zeilen.append({"schwellenwert": schwellenwert, "n_artikel": 0,
                           "precision_mean": 0.0, "recall_mean": 0.0,
                           "f1_mean": 0.0, "coverage": 0.0})
            continue

        zeilen.append({
            "schwellenwert":  schwellenwert,
            "n_artikel":      len(df_ev),
            "precision_mean": float(df_ev["precision"].mean()),
            "recall_mean":    float(df_ev["recall"].mean()),
            "f1_mean":        float(df_ev["f1"].mean()),
            "coverage":       float((df_ev["precision"] > 0).mean()),
        })
    return pd.DataFrame(zeilen)


# ============================================================
# 12. SEGMENTIERTE EVALUATION (Hilfsfunktion fuer Grafiken)
# ============================================================

def segmentierte_evaluation(df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    q25 = df_eval["gt_groesse"].quantile(0.25)
    q75 = df_eval["gt_groesse"].quantile(0.75)

    def _segment(gt: float) -> str:
        if gt <= q25:
            return "Klein"
        elif gt <= q75:
            return "Mittel"
        return "Gross"

    segmente = df_eval["gt_groesse"].apply(_segment)
    df_mit_seg = df_eval.copy()
    df_mit_seg["segment"] = segmente

    seg_stats = (
        df_mit_seg.groupby("segment")
        .agg(
            n_produkte=("produkt_id", "count"),
            precision_mean=("precision", "mean"),
            precision_median=("precision", "median"),
            recall_mean=("recall", "mean"),
            recall_median=("recall", "median"),
            f1_mean=("f1", "mean"),
            f1_median=("f1", "median"),
        )
        .reset_index()
    )

    reihenfolge = ["Klein", "Mittel", "Gross"]
    seg_stats["segment"] = pd.Categorical(seg_stats["segment"], categories=reihenfolge, ordered=True)
    seg_stats = seg_stats.sort_values("segment").reset_index(drop=True)

    return seg_stats, segmente


# ============================================================
# 13. CSV-EXPORTS
# ============================================================

def exportiere_metriken(
    df_eval: pd.DataFrame,
    coverage: dict,
    hypothese: dict,
    dq_info: dict,
    df_filter_vgl: pd.DataFrame,
    df_sensitivitaet: pd.DataFrame,
    df_identische: pd.DataFrame,
    df_ausgeschlossen: pd.DataFrame,
    train_test: dict,
    n_bestellungen: int = 0,
    n_bestellungen_ge2: int = 0,
    mapping_qualitaet: dict | None = None,
    kategorie_precision: float | None = None,
    pfad: str = str(METRIKEN_CSV),
) -> None:
    """Exportiert alle Metriken in output/metriken.csv (Format: kategorie, metrik, wert)."""
    stats = berechne_statistiken(df_eval)
    zeilen: list[tuple] = []

    mittlere_gt = float(df_eval["gt_groesse"].mean()) if len(df_eval) > 0 else 0.0

    # gesamt
    zeilen += [
        ("gesamt", "precision_mean",         stats["precision"]["mean"]),
        ("gesamt", "precision_median",        stats["precision"]["median"]),
        ("gesamt", "precision_std",           stats["precision"]["std"]),
        ("gesamt", "precision_min",           stats["precision"]["min"]),
        ("gesamt", "precision_max",           stats["precision"]["max"]),
        ("gesamt", "precision_anteil_null",   stats["precision"]["anteil_null"]),
        ("gesamt", "precision_anteil_ge_06",  stats["precision"]["anteil_gte06"]),
        ("gesamt", "recall_mean",             stats["recall"]["mean"]),
        ("gesamt", "recall_median",           stats["recall"]["median"]),
        ("gesamt", "recall_std",              stats["recall"]["std"]),
        ("gesamt", "recall_min",              stats["recall"]["min"]),
        ("gesamt", "recall_max",              stats["recall"]["max"]),
        ("gesamt", "f1_mean",                 stats["f1"]["mean"]),
        ("gesamt", "f1_median",               stats["f1"]["median"]),
        ("gesamt", "f1_std",                  stats["f1"]["std"]),
        ("gesamt", "f1_min",                  stats["f1"]["min"]),
        ("gesamt", "f1_max",                  stats["f1"]["max"]),
        ("gesamt", "tp_mean",                 float(df_eval["tp"].mean())),
        ("gesamt", "tp_median",               float(df_eval["tp"].median())),
        ("gesamt", "fp_mean",                 float(df_eval["fp"].mean())),
        ("gesamt", "fp_median",               float(df_eval["fp"].median())),
        ("gesamt", "fn_mean",                 float(df_eval["fn"].mean())),
        ("gesamt", "fn_median",               float(df_eval["fn"].median())),
        ("gesamt", "coverage_k_empfehlungen", coverage.get("anteil_genau_k_empf", 0)),
        ("gesamt", "coverage_mind_1_treffer", coverage.get("anteil_min_1_treffer", 0)),
        ("gesamt", "coverage_mind_2_treffer", coverage.get("anteil_min_2_treffer", 0)),
        ("gesamt", "coverage_ge_05",          coverage.get("anteil_p_gte_05", 0)),
        ("gesamt", "coverage_ge_06",          coverage.get("anteil_p_gte_06", 0)),
        ("gesamt", "n_evaluierbar",           coverage.get("n_evaluierbar", 0)),
        ("gesamt", "n_ausgeschlossen",        len(df_ausgeschlossen)),
        ("gesamt", "n_bestellungen",          n_bestellungen),
        ("gesamt", "n_bestellungen_ge2",      n_bestellungen_ge2),
        ("gesamt", "mittlere_gt_groesse",     mittlere_gt),
    ]

    # datenqualitaet
    n_identische_gruppen = len(df_identische)
    zeilen += [
        ("datenqualitaet", "n_total",               dq_info.get("n_total", 0)),
        ("datenqualitaet", "n_gefiltert",            dq_info.get("n_total", 0) - len(df_ausgeschlossen)),
        ("datenqualitaet", "n_ausgeschlossen",       len(df_ausgeschlossen)),
        ("datenqualitaet", "n_identische_gruppen",   n_identische_gruppen),
        ("datenqualitaet", "min_text_laenge",        MIN_TEXT_LAENGE),
        ("datenqualitaet", "mean_text_laenge",       dq_info.get("mean_laenge", 0)),
        ("datenqualitaet", "median_text_laenge",     dq_info.get("median_laenge", 0)),
    ]
    for feld, rate in dq_info.get("befuellungsraten", {}).items():
        zeilen.append(("datenqualitaet", f"befuellungsrate_{feld}", rate))
    for _, vgl_zeile in df_filter_vgl.iterrows():
        metrik = vgl_zeile["metrik"]
        zeilen += [
            ("datenqualitaet", f"vorher_{metrik}", vgl_zeile["ohne_filter"]),
            ("datenqualitaet", f"nachher_{metrik}", vgl_zeile["mit_filter"]),
            ("datenqualitaet", f"differenz_{metrik}", vgl_zeile["differenz"]),
        ]
    for _, s_zeile in df_sensitivitaet.iterrows():
        sw = int(s_zeile["schwellenwert"])
        zeilen += [
            ("datenqualitaet", f"sensitivitaet_{sw}_n_artikel",      s_zeile["n_artikel"]),
            ("datenqualitaet", f"sensitivitaet_{sw}_precision_mean", s_zeile["precision_mean"]),
            ("datenqualitaet", f"sensitivitaet_{sw}_coverage",       s_zeile["coverage"]),
        ]

    # train_test
    df_tr = train_test.get("df_train", pd.DataFrame())
    df_te = train_test.get("df_test",  pd.DataFrame())
    zeilen += [
        ("train_test", "train_anteil",         train_test.get("train_anteil", TRAIN_ANTEIL)),
        ("train_test", "n_train",              train_test.get("n_train", 0)),
        ("train_test", "n_test",               train_test.get("n_test", 0)),
        ("train_test", "train_precision_mean", float(df_tr["precision"].mean()) if len(df_tr) > 0 else 0.0),
        ("train_test", "train_recall_mean",    float(df_tr["recall"].mean())    if len(df_tr) > 0 else 0.0),
        ("train_test", "train_f1_mean",        float(df_tr["f1"].mean())        if len(df_tr) > 0 else 0.0),
        ("train_test", "test_precision_mean",  float(df_te["precision"].mean()) if len(df_te) > 0 else 0.0),
        ("train_test", "test_recall_mean",     float(df_te["recall"].mean())    if len(df_te) > 0 else 0.0),
        ("train_test", "test_f1_mean",         float(df_te["f1"].mean())        if len(df_te) > 0 else 0.0),
    ]

    # hypothese
    zeilen += [
        ("hypothese", "precision_mean_erreicht", hypothese["precision_mean_erreicht"]),
        ("hypothese", "coverage_erreicht",        hypothese["coverage_erreicht"]),
        ("hypothese", "ergebnis",                 hypothese["ergebnis"]),
    ]

    # mapping_qualitaet
    if mapping_qualitaet:
        zeilen += [
            ("mapping_qualitaet", "n_uuid_total",         mapping_qualitaet["n_uuid_total"]),
            ("mapping_qualitaet", "n_uuid_gemappt",        mapping_qualitaet["n_uuid_gemappt"]),
            ("mapping_qualitaet", "n_uuid_nicht_gemappt",  mapping_qualitaet["n_uuid_nicht_gemappt"]),
            ("mapping_qualitaet", "mapping_rate",          mapping_qualitaet["mapping_rate"]),
            ("mapping_qualitaet", "n_child_zu_parent",     mapping_qualitaet["n_child_zu_parent"]),
        ]

    # kategorie_gt (alternative Ground Truth basierend auf Navigationskategorie)
    if kategorie_precision is not None:
        zeilen.append(("gesamt", "kategorie_precision_mean", kategorie_precision))

    df_out = pd.DataFrame(zeilen, columns=["kategorie", "metrik", "wert"])
    df_out.to_csv(pfad, index=False, encoding="utf-8")
    print(f"  Metriken exportiert: {pfad}")


def exportiere_produkt_metriken(
    df_eval: pd.DataFrame,
    pfad: str = str(PRODUKT_CSV),
) -> None:
    """Exportiert Produkt-Level-Metriken in output/produkt_metriken.csv."""
    df_out = df_eval[["produkt_id", "tp", "fp", "fn", "precision", "recall", "f1",
                       "gt_groesse", "n_empfohlen"]].copy()
    df_out = df_out.rename(columns={"n_empfohlen": "n_empfehlungen"})
    df_out.to_csv(pfad, index=False, encoding="utf-8")
    print(f"  Produkt-Metriken exportiert: {pfad}")


def exportiere_datenqualitaet(
    dq_info: dict,
    df_ausgeschlossen: pd.DataFrame,
    df_identische: pd.DataFrame,
    df_filter_vgl: pd.DataFrame,
    df_sensitivitaet: pd.DataFrame,
) -> None:
    """Exportiert alle Datenqualitaets-CSVs."""
    # datenqualitaet_uebersicht.csv
    zeilen: list[dict] = []
    zeilen.append({"kategorie": "gesamt", "metrik": "n_total", "wert": dq_info["n_total"]})
    zeilen.append({"kategorie": "gesamt", "metrik": "mean_text_laenge", "wert": round(dq_info["mean_laenge"], 1)})
    zeilen.append({"kategorie": "gesamt", "metrik": "median_text_laenge", "wert": round(dq_info["median_laenge"], 1)})
    for kat, anzahl in dq_info.get("verteilung", {}).items():
        zeilen.append({"kategorie": "verteilung", "metrik": kat, "wert": anzahl})
    for feld, rate in dq_info.get("befuellungsraten", {}).items():
        zeilen.append({"kategorie": "befuellungsrate", "metrik": feld, "wert": round(rate, 4)})
    pd.DataFrame(zeilen).to_csv(DATENQUALITAET_CSV, index=False, encoding="utf-8")
    print(f"  Datenqualitaet exportiert: {DATENQUALITAET_CSV}")

    # ausgeschlossene_artikel.csv
    df_ausgeschlossen.to_csv(AUSGESCHLOSSENE_CSV, index=False, encoding="utf-8")
    print(f"  Ausgeschlossene exportiert: {AUSGESCHLOSSENE_CSV}")

    # identische_texte_gruppen.csv
    df_identische.to_csv(IDENTISCHE_CSV, index=False, encoding="utf-8")
    print(f"  Identische Texte exportiert: {IDENTISCHE_CSV}")

    # filter_vergleich.csv
    df_filter_vgl.to_csv(FILTER_CSV, index=False, encoding="utf-8")
    print(f"  Filter-Vergleich exportiert: {FILTER_CSV}")

    # sensitivitaet_schwellenwerte.csv
    df_sensitivitaet.to_csv(SENSITIVITAET_CSV, index=False, encoding="utf-8")
    print(f"  Sensitivitaet exportiert: {SENSITIVITAET_CSV}")


# ============================================================
# 14. GRAFIK-HILFSFUNKTIONEN
# ============================================================

def _speichern(fig: plt.Figure, dateiname: str) -> Path:
    pfad = AUSGABE_DIR / dateiname
    fig.savefig(pfad, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pfad


def _hypothesen_linie(ax: plt.Axes, schwelle: float = 0.6, orient: str = "v") -> None:
    if orient == "v":
        ax.axvline(schwelle, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Hypothese ({schwelle})", zorder=5)
    else:
        ax.axhline(schwelle, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Hypothese ({schwelle})", zorder=5)


# ============================================================
# 15. GRAFIKEN 01–14 (Evaluations-Grafiken)
# ============================================================

def plot_01_precision_histogramm(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    werte = df_eval["precision"]
    ax.hist(werte, bins=30, color=FARBE_PRECISION, edgecolor="white", linewidth=0.4, alpha=0.85)
    mittel, median = werte.mean(), werte.median()
    ax.axvline(mittel, color="black",  linestyle="--", linewidth=1.5, label=f"Mittelwert: {mittel:.3f}")
    ax.axvline(median, color="orange", linestyle=":",  linewidth=1.5, label=f"Median: {median:.3f}")
    _hypothesen_linie(ax, 0.6, "v")
    ax.set_xlabel("Precision@6", fontsize=11)
    ax.set_ylabel("Anzahl Produkte", fontsize=10)
    ax.set_title("Verteilung Precision@6 — Gesamtevaluation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "01-Verteilung-Treffergenauigkeit.png")


def plot_02_recall_histogramm(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    werte = df_eval["recall"]
    mittel, median = werte.mean(), werte.median()
    ax.hist(werte, bins=30, color=FARBE_RECALL, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(mittel, color="black",  linestyle="--", linewidth=1.5, label=f"Mittelwert: {mittel:.3f}")
    ax.axvline(median, color="orange", linestyle=":",  linewidth=1.5, label=f"Median: {median:.3f}")
    ax.set_xlabel("Recall@6", fontsize=11)
    ax.set_ylabel("Anzahl Produkte", fontsize=10)
    ax.set_title("Verteilung Recall@6 — Gesamtevaluation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "02-Verteilung-Wiederfindungsrate.png")


def plot_03_f1_histogramm(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    werte = df_eval["f1"]
    mittel, median = werte.mean(), werte.median()
    ax.hist(werte, bins=30, color=FARBE_F1, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(mittel, color="black",  linestyle="--", linewidth=1.5, label=f"Mittelwert: {mittel:.3f}")
    ax.axvline(median, color="orange", linestyle=":",  linewidth=1.5, label=f"Median: {median:.3f}")
    ax.set_xlabel("F1@6", fontsize=11)
    ax.set_ylabel("Anzahl Produkte", fontsize=10)
    ax.set_title("Verteilung F1@6 — Gesamtevaluation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "03-Verteilung-Gesamtguete.png")


def plot_04_precision_recall_scatter(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_eval["recall"], df_eval["precision"],
        c=df_eval["f1"], cmap="viridis",
        alpha=0.55, s=18, linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("F1@6", fontsize=10)
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_xlabel("Recall@6", fontsize=11)
    ax.set_ylabel("Precision@6", fontsize=11)
    ax.set_title("Precision@6 vs. Recall@6 — Gesamtevaluation\n(Farbe = F1@6)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "04-Treffergenauigkeit-vs-Wiederfindung.png")


def plot_05_gt_groesse_vs_recall(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.15, 0.15, len(df_eval))
    ax.scatter(df_eval["gt_groesse"] + jitter, df_eval["recall"],
               alpha=0.35, s=12, color="steelblue", linewidths=0)
    df_sorted = df_eval.sort_values("gt_groesse")
    if len(df_sorted) > 20:
        fenster = max(5, len(df_sorted) // 30)
        rollmean = df_sorted.set_index("gt_groesse")["recall"].rolling(fenster, center=True).mean()
        ax.plot(rollmean.index, rollmean.values, color="tomato", linewidth=2, label="Gleitender Mittelwert")
        ax.legend(fontsize=9)
    ax.set_xlabel("Ground-Truth-Größe", fontsize=11)
    ax.set_ylabel("Recall@6", fontsize=11)
    ax.set_title("GT-Größe vs. Recall@6 — Gesamtevaluation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _speichern(fig, "05-Kaufhistorie-vs-Wiederfindung.png")


def plot_06_gt_groesse_vs_precision(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.15, 0.15, len(df_eval))
    ax.scatter(df_eval["gt_groesse"] + jitter, df_eval["precision"],
               alpha=0.35, s=12, color=FARBE_PRECISION, linewidths=0)
    df_sorted = df_eval.sort_values("gt_groesse")
    if len(df_sorted) > 20:
        fenster = max(5, len(df_sorted) // 30)
        rollmean = df_sorted.set_index("gt_groesse")["precision"].rolling(fenster, center=True).mean()
        ax.plot(rollmean.index, rollmean.values, color="tomato", linewidth=2, label="Gleitender Mittelwert")
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_xlabel("Ground-Truth-Größe", fontsize=11)
    ax.set_ylabel("Precision@6", fontsize=11)
    ax.set_title("GT-Größe vs. Precision@6 — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "06-Kaufhistorie-vs-Treffergenauigkeit.png")


def plot_07_boxplot_metriken(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    df_long = df_eval[["precision", "recall", "f1"]].melt(var_name="Metrik", value_name="Wert")
    df_long["Metrik"] = df_long["Metrik"].map(
        {"precision": "Precision@6", "recall": "Recall@6", "f1": "F1@6"}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_long, x="Metrik", y="Wert",
                palette=[FARBE_PRECISION, FARBE_RECALL, FARBE_F1], ax=ax)
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_ylabel("Wert", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Boxplot der Metriken — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "07-Kennzahlen-Vergleich.png")



def plot_08_top_bottom_f1(df_eval: pd.DataFrame, n: int = 10) -> Path:
    sns.set_style(SEABORN_STYLE)
    top    = df_eval.nlargest(n, "f1").copy()
    bottom = df_eval.nsmallest(n, "f1").copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, farbe, titel in [
        (ax1, top,    PALETTE_POS, f"Top-{n} nach F1@6"),
        (ax2, bottom, PALETTE_NEG, f"Bottom-{n} nach F1@6"),
    ]:
        labels = data["produkt_id"].str[:20]
        bars = ax.barh(labels, data["f1"], color=farbe, edgecolor="white")
        ax.set_xlabel("F1@6", fontsize=10)
        ax.set_title(titel, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, data["f1"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    fig.suptitle("Top/Bottom F1@6 — Gesamtevaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _speichern(fig, "08-Beste-und-schlechteste-Gesamtguete.png")


def plot_9_gt_verteilung(ground_truth: dict[str, set[str]]) -> Path:
    sns.set_style(SEABORN_STYLE)
    groessen = [len(v) for v in ground_truth.values() if v]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(groessen, bins=40, color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.85)
    mittel, median = np.mean(groessen), np.median(groessen)
    ax.axvline(mittel, color="tomato", linestyle="--", linewidth=1.5, label=f"Mittelwert: {mittel:.1f}")
    ax.axvline(median, color="orange", linestyle=":",  linewidth=1.5, label=f"Median: {median:.1f}")
    ax.set_xlabel("Anzahl Co-Purchase-Partner", fontsize=11)
    ax.set_ylabel("Anzahl Produkte", fontsize=11)
    ax.set_title("GT-Größen-Verteilung — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    return _speichern(fig, "9-Verteilung-Kaufhistorie-Groesse.png")


def plot_10_coverage_balken(coverage: dict) -> Path:
    sns.set_style(SEABORN_STYLE)
    labels = ["Genau k Empf.", "Min. 1 Treffer", "Min. 2 Treffer", "P@6 >= 0.5", "P@6 >= 0.6"]
    werte  = [
        coverage.get("anteil_genau_k_empf", 0),
        coverage.get("anteil_min_1_treffer", 0),
        coverage.get("anteil_min_2_treffer", 0),
        coverage.get("anteil_p_gte_05", 0),
        coverage.get("anteil_p_gte_06", 0),
    ]
    farben = [FARBE_PRECISION, PALETTE_POS, PALETTE_POS, FARBE_F1, PALETTE_NEG]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, werte, color=farben, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, werte):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(HYPOTHESE_MINDEST_COVERAGE, color="#d62728", linestyle="--",
               linewidth=1.5, label=f"Mindest-Coverage ({HYPOTHESE_MINDEST_COVERAGE:.0%})")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Anteil Produkte", fontsize=11)
    ax.set_title("Coverage-Kennzahlen — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "10-Abdeckung-Empfehlungen.png")


def plot_11_precision_kumulativ(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    werte = np.sort(df_eval["precision"].values)
    ecdf  = np.arange(1, len(werte) + 1) / len(werte)
    ax.plot(werte, ecdf, color=FARBE_PRECISION, linewidth=2)
    ax.axvline(0.6, color="#d62728", linestyle="--", linewidth=1.5, label="Hypothese (0.6)")
    anteil_gte06 = (df_eval["precision"] >= 0.6).mean()
    ax.axhline(anteil_gte06, color="gray", linestyle=":", linewidth=1,
               label=f"Anteil >= 0.6: {anteil_gte06:.1%}")
    ax.set_xlabel("Precision@6", fontsize=11)
    ax.set_ylabel("Kumul. Anteil Produkte", fontsize=11)
    ax.set_title("Kumulative Verteilung Precision@6 — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "11-Treffergenauigkeit-kumulativ.png")


def plot_12_hits_verteilung(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    anzahl = df_eval["hits"].value_counts().sort_index()
    alle_hits = pd.Series(0, index=range(TOP_N + 1))
    alle_hits.update(anzahl)
    farben = [PALETTE_NEG if i == 0 else (PALETTE_POS if i >= 4 else FARBE_PRECISION)
              for i in range(TOP_N + 1)]
    bars = ax.bar(alle_hits.index, alle_hits.values, color=farben, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, alle_hits.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(alle_hits.values) * 0.01,
                f"{val:,}", ha="center", fontsize=9)
    ax.set_xlabel("Anzahl Treffer", fontsize=11)
    ax.set_ylabel("Anzahl Produkte", fontsize=11)
    ax.set_xticks(range(TOP_N + 1))
    ax.set_title(f"Treffer-Verteilung (0–{TOP_N}) — Gesamtevaluation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _speichern(fig, "12-Verteilung-Treffer-pro-Produkt.png")


def plot_13_violin_metriken(df_eval: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    df_long = df_eval[["precision", "recall", "f1"]].melt(var_name="Metrik", value_name="Wert")
    df_long["Metrik"] = df_long["Metrik"].map(
        {"precision": "Precision@6", "recall": "Recall@6", "f1": "F1@6"}
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df_long, x="Metrik", y="Wert",
                   palette=[FARBE_PRECISION, FARBE_RECALL, FARBE_F1], ax=ax, inner="box")
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_ylabel("Wert", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Violin-Plot der Metriken — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "13-Kennzahlen-Detailverteilung.png")


# ============================================================
# 16. GRAFIKEN 14
# ============================================================

def plot_14_train_test_vergleich(train_test: dict) -> Path:
    sns.set_style(SEABORN_STYLE)
    df_tr = train_test.get("df_train", pd.DataFrame())
    df_te = train_test.get("df_test",  pd.DataFrame())

    metriken   = ["Precision@6", "Recall@6", "F1@6"]
    train_vals = [
        float(df_tr["precision"].mean()) if len(df_tr) > 0 else 0.0,
        float(df_tr["recall"].mean())    if len(df_tr) > 0 else 0.0,
        float(df_tr["f1"].mean())        if len(df_tr) > 0 else 0.0,
    ]
    test_vals = [
        float(df_te["precision"].mean()) if len(df_te) > 0 else 0.0,
        float(df_te["recall"].mean())    if len(df_te) > 0 else 0.0,
        float(df_te["f1"].mean())        if len(df_te) > 0 else 0.0,
    ]

    x = np.arange(len(metriken))
    breite = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_t = ax.bar(x - breite / 2, train_vals, breite, label=f"Train ({int(TRAIN_ANTEIL*100)}%)",
                    color=FARBE_PRECISION, alpha=0.85, edgecolor="white")
    bars_v = ax.bar(x + breite / 2, test_vals,  breite, label=f"Test ({int((1-TRAIN_ANTEIL)*100)}%)",
                    color=FARBE_RECALL,    alpha=0.85, edgecolor="white")
    for bars in [bars_t, bars_v]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_xticks(x)
    ax.set_xticklabels(metriken, fontsize=10)
    ax.set_ylabel("Mittelwert", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Train/Test-Vergleich ({int(TRAIN_ANTEIL*100)}/{int((1-TRAIN_ANTEIL)*100)} sequentiell)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "14-Trainings-vs-Testdaten-Vergleich.png")



# ============================================================
# 17. GRAFIKEN 15–17 (Segmentierte Evaluation)
# ============================================================

def plot_15_precision_boxplot_gt_segmente(df_eval: pd.DataFrame, segmente: pd.Series) -> Path:
    sns.set_style(SEABORN_STYLE)
    df_mit_seg = df_eval.copy()
    df_mit_seg["Segment"] = segmente.values
    reihenfolge = ["Klein", "Mittel", "Gross"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_mit_seg, x="Segment", y="precision", order=reihenfolge,
                palette=[PALETTE_POS, FARBE_PRECISION, PALETTE_NEG], ax=ax)
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_ylabel("Precision@6", fontsize=11)
    ax.set_xlabel("GT-Größen-Segment", fontsize=11)
    ax.set_title("Precision@6 nach GT-Segment — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "15-Treffergenauigkeit-nach-Kaufhistorie.png")


def plot_16_recall_vs_precision_by_gt(df_eval: pd.DataFrame, segmente: pd.Series) -> Path:
    sns.set_style(SEABORN_STYLE)
    df_mit_seg = df_eval.copy()
    df_mit_seg["Segment"] = segmente.values
    farben_map = {"Klein": PALETTE_POS, "Mittel": FARBE_PRECISION, "Gross": PALETTE_NEG}
    fig, ax = plt.subplots(figsize=(8, 6))
    for seg, farbe in farben_map.items():
        teilmenge = df_mit_seg[df_mit_seg["Segment"] == seg]
        ax.scatter(teilmenge["recall"], teilmenge["precision"],
                   c=farbe, alpha=0.5, s=18, label=seg, linewidths=0)
    _hypothesen_linie(ax, 0.6, "h")
    ax.set_xlabel("Recall@6", fontsize=11)
    ax.set_ylabel("Precision@6", fontsize=11)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Precision vs. Recall nach GT-Segment — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "16-Wiederfindung-vs-Genauigkeit-nach-Segment.png")


def plot_17_heatmap_metriken(seg_stats: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    heat_data = seg_stats.set_index("segment")[
        ["precision_mean", "recall_mean", "f1_mean"]
    ].rename(columns={
        "precision_mean": "Precision@6",
        "recall_mean":    "Recall@6",
        "f1_mean":        "F1@6",
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5, cbar_kws={"label": "Mittelwert"})
    ax.set_title("Heatmap Metriken nach GT-Segment — Gesamtevaluation", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Segment", fontsize=10)
    plt.tight_layout()
    return _speichern(fig, "17-Kennzahlen-Heatmap.png")


# ============================================================
# 18. GRAFIKEN 18–21 (Datenqualität)
# ============================================================

def plot_18_dq_textlaenge_histogramm(df_produkte: pd.DataFrame, text_spalten: list[str]) -> Path:
    sns.set_style(SEABORN_STYLE)
    if "text_laenge" not in df_produkte.columns:
        laengen = df_produkte[text_spalten].apply(
            lambda z: len(" ".join(str(v) for v in z if str(v).strip())), axis=1
        )
    else:
        laengen = df_produkte["text_laenge"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(laengen, bins=60, color="steelblue", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(MIN_TEXT_LAENGE, color="#d62728", linestyle="--", linewidth=1.8,
               label=f"Mindest-Textlänge ({MIN_TEXT_LAENGE})")
    ax.set_xlabel("Kombinierte Textlänge (Zeichen)", fontsize=11)
    ax.set_ylabel("Anzahl Artikel", fontsize=11)
    ax.set_title("Textlängen-Verteilung — Produktkatalog", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _speichern(fig, "18-DQ-Verteilung-Textlaenge.png")


def plot_19_dq_befuellung_pro_feld(dq_info: dict) -> Path:
    sns.set_style(SEABORN_STYLE)
    befuellungsraten = dq_info.get("befuellungsraten", {})
    if not befuellungsraten:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return _speichern(fig, "21-DQ-Befuellung-pro-Feld.png")

    felder = list(befuellungsraten.keys())
    raten  = [befuellungsraten[f] for f in felder]
    farben = [PALETTE_POS if r >= 0.5 else PALETTE_NEG for r in raten]

    fig, ax = plt.subplots(figsize=(9, max(4, len(felder) * 0.7)))
    bars = ax.barh(felder, raten, color=farben, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, raten):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Befüllungsrate", fontsize=11)
    ax.set_title("Befüllungsrate pro Textfeld — Produktkatalog", fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    return _speichern(fig, "19-DQ-Befuellung-pro-Feld.png")


def plot_20_dq_felder_pro_artikel(df_produkte: pd.DataFrame, text_spalten: list[str]) -> Path:
    sns.set_style(SEABORN_STYLE)
    n_befuellt = df_produkte[text_spalten].apply(
        lambda zeile: sum(1 for v in zeile if str(v).strip()), axis=1
    )
    anzahl = n_befuellt.value_counts().sort_index()
    alle   = pd.Series(0, index=range(len(text_spalten) + 1))
    alle.update(anzahl)

    fig, ax = plt.subplots(figsize=(9, 5))
    farben = [PALETTE_NEG if i == 0 else (PALETTE_POS if i >= 3 else FARBE_PRECISION)
              for i in range(len(text_spalten) + 1)]
    bars = ax.bar(alle.index, alle.values, color=farben, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, alle.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(alle.values) * 0.01,
                    f"{int(val):,}", ha="center", fontsize=9)
    ax.set_xlabel("Anzahl befüllter Felder", fontsize=11)
    ax.set_ylabel("Anzahl Artikel", fontsize=11)
    ax.set_xticks(range(len(text_spalten) + 1))
    ax.set_title("Befüllte Felder pro Artikel — Produktkatalog", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _speichern(fig, "20-DQ-Befuellte-Felder-pro-Produkt.png")




def plot_21_dq_sensitivitaet(df_sensitivitaet: pd.DataFrame) -> Path:
    sns.set_style(SEABORN_STYLE)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(df_sensitivitaet["schwellenwert"], df_sensitivitaet["precision_mean"],
             color=FARBE_PRECISION, marker="o", linewidth=2, label="Precision@6 Mean")
    ax1.axhline(HYPOTHESE_PRECISION_MEAN, color="#d62728", linestyle="--", linewidth=1.5,
                label=f"Hypothese ({HYPOTHESE_PRECISION_MEAN})")
    ax1.axvline(MIN_TEXT_LAENGE, color="gray", linestyle=":", linewidth=1.5,
                label=f"Gewählter Schwellenwert ({MIN_TEXT_LAENGE})")
    ax2.plot(df_sensitivitaet["schwellenwert"], df_sensitivitaet["n_artikel"],
             color="steelblue", marker="s", linestyle="--", linewidth=1.5, alpha=0.7,
             label="Anzahl Artikel")

    ax1.set_xlabel("Mindest-Textlänge (Zeichen)", fontsize=11)
    ax1.set_ylabel("Precision@6 Mittelwert", fontsize=11, color=FARBE_PRECISION)
    ax2.set_ylabel("Anzahl evaluierbare Artikel", fontsize=11, color="steelblue")
    ax1.set_ylim(0, max(df_sensitivitaet["precision_mean"].max() * 1.3, 0.1))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.set_title("Sensitivitätsanalyse: Schwellenwert vs. Precision@6", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _speichern(fig, "21-DQ-Schwellenwert-Sensitivitaet.png")



def erstelle_alle_grafiken(
    df_eval: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    empfehlungen: dict[str, list[str]],
    coverage: dict,
    seg_stats: pd.DataFrame,
    segmente: pd.Series,
    train_test: dict,
    df_produkte: pd.DataFrame,
    dq_info: dict,
    df_ausgeschlossen: pd.DataFrame,
    df_filter_vgl: pd.DataFrame,
    df_sensitivitaet: pd.DataFrame,
) -> list[Path]:
    pfade = []
    pfade.append(plot_01_precision_histogramm(df_eval))
    pfade.append(plot_02_recall_histogramm(df_eval))
    pfade.append(plot_03_f1_histogramm(df_eval))
    pfade.append(plot_04_precision_recall_scatter(df_eval))
    pfade.append(plot_05_gt_groesse_vs_recall(df_eval))
    pfade.append(plot_06_gt_groesse_vs_precision(df_eval))
    pfade.append(plot_07_boxplot_metriken(df_eval))

    pfade.append(plot_08_top_bottom_f1(df_eval))
    pfade.append(plot_9_gt_verteilung(ground_truth))
    pfade.append(plot_10_coverage_balken(coverage))
    pfade.append(plot_11_precision_kumulativ(df_eval))
    pfade.append(plot_12_hits_verteilung(df_eval))
    pfade.append(plot_13_violin_metriken(df_eval))
    pfade.append(plot_14_train_test_vergleich(train_test))

    pfade.append(plot_15_precision_boxplot_gt_segmente(df_eval, segmente))
    pfade.append(plot_16_recall_vs_precision_by_gt(df_eval, segmente))
    pfade.append(plot_17_heatmap_metriken(seg_stats))
    pfade.append(plot_18_dq_textlaenge_histogramm(df_produkte, TEXT_SPALTEN))
    pfade.append(plot_19_dq_befuellung_pro_feld(dq_info))
    pfade.append(plot_20_dq_felder_pro_artikel(df_produkte, TEXT_SPALTEN))

    pfade.append(plot_21_dq_sensitivitaet(df_sensitivitaet))


    return pfade


# ============================================================
# 19. PDF-BERICHT (4 Seiten, kein Titelblatt) (KI)
# ============================================================

def erstelle_pdf(
    df_eval: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    empfehlungen: dict[str, list[str]],
    n_bestellungen: int,
    n_bestellungen_ge2: int,
    coverage: dict,
    train_test: dict,
    hypothese: dict,
    dq_info: dict,
    df_ausgeschlossen: pd.DataFrame,
    df_filter_vgl: pd.DataFrame,
    df_sensitivitaet: pd.DataFrame,
    mapping_qualitaet: dict | None = None,
) -> None:
    """Erstellt PDF-Bericht — 4 Seiten, rein numerisch, kein Titelblatt."""
    from matplotlib.backends.backend_pdf import PdfPages

    stats       = berechne_statistiken(df_eval)
    gt_groessen = [len(v) for v in ground_truth.values() if v]

    TIT  = "#1a3a5c"
    TXT  = "#333333"
    VAL  = "#1a3a5c"
    MONO = "monospace"

    def _seite(pdf, inhalte: list) -> None:
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        for eintrag in inhalte:
            txt, y, fs, fw, color = eintrag[:5]
            ff = eintrag[5] if len(eintrag) > 5 else "sans-serif"
            ax.text(0.05, y, txt, transform=ax.transAxes, fontsize=fs,
                    fontweight=fw, color=color, va="top", fontfamily=ff)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _z(label: str, val: str, b1: int = 44, b2: int = 12) -> str:
        return "  " + str(label).ljust(b1) + str(val).rjust(b2)

    s    = stats
    n_ev = len(df_eval)
    df_tr = train_test.get("df_train", pd.DataFrame())
    df_te = train_test.get("df_test",  pd.DataFrame())

    with PdfPages(PDF_DATEI) as pdf:

        # ---- Seite 1: Datenübersicht + Hauptmetriken ----
        seite1 = [
            ("=== DATENÜBERSICHT + HAUPTMETRIKEN (Gesamtevaluation) ===", 0.96, 11, "bold", TIT, MONO),
            ("", 0.92, 9, "normal", TXT),
            (_z("Bestellungen gesamt:",              f"{n_bestellungen:,}"),                                    0.88, 10, "normal", VAL, MONO),
            (_z("Bestellungen mit >=2 Produkten:",   f"{n_bestellungen_ge2:,}"),                                0.84, 10, "normal", VAL, MONO),
            (_z("Produkte mit TF-IDF-Empfehlungen:", f"{len(empfehlungen):,}"),                                 0.80, 10, "normal", VAL, MONO),
            (_z("Evaluierbare Produkte:",             f"{n_ev:,}"),                                             0.76, 10, "normal", VAL, MONO),
            (_z("Mittlere GT-Größe:",                 f"{np.mean(gt_groessen):.1f}" if gt_groessen else '—'),   0.72, 10, "normal", VAL, MONO),
            (_z("Median GT-Größe:",                   f"{np.median(gt_groessen):.1f}" if gt_groessen else '—'), 0.68, 10, "normal", VAL, MONO),
            ("", 0.64, 9, "normal", TXT),
            ("  Precision@6:                              Mean       Median      Std        Min        Max", 0.60, 9, "bold", TIT, MONO),
            (_z("",
                f"{s['precision']['mean']:.3f}     {s['precision']['median']:.3f}     {s['precision']['std']:.3f}     {s['precision']['min']:.3f}     {s['precision']['max']:.3f}"),
             0.56, 9, "normal", VAL, MONO),
            (_z("  Anteil P@6 = 0:",   f"{s['precision']['anteil_null']:.1%}", 44, 10),                        0.52, 9, "normal", VAL, MONO),
            (_z("  Anteil P@6 >= 0.6:",f"{s['precision']['anteil_gte06']:.1%}", 44, 10),                       0.48, 9, "normal", VAL, MONO),
            ("", 0.44, 9, "normal", TXT),
            ("  Recall@6:", 0.41, 9, "bold", TIT, MONO),
            (_z("",
                f"{s['recall']['mean']:.3f}     {s['recall']['median']:.3f}     {s['recall']['std']:.3f}     {s['recall']['min']:.3f}     {s['recall']['max']:.3f}"),
             0.37, 9, "normal", VAL, MONO),
            ("", 0.33, 9, "normal", TXT),
            ("  F1@6:", 0.30, 9, "bold", TIT, MONO),
            (_z("",
                f"{s['f1']['mean']:.3f}     {s['f1']['median']:.3f}     {s['f1']['std']:.3f}     {s['f1']['min']:.3f}     {s['f1']['max']:.3f}"),
             0.26, 9, "normal", VAL, MONO),
            ("", 0.22, 9, "normal", TXT),
            (_z("  Ø TP pro Produkt:", f"{df_eval['tp'].mean():.2f} von {TOP_N}", 44, 12),                    0.18, 9, "normal", VAL, MONO),
            (_z("  Ø FP pro Produkt:", f"{df_eval['fp'].mean():.2f}", 44, 12),                                0.14, 9, "normal", VAL, MONO),
            (_z("  Ø FN pro Produkt:", f"{df_eval['fn'].mean():.2f}", 44, 12),                                0.10, 9, "normal", VAL, MONO),
        ]
        _seite(pdf, seite1)

        # ---- Seite 2: Datenqualität ----
        n_total  = dq_info.get("n_total", 0)
        n_ausge  = len(df_ausgeschlossen)
        n_gefi   = n_total - n_ausge
        grund_cnt = df_ausgeschlossen["grund"].value_counts() if not df_ausgeschlossen.empty else pd.Series(dtype=int)

        seite2: list = [
            ("=== DATENQUALITÄT ===", 0.96, 11, "bold", TIT, MONO),
            ("", 0.92, 9, "normal", TXT),
            (_z("Artikel total:",                     f"{n_total:,}"),                                          0.88, 10, "normal", VAL, MONO),
            (_z("Artikel mit Text >= 50 Zeichen:",    f"{n_gefi:,}  ({n_gefi/n_total:.1%})" if n_total else "—"), 0.84, 10, "normal", VAL, MONO),
            (_z("Ausgeschlossen (< 50 Zeichen):",     f"{n_ausge:,} ({n_ausge/n_total:.1%})" if n_total else "—"), 0.80, 10, "normal", VAL, MONO),
            (_z("  davon komplett leer:",             f"{grund_cnt.get('komplett leer', 0):,}"),                 0.76, 9, "normal", VAL, MONO),
            (_z("  davon nur Name:",                  f"{grund_cnt.get('nur Name', 0):,}"),                      0.72, 9, "normal", VAL, MONO),
            (_z("  davon Text zu kurz:",              f"{grund_cnt.get('Text zu kurz', 0):,}"),                  0.68, 9, "normal", VAL, MONO),
            (_z("Mittlere Textlänge (alle):",         f"{dq_info.get('mean_laenge', 0):.0f} Zeichen"),           0.64, 9, "normal", VAL, MONO),
            ("", 0.60, 9, "normal", TXT),
            ("  Sensitivitätsanalyse (Schwellenwert -> Precision@6 / N Artikel):", 0.57, 9, "bold", TIT, MONO),
        ]
        y = 0.53
        for _, s_z in df_sensitivitaet.iterrows():
            sw = int(s_z["schwellenwert"])
            markierung = " <--" if sw == MIN_TEXT_LAENGE else ""
            zeile_txt = f"  Schwellenwert {sw:>3}:   P@6={s_z['precision_mean']:.3f}   N={int(s_z['n_artikel']):,}{markierung}"
            seite2.append((zeile_txt, y, 9, "normal", VAL, MONO))
            y -= 0.04

        y -= 0.01
        seite2.append(("", y, 9, "normal", TXT))
        y -= 0.04
        seite2.append(("  Vorher/Nachher-Vergleich (Filter Schwellenwert 50):", y, 9, "bold", TIT, MONO))
        y -= 0.04
        for _, vgl_z in df_filter_vgl.iterrows():
            seite2.append((f"  {vgl_z['metrik']:<20}  ohne={vgl_z['ohne_filter']:.3f}  mit={vgl_z['mit_filter']:.3f}  diff={vgl_z['differenz']:+.3f}",
                           y, 9, "normal", VAL, MONO))
            y -= 0.04
        _seite(pdf, seite2)

        # ---- Seite 3: TP/FP/FN + Coverage + Train/Test ----
        seite3: list = [
            ("=== TP/FP/FN + COVERAGE + TRAIN/TEST ===", 0.96, 11, "bold", TIT, MONO),
            ("", 0.92, 9, "normal", TXT),
            ("TP/FP/FN (Durchschnitt pro Produkt):", 0.89, 10, "bold", TIT),
            (_z("  Ø TP (korrekte Empfehlungen):", f"{df_eval['tp'].mean():.2f}"),  0.85, 9, "normal", VAL, MONO),
            (_z("  Ø FP (falsche Empfehlungen):", f"{df_eval['fp'].mean():.2f}"),   0.81, 9, "normal", VAL, MONO),
            (_z("  Ø FN (verpasste Käufe):",      f"{df_eval['fn'].mean():.2f}"),   0.77, 9, "normal", VAL, MONO),
            ("", 0.73, 9, "normal", TXT),
            ("Coverage:", 0.70, 10, "bold", TIT),
            (_z("  Produkte mit 6 Empfehlungen:",   f"{coverage.get('anteil_genau_k_empf', 0):.1%}"),  0.66, 9, "normal", VAL, MONO),
            (_z("  Produkte mind. 1 Treffer:",      f"{coverage.get('anteil_min_1_treffer', 0):.1%}"), 0.62, 9, "normal", VAL, MONO),
            (_z("  Produkte mind. 2 Treffer:",      f"{coverage.get('anteil_min_2_treffer', 0):.1%}"), 0.58, 9, "normal", VAL, MONO),
            (_z("  Produkte P@6 >= 0.5:",           f"{coverage.get('anteil_p_gte_05', 0):.1%}"),     0.54, 9, "normal", VAL, MONO),
            (_z("  Produkte P@6 >= 0.6:",           f"{coverage.get('anteil_p_gte_06', 0):.1%}"),     0.50, 9, "bold",   VAL, MONO),
            ("", 0.46, 9, "normal", TXT),
            (f"Train/Test-Split ({int(TRAIN_ANTEIL*100)}/{int((1-TRAIN_ANTEIL)*100)} sequentiell):", 0.43, 10, "bold", TIT),
            (_z("  Bestellungen Train:",            f"{train_test.get('n_train', 0):,}"),             0.39, 9, "normal", VAL, MONO),
            (_z("  Bestellungen Test:",             f"{train_test.get('n_test', 0):,}"),              0.35, 9, "normal", VAL, MONO),
            (_z("  Precision@6 Train:",             f"{df_tr['precision'].mean():.3f}" if len(df_tr) > 0 else "—"), 0.31, 9, "normal", VAL, MONO),
            (_z("  Precision@6 Test:",              f"{df_te['precision'].mean():.3f}" if len(df_te) > 0 else "—"), 0.27, 9, "normal", VAL, MONO),
            (_z("  Recall@6 Train:",                f"{df_tr['recall'].mean():.3f}" if len(df_tr) > 0 else "—"),    0.23, 9, "normal", VAL, MONO),
            (_z("  Recall@6 Test:",                 f"{df_te['recall'].mean():.3f}" if len(df_te) > 0 else "—"),    0.19, 9, "normal", VAL, MONO),
            (_z("  F1@6 Train:",                    f"{df_tr['f1'].mean():.3f}" if len(df_tr) > 0 else "—"),        0.15, 9, "normal", VAL, MONO),
            (_z("  F1@6 Test:",                     f"{df_te['f1'].mean():.3f}" if len(df_te) > 0 else "—"),        0.11, 9, "normal", VAL, MONO),
        ]
        if mapping_qualitaet:
            mq = mapping_qualitaet
            seite3 += [
                ("", 0.07, 9, "normal", TXT),
                ("Mapping-Qualität (UUID -> SKU -> Parent):", 0.04, 10, "bold", TIT),
                (_z("  UUIDs gesamt:",       f"{mq['n_uuid_total']:,}"),                                     0.00, 9, "normal", VAL, MONO),
                (_z("  gemappt:",            f"{mq['n_uuid_gemappt']:,}  ({mq['mapping_rate']:.1%})"),       -0.04, 9, "normal", VAL, MONO),
                (_z("  nicht gemappt:",      f"{mq['n_uuid_nicht_gemappt']:,}"),                             -0.08, 9, "normal", VAL, MONO),
                (_z("  Child->Parent:",      f"{mq['n_child_zu_parent']:,}"),                                -0.12, 9, "normal", VAL, MONO),
            ]
        _seite(pdf, seite3)

        # ---- Seite 4: Hypothesenprüfung ----
        ok_mean  = hypothese["precision_mean_erreicht"]
        ok_cov   = hypothese["coverage_erreicht"]
        ergebnis = hypothese["ergebnis"]
        p_mean   = hypothese["precision_mean"]
        cov_val  = hypothese["coverage"]

        farbe_mean = PALETTE_POS if ok_mean else PALETTE_NEG
        farbe_cov  = PALETTE_POS if ok_cov  else PALETTE_NEG
        farbe_erg  = PALETTE_POS if ergebnis == "bestaetigt" else PALETTE_NEG

        seite4 = [
            ("=== HYPOTHESENPRÜFUNG ===", 0.96, 11, "bold", TIT, MONO),
            ("", 0.92, 9, "normal", TXT),
            ("Hypothese H1: Durchschnittliche Precision@6 >= 0.6", 0.88, 11, "bold", TIT),
            ("", 0.84, 9, "normal", TXT),
            ("Kriterium 1: Precision@6 Durchschnitt >= 0.6", 0.80, 10, "bold", TIT),
            (f"  Ergebnis:   {p_mean:.3f} -> {'ERFÜLLT' if ok_mean else 'NICHT ERFÜLLT'}", 0.76, 10, "bold", farbe_mean, MONO),
            ("", 0.72, 9, "normal", TXT),
            ("Kriterium 2: Coverage (P@6 > 0) >= 50%", 0.68, 10, "bold", TIT),
            (f"  Ergebnis:   {cov_val:.1%} -> {'ERFÜLLT' if ok_cov else 'NICHT ERFÜLLT'}", 0.64, 10, "bold", farbe_cov, MONO),
            ("", 0.60, 9, "normal", TXT),
            (f"Gesamtergebnis: Hypothese {ergebnis.upper()}", 0.55, 13, "bold", farbe_erg),
        ]
        _seite(pdf, seite4)

    print(f"  PDF gespeichert: {PDF_DATEI}")


# ============================================================
# 20. MAIN
# ============================================================

def main() -> None:
    AUSGABE_DIR.mkdir(exist_ok=True)

    print("[1/9] Empfehlungen laden ...")
    empfehlungen = lade_empfehlungen()
    print(f"  Produkte mit Empfehlungen: {len(empfehlungen):,}")

    print("\n[2/9] Produktdaten laden (Datenqualitaets-Analyse) ...")
    df_produkte_alle = lade_produktdaten()
    gefilterte_ids    = set(empfehlungen.keys())
    df_gefiltert      = df_produkte_alle[df_produkte_alle["artikel_id"].isin(gefilterte_ids)].reset_index(drop=True)
    df_ausgeschlossen = df_produkte_alle[~df_produkte_alle["artikel_id"].isin(gefilterte_ids)].reset_index(drop=True)
    print(f"  Produkte gesamt:           {len(df_produkte_alle):,}")
    print(f"  → im Modell (empfehlungen.csv): {len(df_gefiltert):,}")
    print(f"  → ausgeschlossen:          {len(df_ausgeschlossen):,}")

    print("\n[3/9] Bestelldaten laden ...")
    df_bestellungen = lade_bestellungen()
    print(f"  Bestellungen gesamt: {len(df_bestellungen):,}")

    print("\n[4/9] ID-Mappings anwenden ...")
    uuid_zu_sku     = lade_uuid_zu_sku()
    child_zu_parent = lade_child_zu_parent()
    mapping_qualitaet = berechne_mapping_qualitaet(df_bestellungen, uuid_zu_sku, child_zu_parent)
    df_bestellungen = uebersetze_produkt_ids(df_bestellungen, uuid_zu_sku, child_zu_parent)
    n_bestellungen     = len(df_bestellungen)
    n_bestellungen_ge2 = int((df_bestellungen["produkte"].apply(len) >= 2).sum())
    print(f"  UUID-SKU-Mapping:     {len(uuid_zu_sku):,} Eintraege")
    print(f"  Child-Parent-Mapping: {len(child_zu_parent):,} Eintraege")
    print(f"  Mapping-Rate:         {mapping_qualitaet['mapping_rate']:.1%}  "
          f"({mapping_qualitaet['n_uuid_gemappt']:,}/{mapping_qualitaet['n_uuid_total']:,} UUIDs gemappt)")
    print(f"  Bestellungen mit >=2 Produkten: {n_bestellungen_ge2:,}")

    print("\n[5/9] Ground Truth aufbauen & Evaluation berechnen (TP/FP/FN) ...")
    ground_truth = build_ground_truth(df_bestellungen)
    gt_gefiltert = {k: v for k, v in ground_truth.items() if k in gefilterte_ids}
    df_eval      = evaluate_at_k(gt_gefiltert, empfehlungen)
    coverage     = berechne_coverage(df_eval, empfehlungen)
    seg_stats, segmente = segmentierte_evaluation(df_eval)
    hypothese    = pruefe_hypothese(df_eval, coverage)
    print(f"  GT-Produkte (gefiltert): {len(gt_gefiltert):,}")
    print(f"  Evaluierbare Produkte:   {len(df_eval):,}")
    print(f"  Precision@6:             {df_eval['precision'].mean():.4f}")
    print(f"  Recall@6:                {df_eval['recall'].mean():.4f}")
    print(f"  F1@6:                    {df_eval['f1'].mean():.4f}")

    gt_kategorie    = build_kategorie_ground_truth(df_gefiltert)
    gt_kat_gefiltert = {k: v for k, v in gt_kategorie.items() if k in gefilterte_ids}
    df_eval_kat     = evaluate_at_k(gt_kat_gefiltert, empfehlungen)
    p_kat = df_eval_kat["precision"].mean() if len(df_eval_kat) > 0 else 0.0
    print(f"  Kategorie-Precision@6:   {p_kat:.4f}  (gleiche Navigationskategorie als GT)")

    print("\n[6/9] Datenqualitaets-Analyse (Vorher/Nachher, Sensitivitaet) ...")
    dq_info        = analysiere_textqualitaet(df_produkte_alle, TEXT_SPALTEN)
    df_identische  = analysiere_identische_texte(df_gefiltert, TEXT_SPALTEN)
    df_ausgesch_kat = analysiere_ausgeschlossene(df_ausgeschlossen, TEXT_SPALTEN)
    df_filter_vgl  = vergleiche_mit_ohne_filter(ground_truth, empfehlungen, df_produkte_alle, df_gefiltert)
    df_sensitivitaet = sensitivitaetsanalyse(ground_truth, empfehlungen, df_produkte_alle, TEXT_SPALTEN)
    print(f"  Identische Text-Gruppen: {len(df_identische):,}")
    print(f"  Ausgeschlossene:         {len(df_ausgeschlossen):,}")

    print(f"\n[7/9] Train/Test-Split ({int(TRAIN_ANTEIL*100)}/{int((1-TRAIN_ANTEIL)*100)} sequentiell) ...")
    train_test = train_test_evaluation(df_bestellungen, empfehlungen)
    df_tr = train_test["df_train"]
    df_te = train_test["df_test"]
    print(f"  Train: {train_test['n_train']:,} Bestellungen, P@6={df_tr['precision'].mean():.4f}" if len(df_tr) > 0 else f"  Train: {train_test['n_train']:,} Bestellungen")
    print(f"  Test:  {train_test['n_test']:,} Bestellungen,  P@6={df_te['precision'].mean():.4f}" if len(df_te) > 0 else f"  Test: {train_test['n_test']:,} Bestellungen")

    print("\n[8/9] Grafiken erstellen (21 Grafiken) ...")
    erstelle_alle_grafiken(
        df_eval, ground_truth, empfehlungen, coverage, seg_stats, segmente,
        train_test, df_produkte_alle, dq_info, df_ausgesch_kat, df_filter_vgl, df_sensitivitaet,
    )
    print(f"  25 Grafiken in {AUSGABE_DIR}/")

    print("\n[9/9] PDF-Bericht & CSV-Export ...")
    erstelle_pdf(
        df_eval=df_eval,
        ground_truth=ground_truth,
        empfehlungen=empfehlungen,
        n_bestellungen=n_bestellungen,
        n_bestellungen_ge2=n_bestellungen_ge2,
        coverage=coverage,
        train_test=train_test,
        hypothese=hypothese,
        dq_info=dq_info,
        df_ausgeschlossen=df_ausgesch_kat,
        df_filter_vgl=df_filter_vgl,
        df_sensitivitaet=df_sensitivitaet,
        mapping_qualitaet=mapping_qualitaet,
    )
    exportiere_metriken(
        df_eval=df_eval,
        coverage=coverage,
        hypothese=hypothese,
        dq_info=dq_info,
        df_filter_vgl=df_filter_vgl,
        df_sensitivitaet=df_sensitivitaet,
        df_identische=df_identische,
        df_ausgeschlossen=df_ausgesch_kat,
        train_test=train_test,
        n_bestellungen=n_bestellungen,
        n_bestellungen_ge2=n_bestellungen_ge2,
        mapping_qualitaet=mapping_qualitaet,
        kategorie_precision=p_kat,
    )
    exportiere_produkt_metriken(df_eval)
    exportiere_datenqualitaet(dq_info, df_ausgesch_kat, df_identische, df_filter_vgl, df_sensitivitaet)

    print(f"\n{'='*40}")
    print("ERGEBNIS")
    print(f"{'='*40}")
    print(f"Datenlage:    {len(df_gefiltert):,}/{len(df_produkte_alle):,} Artikel evaluierbar")
    print(f"Precision@6:  {df_eval['precision'].mean():.3f} (Durchschnitt)")
    print(f"Recall@6:     {df_eval['recall'].mean():.3f} (Durchschnitt)")
    print(f"F1@6:         {df_eval['f1'].mean():.3f} (Durchschnitt)")
    print(f"Oe TP pro Produkt: {df_eval['tp'].mean():.1f} von {TOP_N}")
    print(f"Coverage:     {coverage.get('anteil_min_1_treffer', 0):.1%} (mind. 1 Treffer)")
    tr_p = f"{df_tr['precision'].mean():.3f}" if len(df_tr) > 0 else "—"
    te_p = f"{df_te['precision'].mean():.3f}" if len(df_te) > 0 else "—"
    print(f"Train/Test:   P@6 Train={tr_p} / Test={te_p}")
    print(f"Hypothese H1: {hypothese['ergebnis'].upper()}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
