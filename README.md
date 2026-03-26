# Produktbasiertes Empfehlungssystem im E-Commerce

Dieses Projekt ist Teil der Abschlussarbeit *"Ähnlichkeit als Mehrwert: Produktbasiertes Empfehlungssystem im E-Commerce"* und wurde im Rahmen einer Fallstudie bei der gebana AG entwickelt.

Das System analysiert Produktbeschreibungen aus dem Produktkatalog und berechnet automatisch, welche Produkte einander am ähnlichsten sind. Pro Produkt werden die 6 ähnlichsten Artikel als Empfehlung vorgeschlagen. Anschliessend wird die Qualität dieser Empfehlungen anhand von historischen Bestelldaten überprüft.

## Voraussetzungen

Um den Code auszuführen, wird **Python** benötigt (Version 3.10 oder neuer).

### Python installieren

1. [python.org/downloads](https://www.python.org/downloads/) aufrufen
2. Die neueste Version herunterladen und installieren
3. **Wichtig:** Beim Installieren das Häkchen bei **"Add Python to PATH"** setzen

### Prüfen ob Python installiert ist

Ein Terminal (Eingabeaufforderung) öffnen und eingeben:

```
python --version
```

Wenn eine Versionsnummer erscheint (z. B. `Python 3.12.0`), ist Python bereit.

## Einrichtung

### 1. Repository herunterladen

Entweder als ZIP herunterladen (grüner Button "Code" > "Download ZIP") und entpacken, oder mit Git:

```
git clone <REPOSITORY-URL>
```

### 2. Bestellhistorie herunterladen

Die Datei mit den anonymisierten Bestelldaten ist aus Datenschutzgründen nicht direkt im Repository enthalten. Sie muss separat heruntergeladen werden:

**Download-Link:** `https://drive.proton.me/urls/BCFKWWCEJ8#jTZsHgJ4udNN`

Nach dem Herunterladen die Datei **`Bestellhistorie-anonymisiert.csv`** in den **gleichen Ordner** legen, in dem auch die Python-Skripte liegen. Die Ordnerstruktur sollte dann so aussehen:

```
Endgültig/
├── 01-Empfehlungen-berechnen.py
├── 02-Empfehlungen-auswerten.py
├── Produktbeschreibungen.csv
├── Produktliste.csv
├── Zuordnung-ID-Artikelnummer.csv
├── Bestellhistorie-anonymisiert.csv    <-- hierhin legen
├── ...
```

> **Hinweis:** Ohne diese Datei funktioniert das erste Skript (Empfehlungen berechnen) trotzdem. Nur die Evaluation (zweites Skript) benötigt die Bestellhistorie.

### 3. Benötigte Pakete installieren

Ein Terminal im Projektordner öffnen und folgenden Befehl ausführen:

```
pip install -r requirements.txt
```

Dieser Befehl installiert automatisch alle benötigten Python-Bibliotheken.

## Ausführung

Das Projekt besteht aus zwei Skripten, die **nacheinander** ausgeführt werden:

### Schritt 1: Empfehlungen berechnen

```
python 01-Empfehlungen-berechnen.py
```

Dieses Skript:
- Liest die Produktbeschreibungen ein
- Bereinigt und verarbeitet die Texte
- Berechnet die Ähnlichkeit zwischen allen Produkten (TF-IDF + Kosinus-Ähnlichkeit)
- Gibt pro Produkt die 6 ähnlichsten Artikel aus

**Ergebnis:** Im Ordner `Ergebnisse/` werden drei Dateien erstellt:
- `Produktempfehlungen.csv` — Die fertigen Empfehlungen
- `Produktverzeichnis.csv` — Übersicht aller berechneten Produkte
- `Aehnlichkeitsmatrix.npz` — Die berechnete Ähnlichkeitsmatrix

### Schritt 2: Empfehlungen auswerten

```
python 02-Empfehlungen-auswerten.py
```

Dieses Skript:
- Vergleicht die Empfehlungen mit dem tatsächlichen Kaufverhalten (Bestellhistorie)
- Berechnet Qualitätskennzahlen (Precision, Recall, F1-Score)
- Erstellt 21 Diagramme zur Visualisierung der Ergebnisse
- Generiert einen PDF-Bericht

**Ergebnis:** Im Ordner `Ergebnisse/` werden zusätzlich erstellt:
- 21 Diagramme (PNG-Dateien)
- Mehrere CSV-Dateien mit detaillierten Kennzahlen
- `Ergebnisbericht.pdf` — Zusammenfassender Bericht

## Projektstruktur

```
Endgültig/
│
├── 01-Empfehlungen-berechnen.py        # Hauptskript: Berechnung der Empfehlungen
├── 02-Empfehlungen-auswerten.py        # Evaluation und Visualisierung
├── Empfehlungssystem-Entwicklung.ipynb  # Jupyter Notebook (Entwicklung/Exploration)
├── requirements.txt                    # Liste der benötigten Python-Pakete
│
├── Produktbeschreibungen.csv           # Produktkatalog (PIM-Export)
├── Produktliste.csv                    # Produkthierarchie (Varianten → Stammprodukte)
├── Zuordnung-ID-Artikelnummer.csv      # Mapping: Shop-IDs → Artikelnummern
├── Bestellhistorie-anonymisiert.csv    # Anonymisierte Bestelldaten (separat herunterladen)
│
└── Ergebnisse/                         # Alle generierten Ergebnisse
    ├── Produktempfehlungen.csv         # Die fertigen Empfehlungen
    ├── Ergebnisbericht.pdf             # Zusammenfassender PDF-Bericht
    ├── Qualitaetskennzahlen-Gesamt.csv # Übersicht aller Kennzahlen
    ├── *.png                           # 21 Diagramme
    └── ...                             # Weitere CSV-Dateien mit Detailauswertungen
```

## Ergebnisse ansehen (ohne den Code auszuführen)

Wer die Ergebnisse nur anschauen möchte, ohne den Code selbst auszuführen, findet im Ordner **`Ergebnisse/`** alle fertigen Resultate:

- **`Ergebnisbericht.pdf`** — Der wichtigste Einstiegspunkt: ein vierseitiger Bericht mit den zentralen Kennzahlen und Diagrammen
- **`Produktempfehlungen.csv`** — Die vollständige Liste aller Produktempfehlungen (kann mit Excel geöffnet werden)
- **Die PNG-Dateien** — Einzelne Diagramme zur Verteilung der Kennzahlen, Datenqualität und Segmentanalyse

## Autor

Emircan Akyürek — Abschlussarbeit 2026
