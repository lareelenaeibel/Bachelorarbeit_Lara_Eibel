# Bachelorarbeit – Reproduzierbarkeit der Datenanalyse

Dieses Repository enthält den Python-Code für die empirische Analyse der Bachelorarbeit **«Wer bewertet und warum? Eine Analyse der demografischen Merkmale von Online-Rezensenten»**.

Die Analyse untersucht Zusammenhänge zwischen dem Alter von Rezensent:innen, dem Sentiment von Rezensionstexten und vergebenen Sternebewertungen. Zusätzlich wird eine externe Robustheitsanalyse anhand einer Stichprobe von Amazon-Electronics-Rezensionen durchgeführt.

## Projektstruktur

```text
Bachelorarbeit_Lara_Eibel/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── notebooks/
│   ├── 01_Data_Import.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_VADER.ipynb
│   ├── 04_Regression.ipynb
│   ├── 05_Amazon_Robustness.ipynb
│   └── 06_Results.ipynb
├── results/
├── scripts/
│   └── brant_test.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 1. Repository klonen und Python-Umgebung einrichten

Das Repository zunächst klonen und in das Projektverzeichnis wechseln:

```bash
git clone https://github.com/lareelenaeibel/Bachelorarbeit_Lara_Eibel.git
cd Bachelorarbeit_Lara_Eibel
```

Anschliessend wird empfohlen, eine virtuelle Python-Umgebung anzulegen:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Unter Windows kann die Umgebung beispielsweise mit folgendem Befehl aktiviert werden:

```bash
.venv\Scripts\activate
```

Danach können die benötigten Python-Pakete installiert werden:

```bash
python -m pip install -r requirements.txt
```

## 2. Hauptdatensatz

Für die Hauptanalyse wird der Datensatz **Women's E-Commerce Clothing Reviews** verwendet.

Quelle:

[Kaggle – Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Der Datensatz umfasst im Rohzustand 23'486 Beobachtungen.

Nach dem Download muss die Datei mit folgendem Namen im Ordner `data/raw/` abgelegt werden:

```text
data/raw/Womens Clothing E-Commerce Reviews.csv
```

Der Dateiname sollte unverändert beibehalten werden, da das erste Notebook auf diesen Dateinamen zugreift.

Die daraus erzeugten Zwischenstände werden im Verzeichnis `data/processed/` gespeichert.

## 3. Reihenfolge der Analyse

Die Notebooks befinden sich im Ordner `notebooks/` und sind in der folgenden Reihenfolge vollständig auszuführen:

```text
01_Data_Import.ipynb
02_Data_Cleaning.ipynb
03_VADER.ipynb
04_Regression.ipynb
05_Amazon_Robustness.ipynb
06_Results.ipynb
```

Für eine vollständige Reproduktion sollten die Notebooks nacheinander mit einem Neustart des jeweiligen Kernels und anschliessendem vollständigem Ausführen aller Zellen ausgeführt werden.

### 01_Data_Import.ipynb

Importiert den Hauptdatensatz, prüft dessen grundlegende Struktur und speichert die für die weiteren Schritte benötigten Daten.

### 02_Data_Cleaning.ipynb

Bereinigt und verarbeitet den Hauptdatensatz für die anschliessenden Analysen.

### 03_VADER.ipynb

Berechnet das Sentiment der Rezensionstexte mithilfe von VADER und erzeugt den für die statistischen Analysen verwendeten Compound Score.

### 04_Regression.ipynb

Enthält die statistischen Hauptanalysen zu den drei Hypothesen sowie ergänzende Robustheitsanalysen.

Dazu gehören insbesondere die Regressionsmodelle zu H1, H2 und H3 sowie der Bootstrap-Vergleich der geschätzten Minima der Alterskurven von H1 und H2.

Für den gepaarten Bootstrap werden **1'000 Replikationen** mit einem festgelegten Zufalls-Seed (`42`) verwendet. Dadurch ist die Stichprobenziehung reproduzierbar. Die Berechnung kann je nach verwendeter Hardware einige Zeit in Anspruch nehmen.

Ergänzend wird ein BCa-Konfidenzintervall dokumentiert. Für die dafür verwendete gruppierte Jackknife-Berechnung werden 50 Gruppen und ein festgelegter Seed (`43`) verwendet.

### 05_Amazon_Robustness.ipynb

Führt die externe Robustheitsanalyse anhand von Amazon-Electronics-Rezensionen durch.

Die Rohdaten stammen aus dem **Amazon Reviews 2023** Dataset Repository des McAuley Lab:

[Amazon Reviews 2023 – McAuley Lab auf Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

Das Notebook greift direkt auf die Reviewdaten der Kategorie `Electronics` (`raw/review_categories/Electronics.jsonl`) zu. Die benötigten Daten werden beim Ausführen des Notebooks über das Hugging-Face-Dataset-Repository geladen; eine bereits lokal gespeicherte Amazon-Stichprobe ist daher für die Reproduktion nicht erforderlich.

Die Stichprobenziehung ist im Notebook dokumentiert. Es werden zunächst 25'000 Beobachtungen mit einem festgelegten Seed (`SEED = 42`) aus dem Streaming-Datensatz gezogen. Für die Shuffle-Operation wird eine Buffergrösse von 200'000 Beobachtungen verwendet. Nach der Datenaufbereitung umfasst die für die externe Analyse verwendete Stichprobe 24'992 Beobachtungen.

Für diesen Analyseschritt ist eine aktive Internetverbindung erforderlich.

### 06_Results.ipynb

Führt die zuvor erzeugten Resultate und Abbildungen zusammen. Dieses Notebook berechnet die Hauptmodelle nicht erneut, sondern liest die zuvor gespeicherten Ergebnisdateien ein und stellt die zentralen Resultate konsolidiert dar.

Es sollte daher erst ausgeführt werden, nachdem sowohl `04_Regression.ipynb` als auch `05_Amazon_Robustness.ipynb` vollständig ausgeführt wurden.

## 4. Reproduzierbarkeit

Die für zufallsabhängige Analyseschritte verwendeten Seeds sind direkt in den entsprechenden Notebooks festgelegt.

Bei einer vollständigen Ausführung der Notebooks in der oben angegebenen Reihenfolge werden die aufbereiteten Datensätze, Ergebnisdateien und Abbildungen erneut erzeugt.

Die zentralen Ausgabeverzeichnisse sind:

```text
data/processed/
results/
figures/
```

Bereits vorhandene Dateien in diesen Verzeichnissen können dabei durch neu erzeugte Ergebnisse überschrieben werden.

Die Notebooks verwenden relative Pfade innerhalb der dargestellten Projektstruktur. Für die vorgesehene Ausführung sollten die Notebooks daher innerhalb des Ordners `notebooks/` geöffnet und ausgeführt werden.

## 5. Hinweise zur vollständigen Reproduktion

Für eine Reproduktion von Grund auf wird folgendes Vorgehen empfohlen:

1. Repository klonen.
2. Virtuelle Python-Umgebung erstellen und `requirements.txt` installieren.
3. Den Hauptdatensatz von Kaggle herunterladen und als `data/raw/Womens Clothing E-Commerce Reviews.csv` ablegen.
4. Die Notebooks `01` bis `06` in der angegebenen Reihenfolge vollständig ausführen.
5. Prüfen, ob die neu erzeugten Dateien in `data/processed/`, `results/` und `figures/` vorliegen.

Für `05_Amazon_Robustness.ipynb` ist eine Internetverbindung erforderlich, da die Amazon-Electronics-Daten beim Ausführen des Notebooks aus dem externen Dataset Repository geladen werden.