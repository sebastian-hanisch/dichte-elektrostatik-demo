# Dichte und Drahtlänge – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-dichte-elektrostatik-demo.streamlit.app/)** -->

Stück 11 von 15 der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Konzepte-Seite der Reihe gibt es noch nicht). Stück [9, Quadratische Platzierung](https://github.com/sebastian-hanisch/quadratische-platzierung-demo), relaxiert die Orte reell, Stück [10, Look-ahead-Legalisierung](https://github.com/sebastian-hanisch/look-ahead-legalisierung-demo), legt sie in jeder Runde **hart** auf die Plätze und bindet sie mit Ankern zurück. Dieses Stück fragt, **ob eine weiche Dichtestrafe die harte Legalisierung ersetzen kann** (Kante der Reihe: Kontrast zu Stück 10): die Dichte geht als **Strafe** in die Zielfunktion ein, **elektrostatisch** gerechnet (die Bauteile sind Ladungen, die Poisson-Gleichung gibt das Feld, das sie auseinanderschiebt), ihr Gewicht wächst, und gerundet wird am Ende. Gemessen werden die Lücke zum Optimum, der Überlauf und der Aufwand gegen **keine** Strafe, eine **lokale Überlaufstrafe** und die Schleife aus Stück 10.

## Warum dieses Problem

Eine quadratische Platzierung zieht verbundene Bauteile zusammen und kollabiert dabei zu einem Klumpen; die Rundung muss ihn dann auf die Plätze verteilen. Stück 10 verteilt hart und lässt die Platzierung mit wachsenden Ankern nachziehen. Die Analytische Platzierung der Praxis (ePlace und Nachfolger, **nicht eingesehen**) geht den anderen Weg: sie ersetzt die harte Verteilung durch eine **glatte Strafe auf die Dichte**, deren Gradient ein elektrisches Feld ist. Die Kernfrage: **was bringt die Elektrostatik gegenüber keiner Strafe, gegenüber einer lokalen Strafe und gegenüber der harten Schleife, und was braucht sie dafür?**

Die Antwort dieser Messung: Gegenüber **keiner** Strafe streut die Elektrostatik sehr viel gleichmäßiger (Überlauf 0,20 statt 0,83 bei n = 8); gegenüber der **lokalen** Strafe nicht (0,20 gegen 0,22): ihr Vorteil liegt im **Ergebnis nach der Rundung** (Lücke zum Optimum im Median 3,9 gegen 6,8 %; Stück 10: 6,2 %) und darin, dass die Rundung die Orte nur etwa ein Drittel so weit verschieben muss wie ohne Strafe. Sie kostet rund **900 Auswertungen** gegen 48 lineare Systeme in Stück 10 (nicht vergleichbare Einheiten), sie **braucht zwei Hilfen** (eine **Randzone** um den Block der Plätze und eine **weich gezeichnete Ladung**, die erst schärfer wird), und **mehr Streuung ist nicht besser**: die Lücke ist in der Mitte des Weges am kleinsten. Bei n = 12 und 16 schrumpft der Vorsprung, gegenüber der lokalen Strafe verschwindet er bei n = 16. Das sind **Befunde an sechs erfundenen Familien, erfundenen Pads und einer eigenen einfachen Fassung**, keine Sätze.

## Modell

- **Aufgabe** (wie in Stück 9 und 10): `n` Bauteile auf `n` Plätze eines Rasters (Manhattan-Abstand), dazu `p` **feste Pads** auf einem Ring um das Raster; jedes Pad hängt mit dem Grundgewicht (Median der positiven Flüsse, mindestens 1) an zwei zufälligen Bauteilen. `cost(π) = Σ_{i<k} F[i,k] · |p_i − p_k|₁ + Σ_{i,Pad} w · |p_i − Pad|₁`. Kern (Kosten, Aufzählung, Zufall, sechs Familien der Flussmatrix, Pads, Platzierung, Rundung, Schleife, Ungarische Methode) als Kopie aus Stück 9 und 10 (`dla_place.py`, `dla_loop.py`, `dla_cost.py`, `dla_landscape.py`, `dla_rng.py`, `dla_instance.py`, `dla_assignment.py`). Die Familien und die Pads sind **eigene, einfache Erfindungen**, keine echten Platinen und keine QAPLIB-Instanzen.
- **Glatte Länge** (`dla_penalty.py`): `W(x) = Σ F · (√(dx² + γ²) + √(dy² + γ²))` über Paare und Pads, `γ = 0,3`: die Länge der Kosten, an den Ecken abgerundet; die Umgewichtung `F / max(|Δ|, ε)` aus Stück 9 ist ihr Gradient.
- **Dichte** (`dla_density.py`): über dem Block der Plätze plus einer **Randzone** liegt ein Bin-Raster (zwei Bins je Platzabstand). Jedes Bauteil ist eine Ladung der Masse 1, gleichmäßig auf ein Quadrat der Kantenlänge `σ` (**Ladungsbreite**) verteilt; die Überdeckung der Bins ist exakt. Sollwert: die gleichmäßige Dichte `n / Fläche`.
- **Elektrostatisch:** `ρ̃ = ρ − Mittel(ρ)`, `−∇²ψ = ρ̃` mit Neumann-Rand, gelöst mit der Kosinus-Transformation **als kleine Matrix** (kein FFT-Nachbau), Energie `½ Σ ρ̃ ψ`. **Lokal:** `Σ max(ρ − Soll, 0)²`. **Keine:** nur `W`. Der Gradient nach den Orten ist in beiden Modellen exakt (Test gegen Differenzenquotienten).
- **Strafverfahren:** Startwert = Platzierung aus Stück 9 (zwei Umgewichtungen) plus eine kleine feste Streuung (SplitMix64, ±0,15 Plätze), damit gestapelte Bauteile auseinanderlaufen können; `f = W + λ · N`, `λ` beginnt so, dass beide Gradienten gleich groß sind, und wächst je Stufe um den Faktor 1,8 (**zwölf Stufen**); je Stufe 25 Abstiegsschritte mit Armijo-Rückverfolgung; die Ladungsbreite sinkt linear von dem gewählten Wert (Standard 3) auf 1; die Orte bleiben im Gebiet. Gezählt werden die **Auswertungen** von `f` samt Gradient, jeder Versuch der Rückverfolgung zählt.
- **Rundung:** nach jeder Stufe die lineare Zuordnung der Orte auf die Plätze (Stück 9, Quadrate auf acht Stellen gerundet); gewertet wird die **letzte Stufe** oder die **billigste** der zwölf legalen Anordnungen. **Überlauf** und Verschiebung werden für alle Modelle mit Einheitsquadraten gemessen.
- **Vergleiche:** Stück 9 (einmal runden, zwei Umgewichtungen) und die Schleife aus Stück 10 (Zuordnung, 15 Runden, Ankergewicht 0,2, zwei Umgewichtungen) mit ihren Standardwerten.
- **Lücke und Vergleichswert:** `(Kosten − Vergleichswert) / Vergleichswert`; bis `n = 9` ist der Vergleichswert das aufgezählte Optimum (mit den Pads der Instanz), ab `n = 12` die beste von 100 steilsten Abstiegen aus Zufallsstarts (obere Grenze der Lücke; „erreicht“ heißt gleich oder besser).
- **Experimente:** (1) das Feld (Dichte und Kraft am Start, Handbeispiel); (2) der Weg über die Stufen; (3) die Modelle im Vergleich; (4) was die Fassung braucht (Randzone, Ladungsbreite); (5) jenseits der Aufzählung (`n` = 12, 16). Formeln im 📐-Expander der App.

## Methodik

Experiment 1, 2 und der erste Teil von 3 rechnen **eine** Instanz (Pads, Modell, Randzone, Ladungsbreite und Rundungsregel sind Regler; `n` = 6, 8, 9 mit Aufzählung als Vergleichswert, `n` = 12, 16 mit der besten von 100 Abstiegen). Standard: Zwei Cluster, `n = 8`, Seed 48, vier Pads, elektrostatisch, Randzone 0,5, Ladungsbreite 3, letzte Stufe. Der Vergleich in Experiment 3 rechnet auf Knopfdruck je Familie 20 Instanzen bei `n = 8` (Seeds 100000 bis 100019) mit 2, 4 und 8 Pads: drei Modelle mit zwölf Stufen, Stück 9 und Stück 10; Experiment 5 rechnet dasselbe für `n = 12` (5 Instanzen je Familie) und `n = 16` (3 Instanzen je Familie); Experiment 4 rechnet 120 Instanzen (`n = 8`, 4 Pads) mit je sieben Läufen. Der Vergleichswert hängt von den Pads ab und wird für jede Zahl von Pads neu bestimmt. Alle Aussagen laufen über feste Seeds, nie über einen einzelnen. **Die Parameter (zwölf Stufen, Faktor 1,8, Randzone 0,5, Ladungsbreite 3) wurden an 30 Instanzen (Seeds 100000 bis 100004, 4 Pads) vorgewählt**; eine einmalige Messung an frischen Seeds (200000 bis 200019, in der Planung, nicht Teil der Tests) hat die Mediane bestätigt (n = 8, elektrostatisch: 4,0 gegen 3,9 %). **Aufwand wird in Auswertungen gezählt, nicht in Zeit**; Zeiten stehen nirgends als Zahl in Tests oder in dieser README.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Prozentwerte auf eine Stelle: `n = 8` mit Toleranz 0,06, `n = 12` und `16` mit 0,3; Zählungen bei `n = 8` exakt, sonst ±3). Reihenfolge der Familien: Netzliste, gleichverteilt, dünn besetzt, zwei Cluster, Stern, wenige starke Flüsse. Medianwerte über alle Instanzen und Pad-Zahlen, sofern nicht anders genannt.

| Frage | Befund |
|---|---|
| Handbeispiel | Zwei Bauteile auf zwei Plätzen (Raster 1×2, Randzone 0, Ladungsbreite 1): bei x = 0,4 und 0,6 ist die Dichte je Bin-Spalte **0,2 / 1,8 / 1,8 / 0,2** (Sollwert 1), der Überlauf **0,4**; die Dichtestrafe (Energie 0,08) schiebt Bauteil 1 nach links und Bauteil 2 nach rechts (Gradient ±0,2) |
| Standardinstanz (Zwei Cluster, n = 8, Seed 48, 4 Pads) | Optimum **300**, mittlere Kosten aller Anordnungen 28,7 % darüber. Einmal runden 369, Schleife aus Stück 10 **319**. **Elektrostatisch**, Kosten nach der Rundung je Stufe 301, 300, 300, 300, 300, 300, 300, 300, 301, 301, 305, **305** (Lücke 1,7 %); billigste Stufe **300 (Optimum)** schon in Stufe 2. Lokale Strafe: letzte Stufe 309, billigste 300 (Stufe 2). Keine Strafe: **359**, billigste 346 (Stufe 1). Überlauf elektrostatisch 0,74 → **0,20**, lokal 0,59, keine Strafe 0,91; Verschiebung bei der Rundung 7,7 → 4,5; 76 Auswertungen in Stufe 1, **906** insgesamt (lokal 904, keine 901). Dichte am Startwert das 1,67-fache des Sollwerts, Überlauf 0,83. **Ohne Randzone** letzte Stufe 352, ohne Weichzeichnen 300 bei Überlauf 0,72. **Eine Instanz, kein Befund über die Familie** |
| Lücke nach der Rundung, n = 8 (360 Instanzen) | **elektrostatisch 3,9 %** (billigste Stufe **1,6 %**), lokale Strafe 6,8 %, keine Strafe 7,7 %, Stück 9 8,1 %, Stück 10 6,2 %. Vergleichswert erreicht in 54 / 112 (elektrostatisch letzte / billigste Stufe), 18 (lokal), 18 (keine), 23 (Stück 9), 34 (Stück 10) von 360. Je Pad-Zahl (2 / 4 / 8): elektrostatisch 6,3 / 4,0 / 2,2 %; Stück 10 10,8 / 6,3 / 3,6 %; lokal 10,8 / 7,1 / 4,8 %; keine 12,4 / 9,4 / 4,3 %; billigste Stufe 2,4 / 1,8 / 0,7 % |
| Instanz für Instanz, n = 8 | elektrostatisch gegen Stück 10: besser in **212**, schlechter in 108, gleich in 40 (billigste Stufe: 275 / 32 / 53); gegen die lokale Strafe: 237 / 94 / 29 |
| Familien, n = 8 (60 Instanzen je Familie; Stück 9 / Stück 10 / keine / lokal / elektrostatisch / billigste Stufe) | Netzliste 9,0 / 7,4 / 7,7 / 5,9 / **3,9** / 1,9; gleichverteilt 5,4 / 5,1 / 5,3 / 4,4 / **2,8** / 1,4; dünn 8,7 / 7,3 / 8,5 / 8,9 / **2,7** / 0,9; zwei Cluster 6,2 / 2,2 / 6,0 / 5,2 / **1,5** / 0,0; Stern 8,5 / 7,7 / 8,8 / 7,8 / **7,9** / 3,8; schief 13,6 / 9,3 / 14,5 / 9,9 / **7,1** / 3,1. **In fünf von sechs Familien besser als Stück 10; beim Stern gleich (7,9 gegen 7,7 %)** |
| **Streuung und Verschiebung, n = 8** | Überlauf am Ende (Einheitsquadrate): elektrostatisch **0,20**, lokal 0,22, keine Strafe **0,83**; Verschiebung bei der letzten Rundung 4,0 / 5,7 / 11,3; Streuung der Orte (1 = so verteilt wie die Plätze) 1,32 / 1,28 / 0,09 (die Randzone lässt die Orte über den Block hinauslaufen); Auswertungen je Instanz 908 / 896 / 901 |
| **Der Weg über die Stufen, n = 8 (elektrostatisch)** | Lücke je Stufe 5,7, 4,5, 3,6, 3,4, 3,2, **3,1**, 3,4, 3,8, 3,8, 3,9, 3,8, 3,9 %; der Überlauf fällt dabei von 0,74 auf 0,20 und die Verschiebung von 8,9 auf 4,0: **die gleichmäßigste Stufe ist die letzte, die billigste liegt in der Mitte (Stufe 6)** |
| Lücke nach der Rundung, n = 12 (90 Instanzen) | elektrostatisch **8,6 %** (billigste Stufe 3,1 %), Stück 10 9,1 %, lokal 10,1 %, keine 12,0 %, Stück 9 12,5 %; je Pad-Zahl (2 / 4 / 8): elektrostatisch 8,9 / 9,1 / 5,3 %, Stück 10 10,6 / 10,6 / 5,9 %; besser als Stück 10 in 50, schlechter in 40 von 90. **Stück 10 ist besser bei Netzliste (10,3 gegen 11,5 %), gleichverteilt (6,9 gegen 8,0 %) und zwei Clustern (3,0 gegen 4,8 bis 7,2 %, je nach Rechner)**, elektrostatisch bei dünn (7,5 gegen 12,1 %), Stern (8,9 gegen 9,9 %) und schief (11,2 gegen 19,5 %). Überlauf 0,18 / 0,19 / 0,89, Verschiebung 7,6 / 8,7 / 19,6 (elektrostatisch / lokal / keine) |
| Lücke nach der Rundung, n = 16 (54 Instanzen) | elektrostatisch **8,0 %** (billigste Stufe 4,0 %), **lokal 8,5 %**, Stück 10 10,4 %, keine 14,1 %, Stück 9 13,6 %; je Pad-Zahl (2 / 4 / 8): elektrostatisch 8,3 / 10,6 / 6,7 %, lokal 10,4 / 10,5 / 5,4 %; besser als Stück 10 in 35, schlechter in 18 von 54; **gegen die lokale Strafe 27 besser, 25 schlechter**: kein Vorsprung. Überlauf 0,16 / 0,18 / 0,92, Verschiebung 10,6 / 11,4 / 31,5 |
| **Was die Fassung braucht** (n = 8, 4 Pads, 120 Instanzen, letzte Stufe; Stück 9: 8,8 %) | **Randzone** 0 / 0,25 / 0,5 / 1,0: Lücke 9,6 / 6,0 / **4,0** / 4,6 %, Überlauf 0,13 / 0,17 / 0,20 / 0,36 (ohne Randzone schlechter als Stück 9); **Ladungsbreite** zu Beginn 1 (ohne Weichzeichnen) / 2 / 3 / 4: Lücke 6,6 / 4,2 / **4,0** / 6,7 %, Überlauf **0,53** / 0,20 / 0,20 / 0,21: ohne Weichzeichnen bleibt der Überlauf hoch |
| Orakel | Dichte gleich einer unabhängigen Doppelschleife über Bauteile und Bins; Poisson-Lösung gegen die ausgeschriebene Laplace-Matrix mit Neumann-Rand; Gradient gegen Differenzenquotienten; Kosten jeder gerundeten Anordnung gleich einer einzeln geschriebenen Doppelschleife; glatte Länge gleich einer Doppelschleife; keine Stufe liegt unter dem aufgezählten Optimum (n = 6, alle Familien, alle Modelle); die Rundung gleich `scipy.optimize.linear_sum_assignment` |

**Was die Zahlen nicht sagen:** Die Familien und die **Pads** (Zahl, Ort, Gewicht, Verbindungen) sind erfunden; eine andere Ankeranordnung kann die Zahlen ändern. Die Fassung ist **eigen und einfach** (glatte Länge, Bin-Raster, Randzone, Weichzeichnen, Strafverfahren mit Rückverfolgung); ePlace, RePlAce und DREAMPlace sind nicht gebaut und nicht nachgelesen, und die Parameter sind an wenigen Instanzen vorgewählt. Der Vergleich mit Stück 10 ist einer **Fassung gegen eine Fassung** (Stück 10 mit seinen Standardwerten, nicht gegen die Elektrostatik abgestimmt). Die **billigste Stufe** rundet zwölfmal und hat damit mehr Versuche als die letzte Stufe und als Stück 10; die Hauptzahl ist die letzte Stufe. Ab `n = 12` sind die Lücken obere Grenzen: „0,0 %“ heißt „so gut wie die beste von 100 Abstiegen“, nicht „optimal“. Bei `n = 16` sind es drei Instanzen je Familie.

## Ehrliche Grenzen

- **Erfundene Pads und Familien:** Befunde gelten für diese Anker und diese Familien.
- **Eigene, einfache Fassung:** keine Nesterov-Beschleunigung mit Schrittschätzung (sie blieb in der Vormessung an gestapelten Bauteilen hängen), keine Füllzellen, keine gewichtete Durchschnittslänge, keine Makros; der Name „ePlace-artig“ meint nur glatte Länge plus Poisson-Dichte mit wachsendem Gewicht.
- **Volle Belegung:** `n` Bauteile auf `n` Plätzen ist die härteste Dichteaufgabe; in der Praxis ist ein Teil der Fläche frei. Die Randzone ist unsere Antwort darauf, nicht die der Literatur.
- **Fassung gegen Fassung:** Stück 10 läuft mit seinen Standardwerten; mit anderer Abstimmung kann sich die Rangfolge verschieben.
- **Aufwand in Auswertungen, nicht in Zeit:** Auswertungen der Dichtestrafe (Strafverfahren) und lineare Systeme (Stück 10) sind verschiedene Einheiten.
- **Rechnerabhängigkeit:** Ein Strafverfahren mit Rückverfolgung (Verzweigungen) und einigen hundert Gradienten ist empfindlich. In der Vormessung änderte ein Rauschen von 1e-13 im Startwert bei `n = 8` keine Anordnung (24 von 24), bei `n = 12` aber 3 von 12; die **Mediane** über viele Instanzen blieben stabil (`n = 12`: 8,6 / 8,7 / 8,6 %). **Einzelergebnisse bei `n ≥ 12` und die Zählungen „besser in k von n“ sind rechnerabhängig; belastbar sind die Mediane** (deshalb die Toleranzen in den Tests). Auch die Mediane schwanken etwas: in drei Läufen der Linux-CI auf unterschiedlichen Rechnern wichen sie bei `n = 8` um bis zu 0,3, bei `n = 12` um bis zu 0,6 (bei zwei Clustern, nur 15 Instanzen: elektrostatisch 4,8 / 6,7 / 7,2 %) und bei `n = 16` um bis zu 0,8 Prozentpunkte ab. Die Zahlen in dieser Tabelle stammen aus dem Lauf unter Windows; die Tests erlauben dafür 0,4 / 0,8 / 1,2 Prozentpunkte. Die Rangfolgen, die die README behauptet, blieben in allen Läufen gleich.
- **Vergleichswert ab n = 12:** die beste von 100 Abstiegen ist selbst ein Abstiegsergebnis; die Lücke ist eine obere Grenze der wahren Lücke.
- **Wenige Instanzen:** n = 8: 20 je Familie, n = 12: 5, n = 16: 3.
- **Quellen:** ePlace (Lu u. a.), RePlAce, DREAMPlace und die Poisson-Dichte der analytischen Platzierung sind aus der Fachliteratur übernommen und **nicht nachgelesen**; Selbstangaben der Autoren zur Güte werden nicht übernommen. Glatte Länge, Randzone, Weichzeichnen, Strafverfahren und der Vergleich mit Stück 10 sind eigene Herleitungen und werden durch Tests (Handrechnung, Differenzenquotient, Doppelschleife, ausgeschriebene Laplace-Matrix, Aufzählung) geprüft.

## Verifikation

- **Tests** (`pytest tests/`): **243 Tests**, rund zwölf Minuten (Windows, Python 3.14), sechs Minuten in der Linux-CI. Handwerte (Dichte, Überlauf und Kraft am Handbeispiel von Hand), Eigenschaften (Masse gleich `n`, ein lückenloses Pflaster hat Überlauf und Energie 0, die Energie ist nichtnegativ, die Spiegelung spiegelt die Kraft, Stufen und `λ` folgen dem Plan, die Auswertungen steigen, die Orte bleiben im Gebiet, der Lauf ist deterministisch, Rauschen 1e-13 im Startwert ändert bei `n = 8` keine Anordnung), Orakel (Doppelschleife für die Dichte, ausgeschriebene Laplace-Matrix, Differenzenquotienten, einzeln gerechnete Kosten, Aufzählung, `linear_sum_assignment`; nur mit `requirements-dev.txt`), Verdikte an handgesetzten Werten, Abbildungen, PDF, Rauchtest und Pfade der App (jede Familie, jede Einstellung, `n` = 9, 12, 16, alle Rechenknöpfe), Layout, Footer und Laufzeit-Importe. `test_claims.py` rechnet jede Zahl dieser README nach.
- **Kopie:** Tests vergleichen die Familien mit denen von Stück 1 bis 10, sofern die Schwester-Repos nebenan liegen.
- **Nicht getestet:** Zeiten; kein Mutationstest; Linux-CI: läuft grün, aber erst nach Anpassung der Toleranzen (siehe Rechnerabhängigkeit); Zahlen der Tabellen nicht auf Linux nachgerechnet.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Vergleiche über feste Instanzen, Grenzen, Formulierung |
| `dla_rng.py` | SplitMix64 (Kopie aus Stück 1) |
| `dla_instance.py` | Instanzen: sechs Familien der Flussmatrix, Raster, Handbeispiele |
| `dla_cost.py` | Kosten, Tausch-`Δ` (Kopie aus Stück 2) |
| `dla_landscape.py` | Aufzählung für das Optimum (Kopie aus Stück 2) |
| `dla_assignment.py` | Ungarische Methode (Kopie aus Stück 8) |
| `dla_place.py` | Pads, lineares System, Umgewichtung, Rundung, Kosten mit linearem Term (Kopie aus Stück 9) |
| `dla_loop.py` | Pseudo-Anker, Legalisierer, Schleife (Kopie aus Stück 10) |
| `dla_density.py` | Bin-Raster, Überdeckung, Poisson-Gleichung (Kosinus-Matrix), die beiden Dichtestrafen mit Gradient |
| `dla_penalty.py` | Glatte Länge, Strafverfahren mit Stufen und Rückverfolgung, Rundung je Stufe |
| `dla_evaluation.py` | Zeilen für die Vergleiche, Zusammenfassungen, Ablation, Verdikte |
| `dla_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dla_pdf_export.py` | PDF-Export des Ergebnisses |
| `dla_constants.py` | Größen, Familien, Dichtestrafe, feste Reihen, Farben, Presets |
| `dla_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `dla_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; kein `scipy` zur Laufzeit) |
| `tests/` | Dichte, Strafverfahren, Platzierung, Schleife, Zuordnung, Kosten, Instanzen, Landschaft, Auswertung, Verdikte, Abbildungen, Presets, Layout, AppTest, Orakel, Zahlen der README, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **ePlace im Original** (Nesterov mit Schrittschätzung, Füllzellen, gewichtete Durchschnittslänge, FFT, GPU) **und lernbasierte Platzierung.**
- **Dichte mit Makros, Timing und Routing.**
- **Nachbearbeitung der gerundeten Anordnung** (Tausch, Annealing): Stück 12.
- **Zeitmessung:** gezählt wird in Auswertungen.
- **Asymmetrische Flüsse und Abstände.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
