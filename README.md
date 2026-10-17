# Hauserkennen

Findet das Gebäude zu einem Immobilieninserat, dessen genaue Adresse verborgen ist und dessen Karte nur
einen Kreis zeigt.

Das Werkzeug arbeitet in drei Schritten:

1. **Kandidaten:** Alle Gebäude im Suchkreis kommen aus OpenStreetMap, gefiltert nach Geschosszahl
   und Grundfläche. Datenquelle ist die Overpass-API; ist sie nicht erreichbar, wird automatisch das
   Overture-Maps-Release verwendet.
2. **Straßenbilder:** Für jede Fassade eines Kandidaten werden passende Straßenbilder gesucht, entweder
   Panoramen aus **Google Street View** (Static API, Blickrichtung und Neigung werden zur Fassade
   berechnet) oder Fotos aus **Mapillary** (Panoramen werden passend zugeschnitten).
3. **Bildvergleich:** Das Fassadenfoto aus dem Inserat wird mit jedem Straßenbild verglichen.
   Verwendet werden RootSIFT-Merkmale und eine RANSAC-Homographie; gezählt werden die Merkmale, die
   geometrisch zur selben Fassade passen („Inlier“). Weil Inseratsfotos oft steil nach oben
   fotografiert sind, wird das Foto zusätzlich virtuell „gesenkt“ (0°, 15°, 30°).

Ergebnis ist ein HTML-Bericht mit Rangliste, besten Vergleichsbildern samt eingezeichneten
Übereinstimmungen und direkten Street-View-Links zu jeder Fassade.

> Der Score ist ein **Hinweis, kein Beweis**. Den Treffer immer selbst in Street View gegenprüfen.

## Installation

```bash
python -m pip install -r requirements.txt     # Python ≥ 3.10
```

## 1. Kandidatenliste – ohne API-Key

```bash
python -m hauserkennen kandidaten --lat 47.08432 --lon 15.45938 --radius 280 \
    --ausschliessen "Heinrichstraße 131" --out ergebnis
```

Das erzeugt `ergebnis/kandidaten.html`. Darin gibt es pro Gebäude Links „Street View 1…n“, die
Google Maps direkt vor der jeweiligen Fassade öffnen. Damit lassen sich die Kandidaten auch von Hand
schnell durchklicken. Außerdem entstehen `kandidaten.csv`, `kandidaten.json` und `kandidaten.geojson`;
die GeoJSON-Datei lässt sich z. B. auf geojson.io ansehen.

**Suchkreis bestimmen:** Lies Mittelpunkt und Radius aus der Karte des Inserats ab. Die Koordinaten
bekommst du in Google Maps per Rechtsklick, den Radius mit „Entfernung messen“. Wähle den Radius lieber
etwas größer, denn manche Portale verschieben den Kreis absichtlich.

Für das Inserat 961/36303 (sREAL, Graz-Geidorf) ist der Kreis aus dem Screenshot rekonstruiert:
Mitte ≈ 47.08432 N / 15.45938 O (Ostende der Hilmgasse beim Hilmteich), Radius ≈ 265 m.

## 2. Automatischer Bildvergleich

### Mit Google Street View

1. Lege in der [Google Cloud Console](https://console.cloud.google.com/) ein Projekt mit Abrechnung an,
   aktiviere die **Street View Static API** und erstelle einen API-Key. Beschränke den Key am besten
   auf diese API.
2. Speichere das Fassadenfoto aus dem Inserat als Originalbild, nicht als Screenshot, und schneide es
   auf das Gebäude zu, ohne Pfeile oder Bedienelemente.

```bash
export GOOGLE_MAPS_API_KEY="…"
python -m hauserkennen suche --foto fotos/fassade.jpg \
    --lat 47.08432 --lon 15.45938 --radius 280 --ausschliessen "Heinrichstraße 131" \
    --bilder google --max-bilder 300 --out ergebnis
```

Kosten: Metadaten-Abfragen (welches Panorama steht wo?) sind laut Google kostenlos. Bildabrufe werden
nach der Google-Preisliste abgerechnet, mit monatlichem Freikontingent; die aktuellen Konditionen bitte
bei Google prüfen. Pro Gebäude werden höchstens 4 Panoramen × 2 Neigungen geladen. `--max-bilder`
begrenzt die Zahl neuer Abrufe, und Bilder im Cache werden nie doppelt bezahlt.

### Mit Mapillary (kostenlos)

Erstelle einen Client-Token unter <https://www.mapillary.com/dashboard/developers>.

```bash
export MAPILLARY_TOKEN="MLY|…"
python -m hauserkennen suche --foto fotos/fassade.jpg --lat 47.08432 --lon 15.45938 --radius 280 \
    --bilder mapillary --out ergebnis
```

Mapillary hat in Wohnstraßen oft weniger Abdeckung als Street View, dafür stehen die Bilder unter
CC BY-SA 4.0. Bei Panoramen nimmt das Werkzeug an, dass die Bildmitte der `compass_angle`-Richtung
entspricht.

### Ergebnis lesen

`ergebnis/ergebnis.html` sortiert die Gebäude nach Score:

| Wert | Bedeutung |
|---|---|
| **Inlier** | Merkmale, die geometrisch zur selben Fassade passen. Ab ca. 20 liegt wahrscheinlich dasselbe Gebäude vor; bei verschiedenen Gebäuden bleiben es meist einstellige Werte. |
| **Farbe** | Ähnlichkeit der Fassadenfarben (0–1), nur als Nebenkriterium. |
| **Bestes Vergleichsbild** | Das Straßenbild mit dem höchsten Score, darunter die verbundenen Merkmale. |

Hat der Spitzenreiter mindestens doppelt so viele Inlier wie der Zweite, meldet der Bericht einen
„deutlichsten Treffer“. Sonst bitte die ersten Plätze von Hand vergleichen.

## Optionen (Auszug)

| Option | Standard | Zweck |
|---|---|---|
| `--min-geschosse` | 5 | Mindestens so viele Geschosse inkl. Erdgeschoss. Eine Wohnung im 4. OG braucht ≥ 5. |
| `--max-geschosse` | – | Zum Beispiel 9, um Hochhäuser auszuschließen. |
| `--min-flaeche` | 200 | Mindest-Grundfläche (m²) für Gebäude **ohne** Geschossangabe in OSM. |
| `--ohne-unbekannte` | aus | Gebäude ohne Geschossangabe weglassen. Vorsicht: In OSM fehlt die Angabe oft. |
| `--ausschliessen` | – | Bereits geprüfte Adressen oder OSM-IDs, mehrfach angebbar. |
| `--quelle` | auto | `overpass`, `overture` oder `auto`. |
| `--neigungen` | 0,15,30 | Virtuelle Kamera-Neigungen, die auf das Foto angewandt werden. |
| `--max-gebaeude` | 80 | Höchstzahl Gebäude, für die Bilder geladen werden (die nächstgelegenen zuerst). |

## Grenzen

- Geschosszahlen fehlen in OSM bei vielen Gebäuden. Deshalb bleiben große Gebäude ohne Angabe im Rennen.
- Street-View-Aufnahmen können mehrere Jahre alt sein. Eine neue Fassade, etwa nach einer
  Wärmedämmung, wird dann schlecht erkannt.
- Steile Froschperspektiven, starke Weitwinkelverzerrung und kleine Screenshots senken die Trefferzahl.
  Am besten das Originalfoto in voller Auflösung verwenden.
- Innen- und Balkonfotos eignen sich nicht für den Vergleich, Fassadenfotos schon.

## Rechtliches und faire Nutzung

- Gebäude- und Adressdaten © OpenStreetMap-Mitwirkende (ODbL), ggf. über Overture Maps.
- Google: Verwende deinen eigenen Key und beachte die Google Maps Platform Terms. Street-View-Bilder
  nur für die eigene, einmalige Suche nutzen, nicht weitergeben und den Cache
  (`.hauserkennen-cache/`) danach löschen.
- Mapillary-Bilder: CC BY-SA 4.0.
- Das Werkzeug ist für die Immobiliensuche gedacht, also um ein angebotenes Gebäude zu finden, nicht
  um Personen ausfindig zu machen.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

Die Tests laufen offline. Sie verwenden synthetische Fassaden und simulierte Bildquellen.
