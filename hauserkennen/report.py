"""Outputs: HTML report, GeoJSON and CSV."""

from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

from shapely.geometry import Point, mapping

from .buildings import Building
from .geo import LocalProjection
from .views import street_view_links

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1b1b1b;background:#fafafa}
h1{font-size:1.5rem;margin:0 0 4px} h2{font-size:1.15rem;margin-top:28px}
.meta{color:#555;font-size:.9rem;margin-bottom:16px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:.9rem}
th,td{border-bottom:1px solid #e3e3e3;padding:8px;text-align:left;vertical-align:top}
th{background:#f0f0f0;position:sticky;top:0}
tr.top td{background:#fff8e1}
td.num{text-align:right;white-space:nowrap}
img.thumb{max-width:260px;max-height:200px;border:1px solid #ccc;border-radius:4px}
img.match{max-width:520px;border:1px solid #ccc;border-radius:4px}
.queries img{max-height:220px;margin-right:12px;border:1px solid #ccc;border-radius:4px}
.links a{display:inline-block;margin:0 6px 4px 0}
.note{background:#eef4ff;border:1px solid #c9d8f5;padding:10px 12px;border-radius:6px;font-size:.9rem}
.small{font-size:.8rem;color:#666}
"""


def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def write_geojson(path: Path, buildings: list[Building], lat: float, lon: float, radius_m: float,
                  scores: dict[str, dict] | None = None) -> None:
    proj = LocalProjection(lat, lon)
    circle_xy = Point(0, 0).buffer(radius_m, 64)
    circle = [list(reversed(proj.to_latlon(x, y))) for x, y in circle_xy.exterior.coords]
    features = [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [circle]},
        "properties": {"name": "Suchkreis", "radius_m": radius_m, "stroke": "#d7301f", "fill-opacity": 0.05},
    }]
    for rank, b in enumerate(buildings, 1):
        props = {
            "rang": rank, "adresse": b.label, "geschosse_osm": b.levels, "hoehe_m": b.height_m,
            "entfernung_m": round(b.distance_m), "grundflaeche_m2": round(b.area_m2), "osm": b.osm_url,
        }
        if scores and b.osm_id in scores:
            props.update({k: v for k, v in scores[b.osm_id].items() if k in ("score", "inliers", "farbe")})
        features.append({"type": "Feature", "geometry": mapping(b.polygon), "properties": props})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, buildings: list[Building], scores: dict[str, dict] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["rang", "adresse", "geschosse_osm", "hoehe_m", "entfernung_m", "grundflaeche_m2",
                    "score", "inliers", "lat", "lon", "osm"])
        for rank, b in enumerate(buildings, 1):
            s = (scores or {}).get(b.osm_id, {})
            lat, lon = b.centroid
            w.writerow([rank, b.label, b.levels or "", b.height_m or "", round(b.distance_m), round(b.area_m2),
                        s.get("score", ""), s.get("inliers", ""), f"{lat:.6f}", f"{lon:.6f}", b.osm_url or ""])


def write_html(path: Path, *, title: str, lat: float, lon: float, radius_m: float, source: str,
               buildings: list[Building], query_images: list[str] | None = None,
               scores: dict[str, dict] | None = None, notes: list[str] | None = None) -> None:
    """`scores[osm_id]` may contain: score, inliers, farbe, bild, bild_url, match_bild, anbieter, aufnahme."""
    rows = []
    for rank, b in enumerate(buildings, 1):
        lat_b, lon_b = b.centroid
        proj = LocalProjection(lat_b, lon_b)
        links = street_view_links(b, proj)
        s = (scores or {}).get(b.osm_id, {})
        sv = " ".join(f'<a href="{_e(l["url"])}" target="_blank">Street View {i}</a>' for i, l in enumerate(links, 1))
        osm = f'<a href="{_e(b.osm_url)}" target="_blank">OSM</a>' if b.osm_url else ""
        gmaps = f'<a href="https://www.google.com/maps/search/?api=1&query={lat_b:.6f},{lon_b:.6f}" target="_blank">Karte</a>'
        img_cell = ""
        if s.get("bild"):
            img_cell = f'<a href="{_e(s.get("bild_url"))}" target="_blank"><img class="thumb" src="{_e(s["bild"])}"></a>'
            if s.get("aufnahme"):
                img_cell += f'<div class="small">{_e(s.get("anbieter"))}, Aufnahme {_e(s["aufnahme"])}</div>'
        if s.get("match_bild"):
            img_cell += f'<div><img class="match" src="{_e(s["match_bild"])}"></div>'
        score_cells = ""
        if scores is not None:
            score_cells = (f'<td class="num">{_e(s.get("score", "–"))}</td><td class="num">{_e(s.get("inliers", "–"))}</td>'
                           f'<td class="num">{_e(s.get("farbe", "–"))}</td><td class="num">{_e(s.get("bilder", 0))}</td>')
        cls = ' class="top"' if scores and rank <= 3 and s.get("inliers", 0) else ""
        image_td = f"<td>{img_cell}</td>" if scores is not None else ""
        rows.append(
            f"<tr{cls}><td class='num'>{rank}</td><td><b>{_e(b.label)}</b><div class='small'>{_e(b.osm_id)}"
            f"{' · ' + _e(b.building_type) if b.building_type else ''}</div></td>"
            f"<td class='num'>{_e(b.levels or '?')}</td><td class='num'>{round(b.distance_m)} m</td>"
            f"<td class='num'>{round(b.area_m2)} m²</td>{score_cells}"
            f"<td class='links'>{sv} {osm} {gmaps}</td>{image_td}</tr>"
        )
    score_head = "<th>Score</th><th>Inlier</th><th>Farbe</th><th>Bilder</th>" if scores is not None else ""
    image_head = "<th>Bestes Vergleichsbild</th>" if scores is not None else ""
    queries = ""
    if query_images:
        queries = "<h2>Gesuchtes Foto</h2><div class='queries'>" + "".join(
            f'<img src="{_e(q)}">' for q in query_images) + "</div>"
    note_html = "".join(f"<p class='note'>{_e(n)}</p>" for n in (notes or []))
    doc = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title><style>{CSS}</style></head><body>
<h1>{_e(title)}</h1>
<div class="meta">Suchkreis {lat:.5f}, {lon:.5f} · Radius {radius_m:.0f} m · Gebäudedaten: {_e(source)} ·
{len(buildings)} Kandidaten · erstellt {datetime.now():%d.%m.%Y %H:%M}</div>
{note_html}{queries}
<h2>Kandidaten</h2>
<table><thead><tr><th>#</th><th>Gebäude / Adresse</th><th>Geschosse (OSM)</th><th>Abstand Kreismitte</th>
<th>Grundfläche</th>{score_head}<th>Links</th>{image_head}</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<p class="small">Gebäude- und Adressdaten © OpenStreetMap-Mitwirkende (ODbL), ggf. über Overture Maps.
Straßenbilder © Google bzw. Mapillary-Mitwirkende (CC BY-SA 4.0). Der Score ist ein Hinweis, kein Beweis –
das Ergebnis immer selbst in Street View prüfen.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def fmt(v: float | None, digits: int = 2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, digits)
