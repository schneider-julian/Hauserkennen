"""Command line: `python -m hauserkennen kandidaten|suche ...`"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .pipeline import find_candidates, search, write_candidate_outputs


def _add_area_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Suchgebiet")
    g.add_argument("--lat", type=float, required=True, help="Breitengrad der Kreismitte (z. B. 47.08432)")
    g.add_argument("--lon", type=float, required=True, help="Längengrad der Kreismitte (z. B. 15.45938)")
    g.add_argument("--radius", type=float, default=300, help="Radius in Metern (Standard 300)")
    g.add_argument("--quelle", choices=("auto", "overpass", "overture"), default="auto",
                   help="Gebäudedaten: overpass = OpenStreetMap live, overture = Overture-Maps-Release (S3), "
                        "auto = Overpass, bei Fehler Overture (Standard)")
    f = p.add_argument_group("Filter")
    f.add_argument("--min-geschosse", type=int, default=5,
                   help="Mindestanzahl Geschosse inkl. Erdgeschoss (Standard 5)")
    f.add_argument("--max-geschosse", type=int, default=None, help="Höchstanzahl Geschosse")
    f.add_argument("--min-flaeche", type=float, default=200,
                   help="Mindest-Grundfläche in m² für Gebäude ohne Geschossangabe (Standard 200)")
    f.add_argument("--ohne-unbekannte", action="store_true",
                   help="Gebäude ohne Geschossangabe in OSM weglassen")
    f.add_argument("--ausschliessen", action="append", default=[], metavar="ADRESSE_ODER_OSM_ID",
                   help="Bereits ausgeschlossene Häuser, z. B. 'Heinrichstraße 131' oder 'way/94100643' (mehrfach möglich)")
    p.add_argument("--out", type=Path, default=Path("ergebnis"), help="Ausgabeordner (Standard ./ergebnis)")
    p.add_argument("--cache", type=Path, default=Path(".hauserkennen-cache"), help="Cache-Ordner")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m hauserkennen",
        description="Findet das Gebäude zu einem Immobilienfoto: Kandidaten aus OpenStreetMap, "
                    "Vergleich mit Google-Street-View- oder Mapillary-Bildern.",
    )
    sub = ap.add_subparsers(dest="befehl", required=True)

    k = sub.add_parser("kandidaten", help="Kandidatengebäude im Suchkreis auflisten (mit Street-View-Links, ohne API-Key)")
    _add_area_args(k)

    s = sub.add_parser("suche", help="Kandidaten holen, Straßenbilder laden und mit dem Foto vergleichen")
    _add_area_args(s)
    s.add_argument("--foto", action="append", required=True, help="Fassadenfoto aus dem Inserat (mehrfach möglich)")
    s.add_argument("--bilder", choices=("google", "mapillary"), default="google", help="Bildquelle (Standard google)")
    s.add_argument("--google-key", default=os.environ.get("GOOGLE_MAPS_API_KEY"),
                   help="Google-Maps-API-Key mit aktivierter Street View Static API (oder GOOGLE_MAPS_API_KEY)")
    s.add_argument("--mapillary-token", default=os.environ.get("MAPILLARY_TOKEN"),
                   help="Mapillary-Client-Token (oder MAPILLARY_TOKEN)")
    s.add_argument("--max-bilder", type=int, default=300,
                   help="Höchstzahl neu heruntergeladener Bilder (Kostenbremse, Standard 300)")
    s.add_argument("--max-gebaeude", type=int, default=80, help="Höchstzahl geprüfter Gebäude (Standard 80)")
    s.add_argument("--neigungen", default="0,15,30",
                   help="Virtuelle Kamera-Neigungen fürs Foto in Grad, kommagetrennt (Standard 0,15,30)")

    a = ap.parse_args(argv)
    try:
        return _run(a)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace) -> int:
    if a.befehl == "suche":
        missing = [f for f in a.foto if not Path(f).is_file()]
        if missing:
            raise ValueError(f"Foto nicht gefunden: {', '.join(missing)}")
        if a.bilder == "google" and not a.google_key:
            raise ValueError("Für --bilder google wird ein API-Key gebraucht (GOOGLE_MAPS_API_KEY oder --google-key).")
        if a.bilder == "mapillary" and not a.mapillary_token:
            raise ValueError("Für --bilder mapillary wird ein Token gebraucht (MAPILLARY_TOKEN oder --mapillary-token).")
    a.cache.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    log(f"Lade Gebäude ({a.quelle}) im Umkreis von {a.radius:.0f} m …")
    all_b, cands, a.quelle = find_candidates(
        a.lat, a.lon, a.radius, source=a.quelle, cache_dir=a.cache, min_levels=a.min_geschosse,
        max_levels=a.max_geschosse, min_area_m2=a.min_flaeche, include_unknown=not a.ohne_unbekannte,
        exclude=a.ausschliessen, log=log,
    )
    known = sum(1 for b in cands if b.est_levels)
    log(f"{len(all_b)} Gebäude geladen ({a.quelle}), {len(cands)} Kandidaten ({known} mit Geschossangabe).")

    if a.befehl == "kandidaten":
        notes = [f"{len(cands)} Kandidaten: Gebäude mit ≥ {a.min_geschosse} Geschossen"
                 + (f" oder ohne Geschossangabe und ≥ {a.min_flaeche:.0f} m² Grundfläche" if not a.ohne_unbekannte else "")
                 + ". Die Street-View-Links öffnen Google Maps direkt vor der jeweiligen Fassade."]
        if a.ausschliessen:
            notes.append("Ausgeschlossen: " + ", ".join(a.ausschliessen))
        report = write_candidate_outputs(a.out, cands, a.lat, a.lon, a.radius, a.quelle, notes)
        print(f"Bericht: {report}")
        return 0

    tilts = tuple(float(t) for t in a.neigungen.split(",") if t.strip())
    report, ranked, scores = search(
        cands, a.foto, provider=a.bilder, out=a.out, cache_dir=a.cache, lat=a.lat, lon=a.lon, radius_m=a.radius,
        source=a.quelle, google_key=a.google_key, mapillary_token=a.mapillary_token, max_images=a.max_bilder,
        max_buildings=a.max_gebaeude, tilts=tilts, log=log,
    )
    print("\nTop 5:")
    for i, b in enumerate(ranked[:5], 1):
        s = scores.get(b.osm_id, {})
        print(f"  {i}. {b.label:45s} Score {s.get('score', '–'):>6}  Inlier {s.get('inliers', '–'):>4}  "
              f"Geschosse {b.levels or '?'}")
    print(f"Bericht: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
