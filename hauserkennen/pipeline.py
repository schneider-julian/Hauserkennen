"""End-to-end steps used by the command line."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable

import cv2

from .buildings import Building, filter_candidates, load_buildings, save
from .imagery import (BudgetExceeded, MapillaryClient, StreetViewClient, collect_google, collect_mapillary)
from .matching import MIN_INLIERS, FacadeMatcher, draw_matches, read_image, resize_max
from .report import write_csv, write_geojson, write_html

_ADDR = re.compile(r"^(.*?)[\s,]*(\d+\s*[a-zA-Z]?)?(?:\s*/.*)?$")


def _norm_street(s: str) -> str:
    s = s.lower().replace("ß", "ss")
    s = re.sub(r"str\.?$", "strasse", s.strip())
    return re.sub(r"[^a-z0-9äöü]", "", s)


def _split_address(text: str) -> tuple[str, str | None]:
    m = _ADDR.match(text.strip())
    street, number = (m.group(1), m.group(2)) if m else (text, None)
    return _norm_street(street), (number.replace(" ", "").lower() if number else None)


def address_matches(term: str, address: str) -> bool:
    """'Heinrichstraße 131' matches 'Heinrichstraße 131' and 'Heinrichstr. 131/4', not 'Heinrichstraße 13'.
    A term without house number matches the whole street."""
    t_street, t_no = _split_address(term)
    a_street, a_no = _split_address(address)
    if not t_street or t_street != a_street:
        return False
    return t_no is None or t_no == a_no


def is_excluded(b: Building, terms: list[str]) -> bool:
    for term in terms:
        if re.fullmatch(r"(way|relation)/\d+", term.strip()) and b.osm_id == term.strip():
            return True
        if any(address_matches(term, a) for a in b.addresses):
            return True
    return False


def find_candidates(lat: float, lon: float, radius_m: float, *, source: str, cache_dir: Path, min_levels: int,
                    max_levels: int | None, min_area_m2: float, include_unknown: bool, exclude: list[str],
                    log: Callable[[str], None] = print) -> tuple[list[Building], list[Building], str]:
    """(all buildings near the circle, filtered candidates, data source used)"""
    buildings, used = load_buildings(lat, lon, radius_m, source, cache_dir, log=log)
    kept = [b for b in buildings if not is_excluded(b, exclude)]
    cands = filter_candidates(kept, radius_m, min_levels=min_levels, max_levels=max_levels,
                              min_area_unknown_m2=min_area_m2, include_unknown=include_unknown)
    return buildings, cands, used


def write_candidate_outputs(out: Path, cands: list[Building], lat: float, lon: float, radius_m: float,
                            source: str, notes: list[str]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    save(cands, out / "kandidaten.json")
    write_geojson(out / "kandidaten.geojson", cands, lat, lon, radius_m)
    write_csv(out / "kandidaten.csv", cands)
    report = out / "kandidaten.html"
    write_html(report, title="Kandidatengebäude im Suchkreis", lat=lat, lon=lon, radius_m=radius_m,
               source=source, buildings=cands, notes=notes)
    return report


def search(cands: list[Building], photos: list[str], *, provider: str, out: Path, cache_dir: Path, lat: float,
           lon: float, radius_m: float, source: str, google_key: str | None = None,
           mapillary_token: str | None = None, max_images: int = 300, max_buildings: int = 80,
           tilts: tuple[float, ...] = (0.0, 15.0, 30.0),
           log: Callable[[str], None] = print) -> tuple[Path, list[Building], dict[str, dict]]:
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "bilder"
    img_dir.mkdir(exist_ok=True)
    matcher = FacadeMatcher()

    queries, query_by_label, query_copies = [], {}, []
    for i, p in enumerate(photos, 1):
        img = read_image(p)
        feats = matcher.query_features(img, tilts, name=f"Foto {i}")
        queries += feats
        query_by_label.update({f.label: f for f in feats})
        copy = img_dir / f"foto_{i}.jpg"
        cv2.imwrite(str(copy), resize_max(img, 900))
        query_copies.append(f"bilder/{copy.name}")
        log(f"Foto {i}: {sum(len(f.keypoints) for f in feats)} Merkmale in {len(feats)} Perspektiven")

    if provider == "google":
        client = StreetViewClient(google_key or "", cache_dir, max_images=max_images)
        collect = lambda b: collect_google(b, client)  # noqa: E731
    elif provider == "mapillary":
        client = MapillaryClient(mapillary_token or "", cache_dir, max_images=max_images)
        collect = lambda b: collect_mapillary(b, client)  # noqa: E731
    else:
        raise ValueError(f"Unbekannte Bildquelle: {provider}")

    notes: list[str] = []
    scores: dict[str, dict] = {}
    todo = cands[:max_buildings]
    if len(cands) > max_buildings:
        notes.append(f"Nur die {max_buildings} nächstgelegenen von {len(cands)} Kandidaten wurden mit Bildern geprüft "
                     f"(--max-gebaeude erhöhen).")
    for n, b in enumerate(todo, 1):
        try:
            images = collect(b)
        except BudgetExceeded as e:
            notes.append(f"{e} – Suche nach {n - 1} Gebäuden abgebrochen (--max-bilder erhöhen).")
            log(str(e))
            break
        except Exception as e:  # network hiccup or API error for one building: keep going
            log(f"[{n}/{len(todo)}] {b.label}: Fehler beim Bilderholen: {e}")
            if "REQUEST_DENIED" in str(e) or "401" in str(e) or "403" in str(e):
                raise
            continue
        best = None
        for si in images:
            try:
                res = matcher.best_match(queries, read_image(si.path))
            except ValueError:
                continue
            if best is None or res.score > best[0].score:
                best = (res, si)
        rec: dict = {"bilder": len(images)}
        if best:
            res, si = best
            safe = re.sub(r"[^A-Za-z0-9]+", "_", b.osm_id)
            dst = img_dir / f"{safe}.jpg"
            shutil.copyfile(si.path, dst)
            rec.update(score=round(res.score, 1), inliers=res.inliers, farbe=round(res.color_similarity, 2),
                       anbieter=si.provider, aufnahme=si.captured, bild=f"bilder/{dst.name}", bild_url=si.url,
                       treffer_perspektive=res.query_label)
            if res.inliers >= MIN_INLIERS and res.query_label in query_by_label:
                vis = draw_matches(query_by_label[res.query_label].image, read_image(si.path), res)
                mpath = img_dir / f"{safe}_match.jpg"
                cv2.imwrite(str(mpath), vis)
                rec["match_bild"] = f"bilder/{mpath.name}"
        scores[b.osm_id] = rec
        log(f"[{n}/{len(todo)}] {b.label}: {len(images)} Bilder, Score {rec.get('score', '–')}, "
            f"Inlier {rec.get('inliers', '–')}")

    ranked = sorted(todo, key=lambda b: scores.get(b.osm_id, {}).get("score", -1.0), reverse=True)
    inl = [scores.get(b.osm_id, {}).get("inliers", 0) for b in ranked]
    top, second = (inl[0] if inl else 0), (inl[1] if len(inl) > 1 else 0)
    if top < MIN_INLIERS:
        notes.insert(0, "Kein eindeutiger Treffer: Kein Straßenbild teilt genug geometrisch passende Merkmale "
                        "mit dem Foto. Kandidaten bitte über die Street-View-Links manuell prüfen.")
    elif top >= 2 * max(second, MIN_INLIERS // 2):
        notes.insert(0, f"Deutlichster Treffer: {ranked[0].label} ({top} geometrisch bestätigte Merkmale, "
                        f"nächster Kandidat {second}). Bitte in Street View gegenprüfen.")
    else:
        notes.insert(0, "Mehrere Kandidaten liegen nah beieinander – die ersten Plätze manuell vergleichen.")

    save(ranked, out / "ergebnis_gebaeude.json")
    (out / "ergebnis_scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    write_geojson(out / "ergebnis.geojson", ranked, lat, lon, radius_m, scores)
    write_csv(out / "ergebnis.csv", ranked, scores)
    report = out / "ergebnis.html"
    write_html(report, title="Gebäudesuche per Bildvergleich", lat=lat, lon=lon, radius_m=radius_m,
               source=f"{source}, Bilder: {provider}", buildings=ranked, query_images=query_copies,
               scores=scores, notes=notes)
    return report, ranked, scores
