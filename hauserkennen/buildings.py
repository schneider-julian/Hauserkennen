"""Candidate buildings inside the search circle.

Two interchangeable sources:

* ``overpass`` - live OpenStreetMap data through the Overpass API (default).
* ``overture`` - the monthly Overture Maps release on AWS S3 (OSM-derived for
  Austria). Slower, but works where Overpass is blocked.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import shapely
from shapely import wkb
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from . import http
from .geo import LocalProjection, bbox_around

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERTURE_BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/"

# Building types that can never be the multi-storey residential building we look for.
NON_CANDIDATE_TYPES = {
    "garage", "garages", "shed", "roof", "carport", "hut", "kiosk", "service",
    "transformer_tower", "greenhouse", "church", "chapel", "bunker", "toilets",
}
FLOOR_HEIGHT_M = 3.0


@dataclass
class Building:
    osm_id: str
    polygon: Polygon  # lon/lat
    levels: int | None = None
    height_m: float | None = None
    building_type: str | None = None
    name: str | None = None
    addresses: list[str] = field(default_factory=list)
    source: str = ""
    # filled in by `prepare()`
    distance_m: float = math.nan
    area_m2: float = math.nan

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return c.y, c.x

    @property
    def est_levels(self) -> int | None:
        if self.levels:
            return self.levels
        if self.height_m:
            return max(1, round(self.height_m / FLOOR_HEIGHT_M))
        return None

    @property
    def est_height_m(self) -> float | None:
        if self.height_m:
            return self.height_m
        if self.levels:
            return self.levels * FLOOR_HEIGHT_M
        return None

    @property
    def osm_url(self) -> str | None:
        if self.osm_id.startswith(("way/", "relation/")):
            return f"https://www.openstreetmap.org/{self.osm_id}"
        return None

    @property
    def label(self) -> str:
        if self.addresses:
            return ", ".join(self.addresses[:3]) + (" …" if len(self.addresses) > 3 else "")
        return self.name or f"ohne Adresse ({self.osm_id})"

    def projected(self, proj: LocalProjection) -> Polygon:
        return shapely.transform(self.polygon, proj.lonlat_array_to_xy)

    def to_dict(self) -> dict:
        lat, lon = self.centroid
        return {
            "osm_id": self.osm_id,
            "label": self.label,
            "addresses": self.addresses,
            "levels": self.levels,
            "height_m": self.height_m,
            "est_levels": self.est_levels,
            "building_type": self.building_type,
            "name": self.name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "distance_m": round(self.distance_m, 1),
            "area_m2": round(self.area_m2, 1),
            "source": self.source,
            "polygon": [[round(x, 7), round(y, 7)] for x, y in self.polygon.exterior.coords],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Building":
        b = cls(
            osm_id=d["osm_id"],
            polygon=Polygon(d["polygon"]),
            levels=d.get("levels"),
            height_m=d.get("height_m"),
            building_type=d.get("building_type"),
            name=d.get("name"),
            addresses=list(d.get("addresses") or []),
            source=d.get("source", ""),
        )
        b.distance_m = d.get("distance_m", math.nan)
        b.area_m2 = d.get("area_m2", math.nan)
        return b


@dataclass
class AddressPoint:
    lat: float
    lon: float
    text: str


# --------------------------------------------------------------------------- helpers

def _parse_int(value) -> int | None:
    if value is None:
        return None
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(value))
    if not m:
        return None
    return int(round(float(m.group(1).replace(",", "."))))


def _parse_float(value) -> float | None:
    if value is None:
        return None
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(value))
    return float(m.group(1).replace(",", ".")) if m else None


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return max(polys, key=lambda g: g.area) if polys else None


def _address_text(street: str | None, number: str | None, unit: str | None = None) -> str | None:
    if not street and not number:
        return None
    text = " ".join(p for p in (street, number) if p)
    return f"{text}/{unit}" if unit else text


def assign_addresses(buildings: list[Building], points: list[AddressPoint], max_dist_m: float = 15.0) -> None:
    """Attach address points to the building that contains them (or the nearest one)."""
    if not buildings or not points:
        return
    proj = LocalProjection(*buildings[0].centroid)
    polys = [b.projected(proj) for b in buildings]
    tree = STRtree(polys)
    for p in points:
        pt = Point(proj.to_xy(p.lat, p.lon))
        hits = [int(i) for i in tree.query(pt, predicate="intersects")]
        if not hits:
            i = int(tree.nearest(pt))
            if polys[i].distance(pt) > max_dist_m:
                continue
            hits = [i]
        for i in hits:
            if p.text not in buildings[i].addresses:
                buildings[i].addresses.append(p.text)
    for b in buildings:
        b.addresses.sort(key=_address_sort_key)


def _address_sort_key(text: str):
    m = re.match(r"(.*?)(\d+)(.*)", text)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (text, 0, "")


def prepare(buildings: list[Building], lat: float, lon: float) -> list[Building]:
    """Compute distance to the search centre and footprint area."""
    proj = LocalProjection(lat, lon)
    centre = Point(0.0, 0.0)
    for b in buildings:
        poly = b.projected(proj)
        b.distance_m = poly.distance(centre)
        b.area_m2 = poly.area
    return buildings


def filter_candidates(
    buildings: list[Building],
    radius_m: float,
    min_levels: int = 5,
    max_levels: int | None = None,
    min_area_unknown_m2: float = 200.0,
    include_unknown: bool = True,
    exclude: set[str] | None = None,
) -> list[Building]:
    """Keep buildings inside the circle that can plausibly have the flat.

    Buildings without a storey count are kept when their footprint is large
    enough, because OSM often lacks `building:levels`.
    """
    exclude = exclude or set()
    out = []
    for b in buildings:
        if b.distance_m > radius_m or b.osm_id in exclude:
            continue
        if (b.building_type or "").lower() in NON_CANDIDATE_TYPES:
            continue
        lv = b.est_levels
        if lv is not None:
            if lv < min_levels or (max_levels is not None and lv > max_levels):
                continue
        elif not include_unknown or b.area_m2 < min_area_unknown_m2:
            continue
        out.append(b)
    out.sort(key=lambda b: (b.est_levels is None, b.distance_m))
    return out


def save(buildings: list[Building], path: Path) -> None:
    path.write_text(json.dumps([b.to_dict() for b in buildings], ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> list[Building]:
    return [Building.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------- Overpass

def overpass_query(lat: float, lon: float, radius_m: float) -> str:
    r = int(math.ceil(radius_m))
    return (
        "[out:json][timeout:90];\n"
        "(\n"
        f'  way["building"](around:{r},{lat},{lon});\n'
        f'  relation["building"](around:{r},{lat},{lon});\n'
        f'  node["addr:housenumber"](around:{r + 30},{lat},{lon});\n'
        ");\n"
        "out geom;\n"
    )


def parse_overpass(data: dict) -> tuple[list[Building], list[AddressPoint]]:
    buildings, points = [], []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = el.get("type")
        if kind == "node":
            text = _address_text(tags.get("addr:street") or tags.get("addr:place"), tags.get("addr:housenumber"))
            if text and "lat" in el:
                points.append(AddressPoint(el["lat"], el["lon"], text))
            continue
        poly = None
        if kind == "way" and len(el.get("geometry", [])) >= 4:
            ring = [(p["lon"], p["lat"]) for p in el["geometry"]]
            poly = Polygon(ring) if ring[0] == ring[-1] else None
        elif kind == "relation":
            outers = [
                LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                for m in el.get("members", [])
                if m.get("role") == "outer" and len(m.get("geometry", [])) >= 2
            ]
            if outers:
                poly = _largest_polygon(unary_union(list(polygonize(linemerge(outers)))))
        if poly is None or not poly.is_valid or poly.area == 0:
            if poly is not None:
                poly = _largest_polygon(poly.buffer(0))
            if poly is None:
                continue
        b = Building(
            osm_id=f"{kind}/{el['id']}",
            polygon=poly,
            levels=_parse_int(tags.get("building:levels")),
            height_m=_parse_float(tags.get("height")),
            building_type=tags.get("building"),
            name=tags.get("name"),
            source="overpass",
        )
        text = _address_text(tags.get("addr:street"), tags.get("addr:housenumber"))
        if text:
            b.addresses.append(text)
        buildings.append(b)
    return buildings, points


def load_overpass(lat: float, lon: float, radius_m: float, endpoint: str | None = None) -> list[Building]:
    query = overpass_query(lat, lon, radius_m)
    last_error: Exception | None = None
    for url in ([endpoint] if endpoint else OVERPASS_ENDPOINTS):
        try:
            r = http.session().post(url, data={"data": query}, timeout=120)
            r.raise_for_status()
            buildings, points = parse_overpass(r.json())
            assign_addresses(buildings, points)
            return prepare(buildings, lat, lon)
        except Exception as e:  # try the next mirror
            last_error = e
    raise RuntimeError(f"Overpass nicht erreichbar: {last_error}")


# --------------------------------------------------------------------------- Overture

def _s3_list(prefix: str, delimiter: str | None = None) -> tuple[list[tuple[str, int]], list[str]]:
    keys, prefixes, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        text = http.get(OVERTURE_BUCKET, params=params).text
        keys += list(zip(re.findall(r"<Key>([^<]+)</Key>", text), map(int, re.findall(r"<Size>(\d+)</Size>", text))))
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not m:
            return keys, prefixes
        token = m.group(1)


def latest_overture_release() -> str:
    _, prefixes = _s3_list("release/", "/")
    releases = sorted(p.split("/")[1] for p in prefixes if re.match(r"release/\d{4}-\d{2}-\d{2}", p))
    if not releases:
        raise RuntimeError("Keine Overture-Releases gefunden")
    return releases[-1]


def _row_groups_in_bbox(key: str, bbox: tuple[float, float, float, float]) -> list[int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    url = OVERTURE_BUCKET + key
    tail = http.get(url, headers={"Range": "bytes=-8"}).content
    footer_len = struct.unpack("<I", tail[:4])[0]
    footer = http.get(url, headers={"Range": f"bytes=-{footer_len + 8}"}).content
    md = pq.read_metadata(pa.BufferReader(b"PAR1" + footer))
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    idx = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
    xmin, ymin, xmax, ymax = bbox
    hits = []
    for rg in range(md.num_row_groups):
        st = {k: md.row_group(rg).column(i).statistics for k, i in idx.items()}
        if any(s is None or not s.has_min_max for s in st.values()):
            hits.append(rg)
            continue
        if st["bbox.xmin"].min <= xmax and st["bbox.xmax"].max >= xmin and st["bbox.ymin"].min <= ymax and st["bbox.ymax"].max >= ymin:
            hits.append(rg)
    return hits


def _overture_rows(release: str, theme: str, kind: str, bbox, columns, cache_dir: Path) -> list[dict]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    prefix = f"release/{release}/theme={theme}/type={kind}/"
    cache = cache_dir / f"overture_{release}_{theme}_{kind}_{'_'.join(f'{v:.4f}' for v in bbox)}.json"
    if cache.exists():
        hits = json.loads(cache.read_text())
    else:
        files = [(k, s) for k, s in _s3_list(prefix)[0] if k.endswith(".parquet")]
        with cf.ThreadPoolExecutor(24) as ex:
            found = list(ex.map(lambda f: (f, _row_groups_in_bbox(f[0], bbox)), files))
        hits = [{"key": k, "size": s, "row_groups": rgs} for (k, s), rgs in found if rgs]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hits))
    xmin, ymin, xmax, ymax = bbox
    rows: list[dict] = []
    for h in hits:
        f = pq.ParquetFile(pa.PythonFile(http.HTTPRangeFile(OVERTURE_BUCKET + h["key"], h["size"]), mode="r"))
        for rg in h["row_groups"]:
            t = f.read_row_group(rg, columns=columns)
            b = t.column("bbox")
            cx = pc.divide(pc.add(pc.struct_field(b, "xmin"), pc.struct_field(b, "xmax")), 2)
            cy = pc.divide(pc.add(pc.struct_field(b, "ymin"), pc.struct_field(b, "ymax")), 2)
            mask = pc.and_(
                pc.and_(pc.greater_equal(cx, xmin), pc.less_equal(cx, xmax)),
                pc.and_(pc.greater_equal(cy, ymin), pc.less_equal(cy, ymax)),
            )
            rows += t.filter(mask).to_pylist()
    return rows


def _overture_osm_id(sources) -> str | None:
    for s in sources or []:
        rid = s.get("record_id") or ""
        m = re.match(r"([nwr])(\d+)", rid)
        if s.get("dataset") == "OpenStreetMap" and m:
            return {"n": "node", "w": "way", "r": "relation"}[m.group(1)] + "/" + m.group(2)
    return None


def load_overture(lat: float, lon: float, radius_m: float, cache_dir: Path, release: str | None = None) -> list[Building]:
    release = release or latest_overture_release()
    bbox = bbox_around(lat, lon, radius_m + 60)
    rows = _overture_rows(
        release, "buildings", "building", bbox,
        ["id", "geometry", "bbox", "height", "num_floors", "names", "class", "subtype", "sources"], cache_dir,
    )
    buildings = []
    for r in rows:
        poly = _largest_polygon(wkb.loads(r["geometry"]))
        if poly is None:
            continue
        names = r.get("names") or {}
        buildings.append(Building(
            osm_id=_overture_osm_id(r.get("sources")) or f"overture/{r['id']}",
            polygon=poly,
            levels=r.get("num_floors"),
            height_m=r.get("height"),
            building_type=r.get("class") or r.get("subtype"),
            name=names.get("primary") if isinstance(names, dict) else None,
            source=f"overture {release}",
        ))
    addr_rows = _overture_rows(
        release, "addresses", "address", bbox, ["street", "number", "unit", "geometry", "bbox"], cache_dir,
    )
    points = []
    for a in addr_rows:
        text = _address_text(a.get("street"), a.get("number"), a.get("unit"))
        if text:
            p = wkb.loads(a["geometry"])
            points.append(AddressPoint(p.y, p.x, text))
    assign_addresses(buildings, points)
    return prepare(buildings, lat, lon)


def load_buildings(lat: float, lon: float, radius_m: float, source: str, cache_dir: Path,
                   log=lambda msg: None) -> tuple[list[Building], str]:
    """Returns (buildings, source actually used). `auto` tries Overpass first, then Overture."""
    if source == "overpass":
        return load_overpass(lat, lon, radius_m + 60), "overpass"
    if source == "overture":
        return load_overture(lat, lon, radius_m, cache_dir), "overture"
    if source == "auto":
        try:
            return load_overpass(lat, lon, radius_m + 60), "overpass"
        except RuntimeError as e:
            log(f"{e} – weiche auf Overture Maps aus (dauert etwas länger) …")
            return load_overture(lat, lon, radius_m, cache_dir), "overture"
    raise ValueError(f"Unbekannte Quelle: {source}")
