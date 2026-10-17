"""Where to stand and where to look to see a building's facades."""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlencode

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from .buildings import Building
from .geo import LocalProjection

CAMERA_HEIGHT_M = 2.5


@dataclass
class Facade:
    mid: tuple[float, float]       # local metres
    normal: tuple[float, float]    # unit vector pointing away from the building
    length: float


@dataclass
class ViewTarget:
    heading: float      # compass degrees, 0 = north
    pitch: float        # degrees above the horizon
    distance_m: float   # camera to nearest point of the footprint


def facades(poly_xy: Polygon, min_len: float = 8.0, simplify_m: float = 1.0) -> list[Facade]:
    """Footprint edges long enough to be a facade, with outward normals."""
    poly = poly_xy.simplify(simplify_m, preserve_topology=True)
    ring = poly.exterior
    coords = list(ring.coords)
    if not ring.is_ccw:
        coords.reverse()
    out = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        # counter-clockwise ring: the interior is on the left, so the right-hand normal points outward
        out.append(Facade(((x1 + x2) / 2, (y1 + y2) / 2), (dy / length, -dx / length), length))
    return out


def probe_points(building: Building, proj: LocalProjection, offset_m: float = 15.0,
                 min_spacing_m: float = 10.0) -> list[tuple[float, float]]:
    """Points in front of each facade (lat, lon) where street-level imagery is searched."""
    poly = building.projected(proj)
    facs = facades(poly)
    if not facs:  # tiny or odd footprint: look from four sides
        c = poly.centroid
        r = math.sqrt(max(poly.area, 1.0) / math.pi) + offset_m
        pts = [(c.x + r * dx, c.y + r * dy) for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))]
    else:
        pts = [(f.mid[0] + f.normal[0] * offset_m, f.mid[1] + f.normal[1] * offset_m) for f in facs]
    kept: list[tuple[float, float]] = []
    for p in pts:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_spacing_m for q in kept):
            kept.append(p)
    return [proj.to_latlon(x, y) for x, y in kept]


def view_towards(viewer_xy: tuple[float, float], poly_xy: Polygon, height_m: float | None,
                 aim_fraction: float = 0.55) -> ViewTarget:
    """Heading and pitch from the camera to the nearest part of the building."""
    viewer = Point(viewer_xy)
    _, target = nearest_points(viewer, poly_xy.exterior)
    dx, dy = target.x - viewer.x, target.y - viewer.y
    dist = max(math.hypot(dx, dy), 1.0)
    heading = math.degrees(math.atan2(dx, dy)) % 360.0
    aim_z = min((height_m or 15.0) * aim_fraction, 20.0)
    pitch = math.degrees(math.atan2(aim_z - CAMERA_HEIGHT_M, dist))
    return ViewTarget(heading, max(5.0, min(pitch, 50.0)), dist)


def maps_pano_url(lat: float, lon: float, heading: float, pitch: float = 15.0, fov: float = 90.0,
                  pano_id: str | None = None) -> str:
    """Google Maps link that opens Street View at this position (no API key needed)."""
    params = {"api": "1", "map_action": "pano", "heading": round(heading), "pitch": round(pitch), "fov": round(fov)}
    if pano_id:
        params["pano"] = pano_id
    else:
        params["viewpoint"] = f"{lat:.6f},{lon:.6f}"
    return "https://www.google.com/maps/@?" + urlencode(params)


def street_view_links(building: Building, proj: LocalProjection, offset_m: float = 15.0) -> list[dict]:
    """One Street View link per facade, looking at the facade from the street side."""
    poly = building.projected(proj)
    links = []
    for lat, lon in probe_points(building, proj, offset_m=offset_m):
        t = view_towards(proj.to_xy(lat, lon), poly, building.est_height_m)
        links.append({"lat": lat, "lon": lon, "heading": round(t.heading), "pitch": round(t.pitch),
                      "url": maps_pano_url(lat, lon, t.heading, t.pitch)})
    return links
