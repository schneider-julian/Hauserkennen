"""Street-level imagery around a candidate building.

* Google Street View Static API (needs an API key with the Street View Static
  API enabled; metadata requests are free, image requests are billed).
* Mapillary Graph API v4 (free access token, imagery under CC BY-SA 4.0).

Every download is cached on disk, so re-running the search costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import Point

from . import http
from .buildings import Building
from .geo import LocalProjection, angle_diff_deg, bbox_around
from .views import maps_pano_url, probe_points, view_towards

SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"
MAPILLARY_FIELDS = ",".join([
    "id", "computed_geometry", "geometry", "computed_compass_angle", "compass_angle",
    "is_pano", "thumb_1024_url", "thumb_2048_url", "captured_at",
])


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class StreetImage:
    path: str
    provider: str
    building_id: str
    source_id: str          # Street View pano id or Mapillary image id
    lat: float              # camera position
    lon: float
    heading: float
    pitch: float
    fov: float
    distance_m: float
    captured: str | None
    url: str                # where a human can look at the same view
    attribution: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- Google

class StreetViewClient:
    def __init__(self, api_key: str, cache_dir: Path, max_images: int = 300, size: int = 640):
        if not api_key:
            raise ValueError("Google-API-Key fehlt (Umgebungsvariable GOOGLE_MAPS_API_KEY oder --google-key)")
        self.key = api_key
        self.cache = Path(cache_dir) / "streetview"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.size = size
        self.downloaded = 0

    def metadata(self, lat: float, lon: float, radius: int = 35) -> dict | None:
        """Nearest outdoor panorama within `radius` metres, or None. Free of charge."""
        f = self.cache / f"meta_{lat:.6f}_{lon:.6f}_{radius}.json"
        if f.exists():
            data = json.loads(f.read_text())
        else:
            data = http.get(SV_METADATA_URL, params={
                "location": f"{lat:.6f},{lon:.6f}", "radius": radius, "source": "outdoor", "key": self.key,
            }).json()
            if data.get("status") in ("OK", "ZERO_RESULTS", "NOT_FOUND"):
                f.write_text(json.dumps(data))
        status = data.get("status")
        if status == "OK":
            return data
        if status in ("ZERO_RESULTS", "NOT_FOUND"):
            return None
        raise RuntimeError(f"Street View Metadata: {status} {data.get('error_message', '')}".strip())

    def image(self, pano_id: str, heading: float, pitch: float, fov: float) -> Path:
        name = f"sv_{pano_id}_{heading:.0f}_{pitch:.0f}_{fov:.0f}_{self.size}.jpg"
        f = self.cache / name
        if f.exists():
            return f
        if self.downloaded >= self.max_images:
            raise BudgetExceeded(f"Bild-Budget von {self.max_images} Street-View-Bildern erreicht")
        r = http.session().get(SV_IMAGE_URL, params={
            "size": f"{self.size}x{self.size}", "pano": pano_id, "heading": f"{heading:.1f}",
            "pitch": f"{pitch:.1f}", "fov": f"{fov:.0f}", "return_error_code": "true", "key": self.key,
        }, timeout=60)
        r.raise_for_status()
        self.downloaded += 1
        f.write_bytes(r.content)
        return f


def collect_google(building: Building, client: StreetViewClient, max_panos: int = 4,
                   extra_pitch: float = 20.0, fov: float = 90.0, max_dist_m: float = 70.0) -> list[StreetImage]:
    """Panoramas in front of each facade, each looked at twice: at mid height and higher up."""
    lat0, lon0 = building.centroid
    proj = LocalProjection(lat0, lon0)
    poly = building.projected(proj)
    panos: dict[str, dict] = {}
    for lat, lon in probe_points(building, proj):
        meta = client.metadata(lat, lon)
        if meta and meta["pano_id"] not in panos:
            panos[meta["pano_id"]] = meta
    views = []
    for meta in panos.values():
        loc = meta["location"]
        xy = proj.to_xy(loc["lat"], loc["lng"])
        if poly.contains(Point(xy)):
            continue
        t = view_towards(xy, poly, building.est_height_m)
        if t.distance_m <= max_dist_m:
            views.append((t, meta))
    views.sort(key=lambda v: v[0].distance_m)
    images = []
    for t, meta in views[:max_panos]:
        for pitch in (t.pitch, min(t.pitch + extra_pitch, 60.0)):
            try:
                path = client.image(meta["pano_id"], t.heading, pitch, fov)
            except requests.HTTPError:  # e.g. 404 when the panorama has no imagery for this view
                continue
            images.append(StreetImage(
                path=str(path), provider="google", building_id=building.osm_id, source_id=meta["pano_id"],
                lat=meta["location"]["lat"], lon=meta["location"]["lng"], heading=t.heading, pitch=pitch, fov=fov,
                distance_m=t.distance_m, captured=meta.get("date"),
                url=maps_pano_url(meta["location"]["lat"], meta["location"]["lng"], t.heading, pitch, fov, meta["pano_id"]),
                attribution=meta.get("copyright") or "© Google",
            ))
    return images


# --------------------------------------------------------------------------- Mapillary

class MapillaryClient:
    def __init__(self, token: str, cache_dir: Path, max_images: int = 300):
        if not token:
            raise ValueError("Mapillary-Token fehlt (Umgebungsvariable MAPILLARY_TOKEN oder --mapillary-token)")
        self.token = token
        self.cache = Path(cache_dir) / "mapillary"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.downloaded = 0

    def images_in_bbox(self, bbox: tuple[float, float, float, float]) -> list[dict]:
        key = hashlib.sha1(",".join(f"{v:.6f}" for v in bbox).encode()).hexdigest()[:16]
        f = self.cache / f"search_{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        data = http.get(MAPILLARY_IMAGES_URL, params={
            "fields": MAPILLARY_FIELDS, "bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": 2000,
        }, headers={"Authorization": f"OAuth {self.token}"}).json().get("data", [])
        f.write_text(json.dumps(data))
        return data

    def download(self, url: str, name: str) -> Path:
        f = self.cache / name
        if f.exists():
            return f
        if self.downloaded >= self.max_images:
            raise BudgetExceeded(f"Bild-Budget von {self.max_images} Mapillary-Bildern erreicht")
        f.write_bytes(http.get(url, timeout=120).content)
        self.downloaded += 1
        return f


def equirect_to_perspective(pano: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float,
                            out_w: int = 800, out_h: int = 800) -> np.ndarray:
    """Cut a pinhole view out of an equirectangular panorama.

    `yaw_deg` is measured clockwise from the panorama's centre column,
    `pitch_deg` upwards from the horizon.
    """
    import cv2

    h, w = pano.shape[:2]
    f = (out_w / 2) / math.tan(math.radians(fov_deg) / 2)
    xs, ys = np.meshgrid(np.arange(out_w) - out_w / 2 + 0.5, np.arange(out_h) - out_h / 2 + 0.5)
    rays = np.stack([xs, ys, np.full_like(xs, f)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    p, yw = math.radians(pitch_deg), math.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(yw), 0, math.sin(yw)], [0, 1, 0], [-math.sin(yw), 0, math.cos(yw)]])
    world = rays @ (ry @ rx).T
    lon = np.arctan2(world[..., 0], world[..., 2])
    lat = np.arcsin(np.clip(-world[..., 1], -1, 1))
    map_x = ((lon / (2 * math.pi) + 0.5) * w).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * h).astype(np.float32)
    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def collect_mapillary(building: Building, client: MapillaryClient, max_images: int = 6,
                      max_dist_m: float = 80.0, max_angle: float = 35.0, fov: float = 90.0) -> list[StreetImage]:
    """Mapillary photos taken near the building and pointing at it (panoramas are cropped)."""
    lat0, lon0 = building.centroid
    proj = LocalProjection(lat0, lon0)
    poly = building.projected(proj)
    reach = max_dist_m + math.sqrt(max(poly.area, 1.0))
    picks = []
    for im in client.images_in_bbox(bbox_around(lat0, lon0, reach)):
        geom = im.get("computed_geometry") or im.get("geometry")
        if not geom:
            continue
        lon, lat = geom["coordinates"][:2]
        xy = proj.to_xy(lat, lon)
        if poly.contains(Point(xy)):
            continue
        t = view_towards(xy, poly, building.est_height_m)
        if not 4.0 <= t.distance_m <= max_dist_m:
            continue
        compass = im.get("computed_compass_angle", im.get("compass_angle"))
        if compass is None:
            continue
        diff = 0.0 if im.get("is_pano") else angle_diff_deg(compass, t.heading)
        if diff > max_angle:
            continue
        picks.append((diff + t.distance_m / 4.0, im, t, lat, lon, compass))
    picks.sort(key=lambda p: p[0])
    images = []
    for _, im, t, lat, lon, compass in picks[:max_images]:
        try:
            image = _mapillary_view(client, im, t, compass, fov)
        except requests.RequestException:  # expired thumbnail URL or network hiccup: skip this photo
            continue
        if image is None:
            continue
        path, heading, pitch = image
        captured = im.get("captured_at")
        if isinstance(captured, (int, float)):  # milliseconds since epoch
            captured = datetime.fromtimestamp(captured / 1000, tz=timezone.utc).date().isoformat()
        images.append(StreetImage(
            path=str(path), provider="mapillary", building_id=building.osm_id, source_id=str(im["id"]),
            lat=lat, lon=lon, heading=heading, pitch=pitch, fov=fov, distance_m=t.distance_m,
            captured=str(captured) if captured else None,
            url=f"https://www.mapillary.com/app/?pKey={im['id']}&focus=photo",
            attribution="© Mapillary-Mitwirkende, CC BY-SA 4.0",
        ))
    return images


def _mapillary_view(client: MapillaryClient, im: dict, t, compass: float, fov: float):
    """(path, heading, pitch) of the image to compare; panoramas are cropped towards the building."""
    import cv2

    if im.get("is_pano") and im.get("thumb_2048_url"):
        raw = client.download(im["thumb_2048_url"], f"mly_{im['id']}_pano.jpg")
        crop_path = client.cache / f"mly_{im['id']}_{t.heading:.0f}_{t.pitch:.0f}.jpg"
        if not crop_path.exists():
            pano = cv2.imread(str(raw))
            if pano is None:
                return None
            cv2.imwrite(str(crop_path), equirect_to_perspective(pano, t.heading - compass, t.pitch, fov))
        return crop_path, t.heading, t.pitch
    if im.get("thumb_1024_url"):
        return client.download(im["thumb_1024_url"], f"mly_{im['id']}.jpg"), compass, 0.0
    return None
