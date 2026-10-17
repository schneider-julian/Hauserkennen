"""Offline tests of image collection and the full search pipeline with fake imagery providers."""

import math

import cv2
import numpy as np
from shapely.geometry import Polygon

from hauserkennen import pipeline
from hauserkennen.buildings import Building, prepare
from hauserkennen.geo import LocalProjection, angle_diff_deg, destination
from hauserkennen.imagery import StreetImage, collect_google, collect_mapillary
from test_matching import _viewpoint_change, synthetic_facade

LAT, LON = 47.0843, 15.4594


def _building(osm_id: str, east_m: float, size_m: float = 24.0, levels: int = 6) -> Building:
    p = LocalProjection(LAT, LON)
    h = size_m / 2
    ring = [p.to_latlon(east_m + x, y)[::-1] for x, y in ((-h, -h), (h, -h), (h, h), (-h, h), (-h, -h))]
    return Building(osm_id, Polygon(ring), levels=levels, addresses=[f"Teststraße {osm_id[-1]}"])


class FakeStreetView:
    """Every probe finds one panorama 20 m south of the building."""

    def __init__(self, building: Building, tmp_path):
        self.lat, self.lon = destination(*building.centroid, 180.0, 32.0)
        self.tmp = tmp_path
        self.requests = []

    def metadata(self, lat, lon, radius=35):
        return {"status": "OK", "pano_id": "PANO1", "location": {"lat": self.lat, "lng": self.lon}, "date": "2024-05"}

    def image(self, pano_id, heading, pitch, fov):
        self.requests.append((heading, pitch))
        path = self.tmp / f"{pano_id}_{len(self.requests)}.jpg"
        cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
        return path


def test_collect_google_looks_at_building_from_panorama(tmp_path):
    b = prepare([_building("way/1", 0)], LAT, LON)[0]
    client = FakeStreetView(b, tmp_path)
    images = collect_google(b, client)
    assert len(images) == 2  # one panorama, two pitches
    assert all(angle_diff_deg(i.heading, 0.0) < 1.0 for i in images)  # panorama is south -> look north
    assert images[1].pitch > images[0].pitch
    assert math.isclose(images[0].distance_m, 20.0, abs_tol=0.5)
    assert "pano=PANO1" in images[0].url


class FakeMapillary:
    def __init__(self, building: Building, tmp_path):
        self.cache = tmp_path
        c = building.centroid
        south = destination(*c, 180.0, 40.0)
        self.items = [
            {"id": 1, "computed_geometry": {"coordinates": [south[1], south[0]]}, "computed_compass_angle": 2.0,
             "is_pano": False, "thumb_1024_url": "u1", "captured_at": 1700000000000},
            {"id": 2, "computed_geometry": {"coordinates": [south[1], south[0]]}, "computed_compass_angle": 180.0,
             "is_pano": False, "thumb_1024_url": "u2"},  # looks away from the building
            {"id": 3, "computed_geometry": {"coordinates": [south[1], south[0]]}, "computed_compass_angle": 90.0,
             "is_pano": True, "thumb_2048_url": "u3"},
        ]

    def images_in_bbox(self, bbox):
        return self.items

    def download(self, url, name):
        path = self.cache / name
        img = np.zeros((256, 512, 3), np.uint8) if url == "u3" else np.full((64, 64, 3), 128, np.uint8)
        cv2.imwrite(str(path), img)
        return path


def test_collect_mapillary_keeps_images_pointing_at_building(tmp_path):
    b = prepare([_building("way/1", 0)], LAT, LON)[0]
    images = collect_mapillary(b, FakeMapillary(b, tmp_path))
    ids = sorted(i.source_id for i in images)
    assert ids == ["1", "3"]
    pano = next(i for i in images if i.source_id == "3")
    assert angle_diff_deg(pano.heading, 0.0) < 1.0 and pano.path.endswith(".jpg")
    assert next(i for i in images if i.source_id == "1").captured == "2023-11-14"


def test_search_ranks_matching_building_first(tmp_path, monkeypatch):
    right, wrong = _building("way/1", 0), _building("way/2", 80)
    cands = prepare([wrong, right], LAT, LON)
    photo = tmp_path / "inserat.jpg"
    cv2.imwrite(str(photo), synthetic_facade(11))
    street = {"way/1": synthetic_facade(11), "way/2": synthetic_facade(12)}

    def fake_collect(b, client):
        path = tmp_path / f"street_{b.osm_id.replace('/', '_')}.jpg"
        cv2.imwrite(str(path), _viewpoint_change(street[b.osm_id]))
        lat, lon = b.centroid
        return [StreetImage(str(path), "google", b.osm_id, "P", lat, lon, 0.0, 15.0, 90.0, 20.0, "2024-05",
                            "https://example.invalid", "© Test")]

    monkeypatch.setattr(pipeline, "collect_google", fake_collect)
    report, ranked, scores = pipeline.search(
        cands, [str(photo)], provider="google", out=tmp_path / "out", cache_dir=tmp_path / "cache",
        lat=LAT, lon=LON, radius_m=200, source="test", google_key="dummy", log=lambda m: None,
    )
    assert [b.osm_id for b in ranked] == ["way/1", "way/2"]
    assert scores["way/1"]["inliers"] > 4 * max(scores["way/2"]["inliers"], 1)
    html = report.read_text(encoding="utf-8")
    assert "Deutlichster Treffer: Teststraße 1" in html and "bilder/way_1_match.jpg" in html
    for name in ("ergebnis.geojson", "ergebnis.csv", "ergebnis_scores.json", "bilder/way_1_match.jpg"):
        assert (tmp_path / "out" / name).exists()
