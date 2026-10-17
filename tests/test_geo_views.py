import math

from shapely.geometry import Polygon

from hauserkennen.buildings import Building
from hauserkennen.geo import LocalProjection, angle_diff_deg, bearing_deg, destination, haversine_m
from hauserkennen.views import facades, maps_pano_url, probe_points, street_view_links, view_towards


def test_bearing_cardinal_directions():
    assert bearing_deg(47.0, 15.0, 47.01, 15.0) == 0.0 or math.isclose(bearing_deg(47.0, 15.0, 47.01, 15.0), 360.0)
    assert math.isclose(bearing_deg(47.0, 15.0, 47.0, 15.01), 90.0, abs_tol=0.01)
    assert math.isclose(bearing_deg(47.0, 15.0, 46.99, 15.0), 180.0, abs_tol=0.01)


def test_destination_round_trip():
    lat, lon = destination(47.0843, 15.4594, 37.0, 250.0)
    assert math.isclose(haversine_m(47.0843, 15.4594, lat, lon), 250.0, rel_tol=1e-6)
    assert math.isclose(bearing_deg(47.0843, 15.4594, lat, lon), 37.0, abs_tol=0.01)


def test_local_projection_round_trip_and_scale():
    p = LocalProjection(47.0843, 15.4594)
    x, y = p.to_xy(47.0853, 15.4614)
    lat, lon = p.to_latlon(x, y)
    assert math.isclose(lat, 47.0853, abs_tol=1e-9) and math.isclose(lon, 15.4614, abs_tol=1e-9)
    assert math.isclose(math.hypot(x, y), haversine_m(47.0843, 15.4594, 47.0853, 15.4614), rel_tol=2e-3)


def test_angle_diff_wraps():
    assert angle_diff_deg(350, 10) == 20
    assert angle_diff_deg(10, 350) == 20
    assert angle_diff_deg(90, 270) == 180


def _square_building(size_m=20.0, lat=47.0843, lon=15.4594):
    p = LocalProjection(lat, lon)
    h = size_m / 2
    ring = [p.to_latlon(x, y)[::-1] for x, y in ((-h, -h), (h, -h), (h, h), (-h, h), (-h, -h))]
    return Building("way/1", Polygon(ring), levels=6), p


def test_facades_have_outward_normals():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    for f in facades(poly):
        probe = (f.mid[0] + f.normal[0], f.mid[1] + f.normal[1])
        assert not poly.contains(Polygon([(probe[0], probe[1]), (probe[0] + 1e-3, probe[1]), (probe[0], probe[1] + 1e-3)]))
    # clockwise input gives the same normals
    assert sorted((round(f.normal[0]), round(f.normal[1])) for f in facades(Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]))) == \
        sorted((round(f.normal[0]), round(f.normal[1])) for f in facades(poly))


def test_view_from_south_looks_north_and_up():
    poly = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    t = view_towards((0, -30), poly, height_m=18)
    assert math.isclose(t.heading, 0.0, abs_tol=0.5) or math.isclose(t.heading, 360.0, abs_tol=0.5)
    assert math.isclose(t.distance_m, 20.0, abs_tol=0.01)
    assert 5 <= t.pitch <= 50


def test_probe_points_and_links():
    b, proj = _square_building()
    pts = probe_points(b, proj, offset_m=15)
    assert len(pts) == 4
    for lat, lon in pts:
        assert math.isclose(haversine_m(lat, lon, *b.centroid), 25.0, abs_tol=0.5)
    links = street_view_links(b, proj)
    assert len(links) == 4 and all(l["url"].startswith("https://www.google.com/maps/@?api=1&map_action=pano") for l in links)
    # every link looks back at the building
    for l in links:
        assert angle_diff_deg(l["heading"], bearing_deg(l["lat"], l["lon"], *b.centroid)) < 1.0


def test_maps_url_with_pano():
    url = maps_pano_url(47.1, 15.4, 123.4, 20, 90, pano_id="abc")
    assert "pano=abc" in url and "heading=123" in url and "viewpoint" not in url
