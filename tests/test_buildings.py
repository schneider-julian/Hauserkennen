from hauserkennen.buildings import assign_addresses, filter_candidates, parse_overpass, prepare
from hauserkennen.pipeline import address_matches, is_excluded

LAT, LON = 47.0843, 15.4594
D = 0.0002  # ~15 m


def _ring(lat, lon, d=D):
    return [{"lat": lat - d, "lon": lon - d}, {"lat": lat - d, "lon": lon + d}, {"lat": lat + d, "lon": lon + d},
            {"lat": lat + d, "lon": lon - d}, {"lat": lat - d, "lon": lon - d}]


OVERPASS = {"elements": [
    {"type": "way", "id": 1, "tags": {"building": "apartments", "building:levels": "6"}, "geometry": _ring(LAT, LON)},
    {"type": "way", "id": 2, "tags": {"building": "house", "building:levels": "2"}, "geometry": _ring(LAT + 0.001, LON)},
    {"type": "way", "id": 3, "tags": {"building": "yes"}, "geometry": _ring(LAT - 0.001, LON, 0.0003)},
    {"type": "way", "id": 4, "tags": {"building": "garage"}, "geometry": _ring(LAT, LON + 0.001, 0.0003)},
    {"type": "way", "id": 5, "tags": {"building": "apartments", "height": "21.5"}, "geometry": _ring(LAT + 0.01, LON)},
    {"type": "relation", "id": 6, "tags": {"building": "yes", "building:levels": "7"}, "members": [
        {"type": "way", "role": "outer", "geometry": _ring(LAT, LON - 0.001)[:3]},
        {"type": "way", "role": "outer", "geometry": _ring(LAT, LON - 0.001)[2:]},
    ]},
    {"type": "node", "id": 10, "lat": LAT + 0.00005, "lon": LON, "tags": {"addr:street": "Heinrichstraße", "addr:housenumber": "131"}},
    {"type": "node", "id": 11, "lat": LAT - 0.00005, "lon": LON, "tags": {"addr:street": "Heinrichstraße", "addr:housenumber": "133"}},
]}


def _load():
    buildings, points = parse_overpass(OVERPASS)
    assign_addresses(buildings, points)
    return prepare(buildings, LAT, LON)


def test_parse_overpass_ways_relations_and_addresses():
    bs = {b.osm_id: b for b in _load()}
    assert set(bs) == {"way/1", "way/2", "way/3", "way/4", "way/5", "relation/6"}
    assert bs["way/1"].levels == 6 and bs["way/1"].addresses == ["Heinrichstraße 131", "Heinrichstraße 133"]
    assert bs["relation/6"].levels == 7 and bs["relation/6"].area_m2 > 500
    assert bs["way/5"].est_levels == 7  # 21.5 m / 3 m
    assert bs["way/1"].distance_m == 0.0


def test_filter_candidates():
    bs = _load()
    ids = [b.osm_id for b in filter_candidates(bs, radius_m=300, min_levels=5, min_area_unknown_m2=200)]
    # 6- and 7-storey buildings first, then the large one without levels; garage, 2-storey house and far one dropped
    assert ids == ["way/1", "relation/6", "way/3"]
    assert [b.osm_id for b in filter_candidates(bs, 300, min_levels=5, include_unknown=False)] == ["way/1", "relation/6"]
    assert [b.osm_id for b in filter_candidates(bs, 300, min_levels=5, max_levels=6, include_unknown=False)] == ["way/1"]


def test_address_matching_and_exclusion():
    assert address_matches("Heinrichstraße 131", "Heinrichstraße 131")
    assert address_matches("Heinrichstr. 131", "Heinrichstraße 131/4")
    assert address_matches("heinrichstrasse 131a", "Heinrichstraße 131a")
    assert not address_matches("Heinrichstraße 13", "Heinrichstraße 131")
    assert not address_matches("Heinrichstraße 131", "Heinrichstraße 131a")
    assert address_matches("Heinrichstraße", "Heinrichstraße 7")
    b = {x.osm_id: x for x in _load()}["way/1"]
    assert is_excluded(b, ["Heinrichstraße 131"]) and is_excluded(b, ["way/1"])
    assert not is_excluded(b, ["Heinrichstraße 13", "way/12"])
