"""Small geodesy helpers.

Everything here uses local approximations that are accurate to well below a
metre for the distances this tool works with (a few hundred metres).
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees (0 = north, 90 = east)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360.0


def destination(lat: float, lon: float, bearing: float, distance_m: float) -> tuple[float, float]:
    """Point reached when travelling `distance_m` from (lat, lon) along `bearing`."""
    d = distance_m / EARTH_RADIUS_M
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass angles."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


class LocalProjection:
    """Equirectangular projection around an origin: x = metres east, y = metres north."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self._ky = math.pi * EARTH_RADIUS_M / 180.0
        self._kx = self._ky * math.cos(math.radians(lat0))

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return self.lat0 + y / self._ky, self.lon0 + x / self._kx

    def lonlat_array_to_xy(self, coords):
        """Vectorised to_xy for an (N, 2) array of (lon, lat), as used by shapely.transform."""
        out = coords.copy()
        out[:, 0] = (coords[:, 0] - self.lon0) * self._kx
        out[:, 1] = (coords[:, 1] - self.lat0) * self._ky
        return out

    def deg_per_m(self) -> tuple[float, float]:
        """(degrees latitude, degrees longitude) per metre at the origin."""
        return 1.0 / self._ky, 1.0 / self._kx


def bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) of a square that contains the circle."""
    dlat, dlon = LocalProjection(lat, lon).deg_per_m()
    return lon - radius_m * dlon, lat - radius_m * dlat, lon + radius_m * dlon, lat + radius_m * dlat
