"""Compare the listing photo with street-level images.

Local features (RootSIFT) are matched with Lowe's ratio test and verified with
a RANSAC homography: a facade is close to planar, so the number of matches
that agree on one homography is a robust "same building" score. Listing photos
are often shot steeply upwards, so the query is additionally re-rendered as if
the camera had been tilted down (pure rotation homography), which brings it
closer to the street-level viewpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# below this many homography inliers a pair is treated as "no geometric match"
MIN_INLIERS = 20


@dataclass
class Features:
    keypoints: np.ndarray        # (N, 2) float32 pixel coordinates
    descriptors: np.ndarray      # (N, 128) float32 RootSIFT
    image: np.ndarray            # BGR image the features were computed on
    color_hist: np.ndarray       # normalised HS histogram without sky
    label: str = ""


@dataclass
class MatchResult:
    inliers: int = 0
    good_matches: int = 0
    color_similarity: float = 0.0
    query_label: str = ""
    query_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    image_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))

    @property
    def score(self) -> float:
        """Inliers dominate; the colour term only orders otherwise equal results."""
        geometric = self.inliers if self.inliers >= MIN_INLIERS else self.inliers * 0.25
        return geometric + 5.0 * self.color_similarity


def read_image(path: str, max_side: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Bild nicht lesbar: {path}")
    return resize_max(img, max_side) if max_side else img


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def sky_mask(img: np.ndarray) -> np.ndarray:
    """True where a pixel looks like blue sky or bright cloud."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 90) & (h <= 130) & (s >= 50) & (v >= 100)
    cloud = (s < 25) & (v > 225)
    return blue | cloud


def color_histogram(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (~sky_mask(img)).astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)


def color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(max(0.0, 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)))


def rotation_homography(width: int, height: int, tilt_down_deg: float, focal_px: float | None = None) -> np.ndarray:
    """Homography that re-renders an image as if the camera were tilted down by `tilt_down_deg`."""
    f = focal_px or 0.75 * max(width, height)
    k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    t = math.radians(tilt_down_deg)
    r = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return k @ r @ np.linalg.inv(k)


def warp_tilt(img: np.ndarray, tilt_down_deg: float, max_side: int = 1024) -> np.ndarray:
    if abs(tilt_down_deg) < 1e-6:
        return resize_max(img, max_side)
    h, w = img.shape[:2]
    hom = rotation_homography(w, h, tilt_down_deg)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, hom).reshape(-1, 2)
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    scale = min(1.0, max_side / max(x1 - x0, y1 - y0))
    fit = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    size = (max(1, int(math.ceil((x1 - x0) * scale))), max(1, int(math.ceil((y1 - y0) * scale))))
    return cv2.warpPerspective(img, fit @ hom, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


class FacadeMatcher:
    def __init__(self, max_side: int = 1024, n_features: int = 5000, ratio: float = 0.8, ransac_px: float = 6.0):
        self.max_side = max_side
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, img: np.ndarray, label: str = "") -> Features:
        img = resize_max(img, self.max_side)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        # ignore black borders introduced by warping
        mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        kps, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None or len(kps) == 0:
            pts, desc = np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
        else:
            pts = np.array([k.pt for k in kps], np.float32)
            desc = desc.astype(np.float32)
            desc /= desc.sum(axis=1, keepdims=True) + 1e-7   # RootSIFT
            desc = np.sqrt(desc)
        return Features(pts, desc, img, color_histogram(img), label)

    def query_features(self, img: np.ndarray, tilts=(0.0, 15.0, 30.0), name: str = "Foto") -> list[Features]:
        return [self.features(warp_tilt(img, t, self.max_side), f"{name}, " + (f"virtuell {t:.0f}° gesenkt" if t else "Original"))
                for t in tilts]

    def match(self, q: Features, c: Features) -> MatchResult:
        result = MatchResult(color_similarity=color_similarity(q.color_hist, c.color_hist), query_label=q.label)
        if len(q.descriptors) < 8 or len(c.descriptors) < 8:
            return result
        pairs = self.bf.knnMatch(q.descriptors, c.descriptors, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        # one-to-one: keep the best match per image keypoint
        best: dict[int, cv2.DMatch] = {}
        for m in good:
            if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                best[m.trainIdx] = m
        good = list(best.values())
        result.good_matches = len(good)
        if len(good) < 8:
            return result
        src = q.keypoints[[m.queryIdx for m in good]]
        dst = c.keypoints[[m.trainIdx for m in good]]
        hom, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, self.ransac_px, maxIters=5000, confidence=0.999)
        if hom is None or mask is None:
            return result
        inl = mask.ravel().astype(bool)
        if not _plausible(hom, src[inl], q.image.shape, dst[inl], c.image.shape):
            return result
        result.inliers = int(inl.sum())
        result.query_points, result.image_points = src[inl], dst[inl]
        return result

    def best_match(self, queries: list[Features], image: np.ndarray) -> MatchResult:
        c = self.features(image)
        results = [self.match(q, c) for q in queries]
        return max(results, key=lambda r: r.score) if results else MatchResult()


def _plausible(hom: np.ndarray, src: np.ndarray, src_shape, dst: np.ndarray, dst_shape) -> bool:
    """Reject degenerate homographies and inliers clustered on a tiny patch."""
    if len(src) < 4:
        return False
    det = np.linalg.det(hom[:2, :2])
    if not 1e-3 < abs(det) < 1e3:
        return False
    for pts, shape in ((src, src_shape), (dst, dst_shape)):
        hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.float32))
        if cv2.contourArea(hull) < 0.01 * shape[0] * shape[1]:
            return False
    return True


def draw_matches(q: np.ndarray, c: np.ndarray, result: MatchResult, max_lines: int = 80, height: int = 360) -> np.ndarray:
    """Side-by-side picture with the verified correspondences."""
    def fit(img):
        s = height / img.shape[0]
        return cv2.resize(img, (max(1, round(img.shape[1] * s)), height)), s

    a, sa = fit(q)
    b, sb = fit(c)
    canvas = np.zeros((height, a.shape[1] + b.shape[1] + 10, 3), np.uint8)
    canvas[:, : a.shape[1]] = a
    canvas[:, a.shape[1] + 10:] = b
    rng = np.random.default_rng(0)
    idx = np.arange(len(result.query_points))
    if len(idx) > max_lines:
        idx = rng.choice(idx, max_lines, replace=False)
    for i in idx:
        p = tuple(int(v) for v in result.query_points[i] * sa)
        r = result.image_points[i] * sb
        p2 = (int(r[0]) + a.shape[1] + 10, int(r[1]))
        color = tuple(int(v) for v in rng.integers(64, 256, 3))
        cv2.line(canvas, p, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 2, color, -1)
        cv2.circle(canvas, p2, 2, color, -1)
    return canvas
