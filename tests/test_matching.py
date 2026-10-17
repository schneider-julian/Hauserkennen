import math

import cv2
import numpy as np

from hauserkennen.imagery import equirect_to_perspective
from hauserkennen.matching import MIN_INLIERS, FacadeMatcher, rotation_homography, warp_tilt


def synthetic_facade(seed: int, w: int = 900, h: int = 700) -> np.ndarray:
    """Beige facade with ribbon windows plus unique details (stains, signs) so that two seeds differ."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (170, 205, 225), np.uint8)
    img = cv2.add(img, rng.integers(0, 25, (h, w, 3), dtype=np.uint8))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    floor_h = int(rng.integers(95, 125))
    for y in range(40, h - floor_h, floor_h):
        x = int(rng.integers(20, 60))
        while x < w - 120:
            ww = int(rng.integers(70, 130))
            cv2.rectangle(img, (x, y), (x + ww, y + floor_h // 2), (200, 160, 140), -1)
            cv2.rectangle(img, (x, y), (x + ww, y + floor_h // 2), (250, 250, 250), 4)
            cv2.line(img, (x + ww // 2, y), (x + ww // 2, y + floor_h // 2), (250, 250, 250), 3)
            x += ww + int(rng.integers(10, 60))
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(4, 18)), c, -1)
        else:
            txt = "".join(chr(int(rng.integers(65, 90))) for _ in range(3))
            cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.5, 1.2)), c, 2)
    return img


def _viewpoint_change(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 40], [w - 10, 0], [w - 70, h - 20], [20, h - 60]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    out = cv2.convertScaleAbs(out, alpha=0.85, beta=15)  # different exposure
    return cv2.resize(out, (640, int(640 * h / w)))


def test_same_facade_matches_different_does_not():
    m = FacadeMatcher()
    query = m.query_features(synthetic_facade(1))
    same = m.best_match(query, _viewpoint_change(synthetic_facade(1)))
    assert same.inliers >= 4 * MIN_INLIERS
    for seed in (2, 5, 6, 7, 8):
        other = m.best_match(query, _viewpoint_change(synthetic_facade(seed)))
        assert other.inliers < MIN_INLIERS, seed
        assert same.score > 5 * max(other.score, 1)


def test_upward_tilted_photo_is_recovered_by_tilt_variants():
    m = FacadeMatcher()
    facade = synthetic_facade(3)
    h, w = facade.shape[:2]
    # simulate a photo shot 30 degrees upwards: inverse of the tilt-down homography
    up = cv2.warpPerspective(facade, np.linalg.inv(rotation_homography(w, h, 30)), (w, h))
    plain = m.best_match([m.features(up)], _viewpoint_change(facade))
    with_tilts = m.best_match(m.query_features(up, tilts=(0, 15, 30)), _viewpoint_change(facade))
    assert with_tilts.inliers >= plain.inliers
    assert with_tilts.inliers >= MIN_INLIERS


def test_rotation_homography_identity_and_warp_size():
    assert np.allclose(rotation_homography(640, 480, 0), np.eye(3))
    out = warp_tilt(synthetic_facade(4), 25, max_side=800)
    assert max(out.shape[:2]) <= 800 and out.mean() > 10


def test_equirect_crop_points_at_requested_direction():
    pano = np.zeros((512, 1024, 3), np.uint8)
    # red marker 90 degrees right of the panorama centre, 20 degrees above the horizon
    u = int((90 / 360 + 0.5) * 1024)
    v = int((0.5 - 20 / 180) * 512)
    cv2.circle(pano, (u, v), 12, (0, 0, 255), -1)
    view = equirect_to_perspective(pano, yaw_deg=90, pitch_deg=20, fov_deg=60, out_w=200, out_h=200)
    ys, xs = np.nonzero(view[..., 2] > 200)
    assert len(xs) > 0
    assert math.isclose(xs.mean(), 100, abs_tol=4) and math.isclose(ys.mean(), 100, abs_tol=4)
