import numpy as np

from nltl_viz.postprocess import apply_grain, apply_vignette, build_vignette_mask


def test_vignette_mask_is_one_at_center():
    # width/height even so the true center (width/2, height/2) falls exactly
    # on a pixel index, giving r=0 there.
    mask = build_vignette_mask(100, 100, fraction=0.5)
    assert mask[50, 50, 0] == 1.0


def test_vignette_mask_decreases_toward_corners():
    mask = build_vignette_mask(100, 100, fraction=0.5)
    center = mask[50, 50, 0]
    corner = mask[0, 0, 0]
    edge_mid = mask[50, 0, 0]
    assert corner < edge_mid < center


def test_apply_vignette_darkens_frame():
    frame = np.full((10, 10, 3), 200, dtype=np.uint8)
    mask = build_vignette_mask(10, 10, fraction=0.8)
    out = apply_vignette(frame, mask)
    assert out[0, 0, 0] < frame[0, 0, 0]
    assert out[5, 5, 0] == frame[5, 5, 0]  # center (r=0) unaffected


def test_zero_strength_grain_is_noop():
    frame = np.full((11, 11, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = apply_grain(frame, 0.0, rng)
    assert np.array_equal(out, frame)


def test_grain_stays_within_byte_bounds():
    frame = np.zeros((11, 11, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = apply_grain(frame, strength=200.0, rng=rng)
    assert out.min() >= 0
    assert out.max() <= 255

    frame_high = np.full((11, 11, 3), 255, dtype=np.uint8)
    out_high = apply_grain(frame_high, strength=200.0, rng=rng)
    assert out_high.min() >= 0
    assert out_high.max() <= 255
