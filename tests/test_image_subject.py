from dataclasses import replace

import numpy as np
import pytest
from PIL import Image as PILImage

from nltl_viz.cli import resolve_subject
from nltl_viz.image import Image
from nltl_viz.palette import BUILTIN as PALETTES
from nltl_viz.preset import BUILTIN as PRESETS
from nltl_viz.render import (
    _BASE_HALF_FRACTION,
    Layout,
    Motion,
    Shape,
    _rect_perimeter_t,
    deformed_image,
    image_box,
    render_frame,
)

PALETTE = PALETTES["black"]
BG = tuple(int(PALETTE.background_color[i : i + 2], 16) for i in (1, 3, 5))
RED = (255, 0, 0, 255)


def _solid(w: int, h: int, rgba=RED) -> Image:
    return Image(PILImage.new("RGBA", (w, h), rgba))


def _preset(motion: str, **overrides):
    base = next((p for p in PRESETS.values() if p.motion == motion), PRESETS["industrial"])
    defaults = {"scale_amplitude": 0.2, "pulse_max_opacity": 0.8, "split_amplitude": 0.2}
    return replace(base, motion=motion, **{**defaults, **overrides})


def _render(image, motion, *, bands=None, flash=0.0, scale_value=0.5, size=200, **kw):
    preset = _preset(motion.value, grain_strength=0.0)
    bands = np.zeros(preset.band_count) if bands is None else bands
    return render_frame(
        bands, flash, (0, 0, 255), preset, PALETTE, size, image, motion, scale_value, **kw
    )


def _red_mask(frame):
    return (frame[:, :, 0] > 200) & (frame[:, :, 1] < 60) & (frame[:, :, 2] < 60)


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


# --- loading -------------------------------------------------------------


def test_load_applies_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = PILImage.Exif()
    exif[0x0112] = 6  # "rotate 90° CW to display"
    PILImage.new("RGB", (40, 20), (10, 20, 30)).save(path, exif=exif)
    image = Image.load(path)
    assert (image.width, image.height) == (20, 40)


def test_load_uses_first_frame_of_animated_file(tmp_path):
    path = tmp_path / "anim.gif"
    frames = [PILImage.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 255, 0))]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    rgba = Image.load(path).premultiplied(8, 8)
    assert rgba[4, 4, 0] > 200 and rgba[4, 4, 1] < 50


def test_load_rejects_non_image(tmp_path):
    path = tmp_path / "not-an-image.png"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Could not read image"):
        Image.load(path)


def test_resample_premultiplies_alpha():
    image = _solid(4, 4, (200, 100, 50, 128))
    px = image.premultiplied(4, 4)[0, 0]
    assert tuple(px) == (100, 50, 25, 128)


# --- geometry ------------------------------------------------------------


def test_contain_box_fits_same_bounding_square_as_shape():
    side = 2 * _BASE_HALF_FRACTION * 200
    landscape = image_box(_solid(400, 200), 200, _BASE_HALF_FRACTION)
    portrait = image_box(_solid(100, 400), 200, _BASE_HALF_FRACTION)
    assert landscape[2:] == (round(side), round(side / 2))
    assert portrait[2:] == (round(side / 4), round(side))
    # centered in the canvas, like the Shape's bounding square
    for x0, y0, w, h in (landscape, portrait):
        assert x0 + w / 2 == pytest.approx(100) and y0 + h / 2 == pytest.approx(100)


def test_rect_perimeter_starts_top_left_and_runs_clockwise():
    hw, hh = 2.0, 1.0  # W=4, H=2, perimeter 12
    dirs = {
        "top-left": (-2.0, -1.0, 0.0),
        "top-mid": (0.0, -1.0, 2 / 12),
        "right-mid": (2.0, 0.0, 5 / 12),
        "bottom-mid": (0.0, 1.0, 8 / 12),
        "left-mid": (-2.0, 0.0, 11 / 12),
    }
    for name, (dx, dy, expected) in dirs.items():
        t = float(_rect_perimeter_t(np.array(dx), np.array(dy), hw, hh))
        assert t == pytest.approx(expected, abs=1e-9) or (expected == 0 and t == pytest.approx(1.0)), name


def test_deform_with_silent_bands_leaves_image_unwarped():
    image = _solid(60, 30)
    region, (left, top), _ = deformed_image(image, np.zeros(8), 200, 0.5)
    x0, y0, w, h = image_box(image, 200, _BASE_HALF_FRACTION)
    assert (left, top) == (int(np.floor(x0)), int(np.floor(y0)))
    assert np.count_nonzero(region[:, :, 3] > 127) == pytest.approx(w * h, rel=0.02)


def test_deform_pushes_loud_band_side_outward_only():
    image = _solid(100, 100)
    bands = np.zeros(8)
    bands[2] = 1.0  # t = 2/8 = 0.25 → the top-right corner of the box
    frame = _render(image, Motion.deform, bands=bands)
    x0, y0, w, h = image_box(image, 200, _BASE_HALF_FRACTION)
    left, top, right, bottom = _bbox(_red_mask(frame))
    assert right > x0 + w + 5 and top < y0 - 5  # top-right corner pushed out
    assert left >= int(x0) - 1 and bottom <= int(np.ceil(y0 + h)) + 1  # bottom-left untouched


# --- per-Motion drawing ----------------------------------------------------


def test_image_pixels_replace_outline_color():
    frame = _render(_solid(50, 50), Motion.rigid)
    assert _red_mask(frame)[100, 100]
    assert tuple(frame[2, 2]) == BG  # background still from the palette


def test_pulse_opacity_follows_envelope_and_fills_frame_edge_to_edge():
    preset = _preset("pulse", grain_strength=0.0)
    image = _solid(50, 50)
    loud = _render(image, Motion.pulse, scale_value=1.0)
    quiet = _render(image, Motion.pulse, scale_value=0.0)
    assert tuple(quiet[100, 100]) == BG
    expected_red = round(preset.pulse_max_opacity * 255 + (1 - preset.pulse_max_opacity) * BG[0])
    assert abs(int(loud[100, 100, 0]) - expected_red) <= 2
    assert loud[1, 100, 0] == loud[100, 100, 0]  # inscribed edge-to-edge


def test_split_slides_halves_apart():
    image = _solid(100, 100)
    frame = _render(image, Motion.split, scale_value=1.0)
    mask = _red_mask(frame)
    top_left = _bbox(mask[:95])[0]
    bottom_left = _bbox(mask[105:])[0]
    assert top_left < bottom_left


def test_rigid_flash_is_a_silhouette_of_the_image_alpha():
    picture = PILImage.new("RGBA", (100, 100), (0, 0, 0, 0))
    picture.paste((255, 0, 0, 255), (0, 0, 50, 100))  # left half opaque, right half transparent
    image = Image(picture)
    preset = _preset("rigid", grain_strength=0.0, flash_size=0.8)
    frame = render_frame(
        np.zeros(preset.band_count), 1.0, (0, 0, 255), preset, PALETTE, 200, image, Motion.rigid
    )
    assert frame[100, 90, 2] > 200  # inside the opaque half: flash color
    assert tuple(frame[100, 110]) == BG  # transparent half: no flash spill


def test_deform_flash_is_drawn_on_top_and_masked_to_image():
    image = _solid(100, 100)
    preset = _preset("deform", grain_strength=0.0, flash_size=2.0)
    frame = render_frame(
        np.zeros(preset.band_count), 1.0, (0, 0, 255), preset, PALETTE, 200, image, Motion.deform
    )
    assert frame[100, 100, 2] > 200  # flash visible over the opaque image
    assert tuple(frame[5, 100]) == BG  # outside the image: no flash, despite a huge radius


def test_grid_draws_four_image_copies():
    frame = _render(_solid(50, 50), Motion.rigid, layout=Layout.grid)
    mask = _red_mask(frame)
    for cy in (50, 150):
        for cx in (50, 150):
            assert mask[cy, cx]


def test_overlay_canvas_keeps_transparent_corners_around_image():
    frame = _render(_solid(50, 50), Motion.deform, width=320, height=180, transparent_background=True)
    assert frame.shape == (180, 320, 4)
    assert frame[0, 0, 3] == 0
    assert frame[90, 160, 3] == 255


# --- CLI subject resolution -------------------------------------------------


def test_subject_defaults_to_face():
    assert resolve_subject(None, None) is Shape.face


def test_image_and_shape_are_mutually_exclusive(tmp_path):
    path = tmp_path / "x.png"
    PILImage.new("RGB", (4, 4)).save(path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_subject(Shape.space, path)
    assert isinstance(resolve_subject(None, path), Image)
