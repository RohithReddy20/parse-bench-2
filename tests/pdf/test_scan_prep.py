from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from docparse.pdf.scan_prep import (
    deskew_image,
    enhance_scan,
    is_degraded_scan,
    prepare_scan,
    slice_horizontal_bands,
)


def _synthetic_text_page(
    width: int = 900,
    height: int = 1200,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    ink_color: tuple[int, int, int] = (10, 10, 10),
    line_pitch: int = 36,
    line_height: int = 12,
    margin_x: int = 90,
    margin_y: int = 90,
) -> Image.Image:
    """Create a synthetic document page with crisp horizontal pseudo-text lines."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(42)
    y = margin_y
    while y + line_height < height - margin_y:
        x = margin_x
        right_limit = width - margin_x - int(rng.integers(0, 140))
        while x < right_limit:
            word_w = int(rng.integers(24, 95))
            if x + word_w > right_limit:
                break
            # Draw segmented vertical bars inside each word so the page resembles text strokes.
            cx = x
            while cx + 4 <= x + word_w:
                draw.rectangle((cx, y, cx + 3, y + line_height), fill=ink_color)
                cx += 6
            x += word_w + int(rng.integers(10, 20))
        y += line_pitch
    return img


def _degrade_yellow_shadow(clean: Image.Image) -> Image.Image:
    """Apply yellowed paper aging, uneven shadow gradient, and ink fading to a clean page."""
    arr = np.asarray(clean, dtype=np.float32)
    h, w, _ = arr.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    # Uneven scanner illumination: bright top-left, heavy shadow at bottom-right.
    illumination = 1.0 - 0.28 * yy - 0.14 * xx
    # Yellowed paper tint (R > G > B) plus faded ink lift.
    paper_tint = np.array([0.92, 0.87, 0.68], dtype=np.float32)
    faded = arr * 0.62 + 82.0
    degraded = faded * illumination[..., None] * paper_tint[None, None, :]
    return Image.fromarray(np.clip(np.rint(degraded), 0, 255).astype(np.uint8), mode="RGB")


def test_is_degraded_scan_distinguishes_clean_digital_from_degraded():
    clean = _synthetic_text_page()
    diag_clean = is_degraded_scan(clean)
    assert set(diag_clean) >= {
        "background_mean",
        "contrast_p95_p5",
        "yellow_cast",
        "ink_ratio",
        "skew_deg",
        "needs_enhance",
    }
    assert diag_clean["needs_enhance"] is False
    assert diag_clean["background_mean"] >= 254.0
    assert abs(diag_clean["yellow_cast"]) < 2.0
    assert diag_clean["ink_ratio"] > 0.02
    assert diag_clean["skew_deg"] == 0.0

    blank = Image.new("RGB", (600, 800), (255, 255, 255))
    diag_blank = is_degraded_scan(blank)
    assert diag_blank["needs_enhance"] is False
    assert diag_blank["ink_ratio"] == 0.0
    assert diag_blank["skew_deg"] == 0.0

    degraded = _degrade_yellow_shadow(clean)
    diag_deg = is_degraded_scan(degraded)
    assert diag_deg["needs_enhance"] is True
    assert diag_deg["background_mean"] < 235.0
    assert diag_deg["yellow_cast"] > 20.0
    assert diag_deg["ink_ratio"] > 0.02


def test_deskew_recovers_known_rotation_and_leaves_upright_untouched():
    clean = _synthetic_text_page(width=1000, height=1300)

    # 0-degree image is left completely untouched.
    same, zero_angle = deskew_image(clean)
    assert zero_angle == 0.0
    assert same is clean

    # Positive and negative 2.5-degree rotations are recovered to within 0.5 degrees.
    for skew_deg in (2.5, -2.5):
        skewed = clean.rotate(skew_deg, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
        fixed, applied = deskew_image(skewed)
        assert abs(applied - (-skew_deg)) <= 0.5, f"expected ~{-skew_deg}, got {applied}"
        assert fixed.size == skewed.size

        # Corners filled with white, not black.
        assert min(fixed.getpixel((2, 2))) >= 245

        # Running deskew a second time finds the straightened page already upright.
        _, residual = deskew_image(fixed)
        assert abs(residual) <= 0.5


def test_deskew_ignores_blank_page():
    blank = Image.new("RGB", (600, 800), (255, 255, 255))
    out, angle = deskew_image(blank)
    assert angle == 0.0
    assert out is blank


def test_enhance_scan_flattens_yellowed_shadowed_page_and_preserves_ink():
    clean = _synthetic_text_page()
    degraded = _degrade_yellow_shadow(clean)

    restored = enhance_scan(degraded)
    assert restored.mode == "RGB"
    assert restored.size == degraded.size

    clean_arr = np.asarray(clean, dtype=np.float32)
    rest_arr = np.asarray(restored, dtype=np.float32)
    gray_clean = clean_arr.mean(axis=-1)
    gray_rest = rest_arr.mean(axis=-1)

    bg_pixels = rest_arr[gray_clean > 240]
    ink_pixels = gray_rest[gray_clean < 40]

    # Background flattened to bright neutral white everywhere (top AND shadowed bottom).
    assert float(bg_pixels.mean()) >= 245.0
    assert abs(float((bg_pixels[:, 0] - bg_pixels[:, 2]).mean())) < 5.0
    top_bg = float(gray_rest[:200, :][gray_clean[:200, :] > 240].mean())
    bot_bg = float(gray_rest[-200:, :][gray_clean[-200:, :] > 240].mean())
    assert abs(top_bg - bot_bg) < 8.0

    # Ink remains sharp and strongly dark compared to the white paper.
    assert float(np.median(ink_pixels)) < 60.0
    assert float(bg_pixels.mean()) - float(np.median(ink_pixels)) > 190.0


def test_enhance_scan_leaves_clean_digital_page_unchanged():
    clean = _synthetic_text_page()
    out = enhance_scan(clean)
    assert out is clean
    assert np.array_equal(np.asarray(out), np.asarray(clean))


def test_enhance_scan_preserves_color_highlights_and_handles_grayscale_scans():
    # Color highlight (red stamp) on yellowed/shadowed paper must remain red, not collapse to gray.
    page = _synthetic_text_page()
    draw = ImageDraw.Draw(page)
    draw.rectangle((620, 120, 780, 220), fill=(210, 35, 35))
    yellowed = _degrade_yellow_shadow(page)
    restored = enhance_scan(yellowed)
    stamp = np.asarray(restored, dtype=np.float32)[135:205, 635:765]
    assert float(stamp[..., 0].mean() - stamp[..., 2].mean()) > 100.0

    # Uniform yellowed paper with a blue highlight exercises the fast uniform-bg RGB path.
    uniform_yellow = _synthetic_text_page(bg_color=(232, 220, 175), ink_color=(115, 110, 95))
    ImageDraw.Draw(uniform_yellow).rectangle((620, 120, 780, 220), fill=(40, 85, 200))
    uniform_restored = enhance_scan(uniform_yellow)
    blue_patch = np.asarray(uniform_restored, dtype=np.float32)[135:205, 635:765]
    assert float(blue_patch[..., 2].mean() - blue_patch[..., 0].mean()) > 100.0
    assert is_degraded_scan(uniform_restored)["background_mean"] >= 248.0

    # Pure grayscale degraded scan restores to crisp high-contrast RGB.
    gray_deg = yellowed.convert("L").convert("RGB")
    gray_rest = enhance_scan(gray_deg)
    assert gray_rest.mode == "RGB"
    diag_after = is_degraded_scan(gray_rest)
    assert diag_after["background_mean"] >= 245.0

    # Faint pencil ink on lightly aged paper stays legible (not erased to white).
    faint = _synthetic_text_page(bg_color=(238, 234, 222), ink_color=(172, 168, 160))
    faint_rest = enhance_scan(faint)
    faint_clean = np.asarray(_synthetic_text_page(), dtype=np.float32).mean(axis=-1)
    faint_gray = np.asarray(faint_rest, dtype=np.float32).mean(axis=-1)
    assert float(np.median(faint_gray[faint_clean < 40])) < 110.0
    assert float(faint_gray[faint_clean > 240].mean()) >= 248.0


def test_prepare_scan_end_to_end_and_flags():
    clean = _synthetic_text_page(width=1000, height=1300)

    out_clean, info_clean = prepare_scan(clean)
    assert out_clean is clean
    assert info_clean["enhanced"] is False
    assert info_clean["deskewed"] is False
    assert info_clean["deskew_angle"] == 0.0

    degraded = _degrade_yellow_shadow(clean)
    skewed_deg = degraded.rotate(2.25, resample=Image.Resampling.BICUBIC, fillcolor=(215, 200, 155))

    out, info = prepare_scan(skewed_deg)
    assert info["enhanced"] is True
    assert info["deskewed"] is True
    assert abs(float(info["deskew_angle"]) - (-2.25)) <= 0.5
    assert out.mode == "RGB"
    assert out.size == skewed_deg.size
    # Enhanced before rotation => corners are clean bright white with no dark/yellow wedges.
    assert min(out.getpixel((4, 4))) >= 245

    # Respects disabling flags independently.
    _, info_no_deskew = prepare_scan(skewed_deg, deskew=False, enhance=True)
    assert info_no_deskew["enhanced"] is True
    assert info_no_deskew["deskewed"] is False

    _, info_no_enhance = prepare_scan(skewed_deg, deskew=True, enhance=False)
    assert info_no_enhance["enhanced"] is False
    assert info_no_enhance["deskewed"] is True


def test_slice_horizontal_bands_snaps_to_whitespace_gutters():
    # Construct a 1000px tall page where heavy text bars sit squarely across the naive cut y's
    # (including y=500, y=460, y=540) with clean whitespace gutters between them.
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 20px-tall dark text lines spaced every 40px, centered on 20, 60, ..., 460, 500, 540, ...
    # So y=460, y=500, y=540 are dead-center inside solid ink bars, while 440, 480, 520, 560 are pure white!
    for center_y in range(20, height - 10, 40):
        draw.rectangle((60, center_y - 8, width - 60, center_y + 8), fill=(15, 15, 15))

    gray = np.asarray(img.convert("L"))
    # Confirm naive midpoint y=500 and overlap targets 460/540 are indeed ink rows.
    assert gray[500, 60 : width - 60].mean() < 30
    assert gray[460, 60 : width - 60].mean() < 30
    assert gray[540, 60 : width - 60].mean() < 30

    # Zero-overlap split at naive midpoint 500 must snap to a pure white gutter (480 or 520).
    bands_no_ov = slice_horizontal_bands(img, n_bands=2, overlap_frac=0.0)
    assert len(bands_no_ov) == 2
    (b0, box0), (b1, box1) = bands_no_ov
    assert box0[1] == 0 and box1[3] == height
    assert box0[3] == box1[1]
    cut_y = box0[3]
    assert abs(cut_y - 500) <= int(0.04 * height)
    assert gray[cut_y, :].min() == 255, f"cut at y={cut_y} sliced through ink"
    assert b0.size == (box0[2] - box0[0], box0[3] - box0[1])
    assert b1.size == (box1[2] - box1[0], box1[3] - box1[1])

    # Default overlapping 2-band split: both interior cut edges snap to white gutters and bands overlap.
    bands = slice_horizontal_bands(img, n_bands=2, overlap_frac=0.08)
    assert len(bands) == 2
    (_, b0_box), (_, b1_box) = bands
    assert b0_box[1] == 0 and b1_box[3] == height
    assert b1_box[1] < b0_box[3], "overlapping bands must overlap"
    assert gray[b0_box[3], :].min() == 255, f"band 0 bottom edge y={b0_box[3]} sliced ink"
    assert gray[b1_box[1], :].min() == 255, f"band 1 top edge y={b1_box[1]} sliced ink"

    # 3-band split and explicit split_ys also cover the full page without slicing text lines.
    bands3 = slice_horizontal_bands(img, n_bands=3, overlap_frac=0.06)
    assert len(bands3) == 3
    assert bands3[0][1][1] == 0 and bands3[-1][1][3] == height
    for i in range(len(bands3) - 1):
        y_end = bands3[i][1][3]
        y_next = bands3[i + 1][1][1]
        assert y_next <= y_end
        assert gray[y_end, :].min() == 255
        assert gray[y_next, :].min() == 255

    explicit = slice_horizontal_bands(img, overlap_frac=0.0, split_ys=[300, 700])
    assert len(explicit) == 3
    for i in range(2):
        assert gray[explicit[i][1][3], :].min() == 255

    single = slice_horizontal_bands(img, n_bands=1)
    assert len(single) == 1 and single[0][1] == (0, 0, width, height)
