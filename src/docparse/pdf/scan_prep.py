"""Lightweight, deterministic image preprocessing for degraded document scans.

Pure Pillow + NumPy restoration aimed at faded, skewed, yellowed, or unevenly lit
image-only scans (historical archives, scanned tax/financial forms, image-only tables)
while leaving crisp born-digital page renders untouched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

_DIAG_MAX_SIDE = 600
_SKEW_MAX_SIDE = 600
_MIN_SKEW_DEG = 0.4
_MIN_SKEW_RATIO = 1.08
_BG_FLOOR = 80.0
_HUGE_PIXELS = 25_000_000
_LUT_CHUNK_ROWS = 512
_MILD_SHARPEN_KERNEL = ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 32, -1, -1, -1, -1), scale=24)


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _downscale_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _sample_rgb(image: Image.Image, max_side: int = _DIAG_MAX_SIDE) -> Image.Image:
    w, h = image.size
    sw, sh = _downscale_size(w, h, max_side)
    small = image if (sw, sh) == (w, h) else image.resize((sw, sh), Image.Resampling.NEAREST)
    return _ensure_rgb(small)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _estimate_coarse_bg_pil(img: Image.Image) -> Image.Image:
    """Coarse (~64px) smooth local paper-brightness grid via morphological dilation + box blur.

    Downscales with NEAREST to ~320px (preserving true paper pixels between text lines),
    strips single-pixel hot noise with a 3x3 median, dilates paper over text strokes with
    a 5x5 max filter, reduces to a ~64px illumination grid, bridges bold headers with a
    3x3 max filter, and smooths with a box blur.
    """
    w, h = img.size
    if min(w, h) < 16:
        return img.copy()
    mw, mh = _downscale_size(w, h, 320)
    mid = img if (mw, mh) == (w, h) else img.resize((mw, mh), Image.Resampling.NEAREST)
    paper = mid.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MaxFilter(5))
    gw, gh = _downscale_size(w, h, 64)
    gw, gh = max(8, gw), max(8, gh)
    coarse = paper.resize((gw, gh), Image.Resampling.BILINEAR)
    return coarse.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.BoxBlur(2))


def _estimate_local_bg_pil(img: Image.Image) -> Image.Image:
    """Full-size smooth local paper-brightness field."""
    w, h = img.size
    if min(w, h) < 16:
        return img.copy()
    return _estimate_coarse_bg_pil(img).resize((w, h), Image.Resampling.BILINEAR)


def _visual_metrics(
    image: Image.Image,
    sample_rgb: Image.Image | None = None,
) -> tuple[dict[str, float | bool], Image.Image, Image.Image, np.ndarray]:
    """Fast scan-quality metrics without skew estimation.

    Returns `(metrics, sample_rgb, sample_gray_img, sample_local_bg)` so downstream steps
    (`_estimate_skew`, `_enhance_core`) reuse the same downscaled view without re-sampling.
    """
    w, h = image.size
    if w == 0 or h == 0:
        empty = Image.new("RGB", (1, 1), (255, 255, 255))
        return (
            {
                "background_mean": 255.0,
                "background_std": 0.0,
                "contrast_p95_p5": 0.0,
                "yellow_cast": 0.0,
                "ink_ratio": 0.0,
                "is_grayscale": True,
                "has_color_content": False,
                "needs_enhance": False,
            },
            empty,
            empty.convert("L"),
            np.full((1, 1), 255.0, dtype=np.float32),
        )

    s_rgb = sample_rgb if sample_rgb is not None else _sample_rgb(image, _DIAG_MAX_SIDE)
    sw, sh = s_rgb.size
    arr = np.asarray(s_rgb, dtype=np.float32)
    gray = _luminance(arr)

    p05, p5, p70, p95 = (float(v) for v in np.percentile(gray, [0.5, 5.0, 70.0, 95.0]))
    bg_mask = gray >= p70
    bg_gray = gray[bg_mask]
    bg_mean = float(bg_gray.mean()) if bg_gray.size else 255.0
    bg_std = float(bg_gray.std()) if bg_gray.size else 0.0
    bg_rgb = arr[bg_mask]
    yellow_cast = float((bg_rgb[:, 0] - bg_rgb[:, 2]).mean()) if bg_rgb.size else 0.0
    contrast_p95_p5 = float(p95 - p5)

    gray_img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    lbg = np.asarray(_estimate_local_bg_pil(gray_img), dtype=np.float32)
    ink_mask = (lbg - gray) > np.maximum(20.0, 0.12 * lbg)
    ink_ratio = float(ink_mask.mean())

    my, mx = max(0, sh // 20), max(0, sw // 20)
    lbg_inner = lbg[my : sh - my, mx : sw - mx] if (sh - 2 * my > 0 and sw - 2 * mx > 0) else lbg
    lbg_p50 = float(np.percentile(lbg_inner, 50.0))

    chroma = arr.max(axis=-1) - arr.min(axis=-1)
    is_raw_grayscale = float(np.percentile(chroma, 99.0)) <= 8.0
    if is_raw_grayscale:
        has_color_content = False
    else:
        # Distinguish genuine chromatic content (stamps, color highlights, blue/red ink)
        # from uniform yellowed/sepia monochrome paper aging by checking saturation after
        # local RGB paper white-balancing.
        bg_rgb_est = np.maximum(np.asarray(_estimate_local_bg_pil(s_rgb), dtype=np.float32), _BG_FLOOR)
        norm_rgb = np.clip(arr / bg_rgb_est * 255.0, 0.0, 255.0)
        wb_max = norm_rgb.max(axis=-1)
        wb_ch = wb_max - norm_rgb.min(axis=-1)
        wb_sat = np.where(wb_max >= 50.0, wb_ch / np.maximum(wb_max, 1.0), 0.0)
        has_color_content = float(((wb_ch > 60.0) & (wb_sat > 0.30)).mean()) >= 0.0015

    has_ink = (bg_mean - p05) >= 15.0 and ink_ratio >= 0.0005
    dark_or_aged_bg = bg_mean < 246.0 or lbg_p50 < 244.0
    tinted_bg = yellow_cast >= 10.0 or (abs(yellow_cast) >= 12.0 and bg_mean < 252.0)
    uneven_bg = bg_std >= 5.0
    faded_offwhite = bg_mean < 253.0 and has_ink and p05 > 75.0
    severely_faded = has_ink and p05 > 115.0 and contrast_p95_p5 < 130.0
    needs_enhance = bool(dark_or_aged_bg or tinted_bg or uneven_bg or faded_offwhite or severely_faded)

    metrics: dict[str, float | bool] = {
        "background_mean": round(bg_mean, 2),
        "background_std": round(bg_std, 2),
        "contrast_p95_p5": round(contrast_p95_p5, 2),
        "yellow_cast": round(yellow_cast, 2),
        "ink_ratio": round(ink_ratio, 4),
        "is_grayscale": is_raw_grayscale,
        "has_color_content": has_color_content,
        "needs_enhance": needs_enhance,
    }
    return metrics, s_rgb, gray_img, lbg


def _estimate_skew(
    image: Image.Image,
    max_angle: float = 5.0,
    step: float = 0.25,
    precomputed: tuple[Image.Image, np.ndarray] | None = None,
) -> tuple[float, float]:
    """Estimate counter-clockwise deskew rotation via horizontal projection-profile variance.

    Returns `(applied_angle_deg, variance_ratio_vs_zero)`. `applied_angle_deg` is `0.0`
    whenever `abs(angle) < _MIN_SKEW_DEG` or `variance_ratio < _MIN_SKEW_RATIO`.
    """
    w, h = image.size
    if min(w, h) < 32 or max_angle <= 0.0 or step <= 0.0:
        return 0.0, 1.0

    if precomputed is not None and w * h > _HUGE_PIXELS:
        gray_img, bg_small = precomputed
        sw, sh = gray_img.size
        gray = np.asarray(gray_img, dtype=np.float32)
    else:
        sw, sh = _downscale_size(w, h, _SKEW_MAX_SIDE)
        resample = Image.Resampling.NEAREST if (w * h > _HUGE_PIXELS) else Image.Resampling.BILINEAR
        small = image if (sw, sh) == (w, h) else image.resize((sw, sh), resample)
        gray_img = small if small.mode == "L" else small.convert("L")
        gray = np.asarray(gray_img, dtype=np.float32)
        bg_small = np.asarray(_estimate_local_bg_pil(gray_img), dtype=np.float32)

    diff = bg_small - gray

    my, mx = max(1, int(sh * 0.03)), max(1, int(sw * 0.03))
    inner = diff[my : sh - my, mx : sw - mx]
    if inner.size == 0:
        return 0.0, 1.0

    thr = max(12.0, float(np.percentile(inner, 90.0)) * 0.5)
    bw = (diff > thr).astype(np.uint8)
    bw[:my, :] = 0
    bw[sh - my :, :] = 0
    bw[:, :mx] = 0
    bw[:, sw - mx :] = 0

    ink_count = int(bw.sum())
    if ink_count < 50 or ink_count < int(0.0005 * inner.size):
        return 0.0, 1.0

    bw_img = Image.fromarray(bw * 255, mode="L")

    def _proj_var(angle: float) -> float:
        if angle == 0.0:
            arr = bw
        else:
            rot = bw_img.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            arr = np.asarray(rot)
        proj = arr.sum(axis=1, dtype=np.float64)
        return float(np.var(proj))

    v0 = _proj_var(0.0)
    if v0 <= 1e-9:
        return 0.0, 1.0

    # `bw` has values {0, 1} while rotated `bw_img` has values {0, 255}; scale v0 to the same 0..255 basis.
    v0_scaled = v0 * (255.0 * 255.0)
    best_angle = 0.0
    best_var = v0_scaled

    n_steps = int(round(max_angle / step))
    for k in range(-n_steps, n_steps + 1):
        if k == 0:
            continue
        ang = round(k * step, 6)
        if abs(ang) > max_angle + 1e-9:
            continue
        v = _proj_var(ang)
        if v > best_var:
            best_var = v
            best_angle = ang

    ratio = best_var / v0_scaled
    if abs(best_angle) < _MIN_SKEW_DEG or ratio < _MIN_SKEW_RATIO:
        return 0.0, round(ratio, 4)
    return round(best_angle, 4), round(ratio, 4)


def is_degraded_scan(image: Image.Image) -> dict[str, float | bool]:
    """Fast visual diagnostics on a PIL image to decide whether scan restoration is needed.

    Returns a dict with:
        - `background_mean`: mean luminance (0..255) of bright paper background pixels
        - `contrast_p95_p5`: 95th minus 5th luminance percentile
        - `yellow_cast`: mean (R - B) on background pixels (positive = yellowed paper)
        - `ink_ratio`: fraction of pixels noticeably darker than local paper background
        - `skew_deg`: decisive counter-clockwise deskew angle in degrees (0.0 if straight)
        - `needs_enhance`: True when the page has dark/uneven/yellowed paper or faded ink
    """
    metrics, _, gray_img, lbg = _visual_metrics(image)
    skew_deg, _ = _estimate_skew(image, precomputed=(gray_img, lbg))
    return {
        "background_mean": float(metrics["background_mean"]),
        "contrast_p95_p5": float(metrics["contrast_p95_p5"]),
        "yellow_cast": float(metrics["yellow_cast"]),
        "ink_ratio": float(metrics["ink_ratio"]),
        "skew_deg": float(skew_deg),
        "needs_enhance": bool(metrics["needs_enhance"]),
    }


def _rotate_image(image: Image.Image, angle: float, is_grayscale: bool | None = None) -> Image.Image:
    resample = Image.Resampling.BILINEAR if (image.width * image.height > _HUGE_PIXELS) else Image.Resampling.BICUBIC
    if image.mode == "L":
        return image.rotate(angle, resample=resample, fillcolor=255)
    if is_grayscale:
        return image.convert("L").rotate(angle, resample=resample, fillcolor=255).convert("RGB")
    rgb = _ensure_rgb(image)
    return rgb.rotate(angle, resample=resample, fillcolor=(255, 255, 255))


def deskew_image(
    image: Image.Image,
    max_angle: float = 5.0,
    step: float = 0.25,
) -> tuple[Image.Image, float]:
    """Deskew a page image by maximizing horizontal projection-profile variance.

    Only rotates when `abs(angle) >= 0.4` degrees and the variance peak is decisive
    (ratio vs 0° >= 1.08). Returns `(image_out, applied_angle_deg)` where
    `applied_angle_deg` is the counter-clockwise rotation applied via `PIL.Image.rotate`
    (`0.0` and the original image when left untouched).
    """
    angle, _ = _estimate_skew(image, max_angle=max_angle, step=step)
    if angle == 0.0:
        return image, 0.0
    is_gray = image.mode == "L"
    if not is_gray:
        sw, sh = _downscale_size(image.width, image.height, 256)
        s = np.asarray(_ensure_rgb(image.resize((sw, sh), Image.Resampling.NEAREST)), dtype=np.int16)
        is_gray = int(np.max(s.max(axis=-1) - s.min(axis=-1))) <= 4
    return _rotate_image(image, angle, is_grayscale=is_gray), angle


def _build_norm_stretch_lut(lo: float, hi: float) -> np.ndarray:
    """Precompute a 64 KB 1D LUT mapping `(pixel << 8) | local_bg` -> restored uint8 pixel.

    Combines local-background division (`pixel / max(bg, _BG_FLOOR) * 255`) with a gentle
    linear contrast stretch (`lo -> dark_target`, `hi -> 255`) in a single cache-resident lookup.
    """
    ii = np.arange(256, dtype=np.float32)[:, None]
    bb = np.maximum(np.arange(256, dtype=np.float32)[None, :], _BG_FLOOR)
    norm = np.clip(ii / bb * 255.0, 0.0, 255.0)

    dark_target = min(15.0, lo)
    span = max(hi - lo, 1.0)
    stretched = (norm - lo) * ((255.0 - dark_target) / span) + dark_target
    return np.rint(np.clip(stretched, 0.0, 255.0)).astype(np.uint8).ravel()


def _compute_stretch_bounds(
    sample_rgb: Image.Image,
    is_grayscale: bool,
    precomputed_gray: tuple[Image.Image, np.ndarray] | None = None,
) -> tuple[float, float]:
    """Estimate gentle `(lo, hi)` stretch points on a small background-normalized preview."""
    if is_grayscale and precomputed_gray is not None:
        gray_img, lbg = precomputed_gray
        a = np.asarray(gray_img, dtype=np.float32)
        b = np.maximum(lbg, _BG_FLOOR)
        norm_gray = np.clip(a / b * 255.0, 0.0, 255.0)
    else:
        base = sample_rgb.convert("L") if is_grayscale else _ensure_rgb(sample_rgb)
        bg = _estimate_local_bg_pil(base)
        a = np.asarray(base, dtype=np.float32)
        b = np.maximum(np.asarray(bg, dtype=np.float32), _BG_FLOOR)
        norm = np.clip(a / b * 255.0, 0.0, 255.0)
        norm_gray = norm if is_grayscale else _luminance(norm)

    sh, sw = norm_gray.shape
    my, mx = max(0, int(sh * 0.03)), max(0, int(sw * 0.03))
    inner = norm_gray[my : sh - my, mx : sw - mx] if (sh - 2 * my > 0 and sw - 2 * mx > 0) else norm_gray

    hi = float(np.clip(np.percentile(inner, 85.0), 220.0, 250.0))
    ink = inner[inner < hi - 18.0]
    if ink.size >= max(8, int(0.0005 * inner.size)):
        lo_ink = float(np.percentile(ink, 15.0))
        lo_p1 = float(np.percentile(inner, 1.5))
        lo = min(lo_ink, lo_p1) if lo_p1 < hi - 18.0 else lo_ink
    else:
        lo = 0.0

    # Cap contrast stretch gain (~3.2x max) so faint paper grain is never amplified into blotches.
    lo = float(np.clip(lo, 0.0, max(0.0, hi - 75.0)))
    return lo, hi


def _apply_lut_chunked(img_arr: np.ndarray, bg_arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    out = np.empty_like(img_arr)
    height = img_arr.shape[0]
    for y in range(0, height, _LUT_CHUNK_ROWS):
        y1 = min(height, y + _LUT_CHUNK_ROWS)
        idx = (img_arr[y:y1].astype(np.uint16) << 8) | bg_arr[y:y1]
        out[y:y1] = lut[idx]
    return out


def _mild_sharpen(img: Image.Image) -> Image.Image:
    if img.width * img.height > _HUGE_PIXELS:
        return img.filter(_MILD_SHARPEN_KERNEL)
    return img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=70, threshold=4))


def _is_uniform_bg(coarse_arr: np.ndarray) -> bool:
    """True when the local paper-brightness grid is flat across the entire page (no shadows/vignetting)."""
    if coarse_arr.ndim == 2:
        p1, p5, p95, p99 = (float(v) for v in np.percentile(coarse_arr, [1.0, 5.0, 95.0, 99.0]))
        return (p95 - p5) <= 12.0 and (p99 - p1) <= 22.0
    for ch in range(coarse_arr.shape[-1]):
        p1, p5, p95, p99 = (float(v) for v in np.percentile(coarse_arr[..., ch], [1.0, 5.0, 95.0, 99.0]))
        if (p95 - p5) > 12.0 or (p99 - p1) > 22.0:
            return False
    return True


def _enhance_core(
    image: Image.Image,
    is_grayscale: bool,
    keep_mode_l: bool = False,
    sample_rgb: Image.Image | None = None,
    precomputed_gray: tuple[Image.Image, np.ndarray] | None = None,
) -> Image.Image:
    """Restore a degraded scan via local background division, gentle contrast stretch, and mild sharpen."""
    s_rgb = sample_rgb if sample_rgb is not None else _sample_rgb(image, _DIAG_MAX_SIDE)
    lo, hi = _compute_stretch_bounds(s_rgb, is_grayscale=is_grayscale, precomputed_gray=precomputed_gray)
    lut = _build_norm_stretch_lut(lo, hi)

    work = image.convert("L") if is_grayscale else _ensure_rgb(image)
    coarse_bg = _estimate_coarse_bg_pil(s_rgb.convert("L") if is_grayscale else s_rgb)
    coarse_arr = np.asarray(coarse_bg)

    if _is_uniform_bg(coarse_arr):
        # Uniform paper illumination: single-pass C LUT in Pillow avoids allocating a full-res bg buffer.
        idx_base = np.arange(256, dtype=np.uint16) << 8
        if is_grayscale:
            bg_val = int(round(float(np.median(coarse_arr))))
            restored = work.point(lut[idx_base | bg_val].tolist())
        else:
            table: list[int] = []
            for ch in range(3):
                bg_val = int(round(float(np.median(coarse_arr[..., ch]))))
                table.extend(lut[idx_base | bg_val].tolist())
            restored = work.point(table)
    else:
        bg = coarse_bg.resize(work.size, Image.Resampling.BILINEAR)
        out_arr = _apply_lut_chunked(np.asarray(work), np.asarray(bg), lut)
        restored = Image.fromarray(out_arr, mode="L" if is_grayscale else "RGB")

    restored = _mild_sharpen(restored)
    if is_grayscale and not keep_mode_l:
        return restored.convert("RGB")
    return restored


def enhance_scan(image: Image.Image) -> Image.Image:
    """Gentle, artifact-free scan restoration; leaves crisp born-digital renders untouched.

    Normalizes uneven yellowed or shadowed paper backgrounds via local background
    division, applies a bounded percentile contrast stretch plus mild unsharp masking,
    and preserves RGB whenever real color highlights/stamps are present (using a faster
    crisp grayscale path converted back to RGB for pure grayscale or monochrome yellowed scans).
    """
    metrics, s_rgb, gray_img, lbg = _visual_metrics(image)
    if not metrics["needs_enhance"]:
        return image
    return _enhance_core(
        image,
        is_grayscale=not bool(metrics["has_color_content"]),
        sample_rgb=s_rgb,
        precomputed_gray=(gray_img, lbg),
    )


def prepare_scan(
    image: Image.Image,
    deskew: bool = True,
    enhance: bool = True,
) -> tuple[Image.Image, dict[str, float | bool]]:
    """Top-level scan preprocessing entry point returning `(processed_image, info_dict)`.

    Runs fast visual diagnostics once; when both enhancement and deskew trigger, restores
    the paper background to white before rotating so white deskew corners blend seamlessly.
    """
    metrics, s_rgb, gray_img, lbg = _visual_metrics(image)
    skew_deg, _ = _estimate_skew(image, precomputed=(gray_img, lbg)) if deskew else (0.0, 1.0)

    do_enhance = bool(enhance and metrics["needs_enhance"])
    do_deskew = bool(deskew and skew_deg != 0.0)
    use_mono_enhance = not bool(metrics["has_color_content"])
    is_raw_gray = bool(metrics["is_grayscale"])

    out = image
    if do_enhance:
        # Keep single-channel L through rotation on monochrome scans to halve rotation cost.
        out = _enhance_core(
            out,
            is_grayscale=use_mono_enhance,
            keep_mode_l=use_mono_enhance and do_deskew,
            sample_rgb=s_rgb,
            precomputed_gray=(gray_img, lbg),
        )
    if do_deskew:
        rot_gray = use_mono_enhance if do_enhance else is_raw_gray
        out = _rotate_image(out, skew_deg, is_grayscale=rot_gray)
        if out.mode != "RGB":
            out = out.convert("RGB")

    info: dict[str, float | bool] = {
        "background_mean": float(metrics["background_mean"]),
        "contrast_p95_p5": float(metrics["contrast_p95_p5"]),
        "yellow_cast": float(metrics["yellow_cast"]),
        "ink_ratio": float(metrics["ink_ratio"]),
        "skew_deg": float(skew_deg),
        "needs_enhance": bool(metrics["needs_enhance"]),
        "deskewed": do_deskew,
        "deskew_angle": float(skew_deg) if do_deskew else 0.0,
        "enhanced": do_enhance,
        "grayscale": use_mono_enhance if do_enhance else is_raw_gray,
    }
    return out, info


def _row_ink_profile(image: Image.Image) -> tuple[np.ndarray, float]:
    """Horizontal ink energy per row on a width-downsampled grayscale view.

    Returns `(profile, y_scale)` where `profile[r]` corresponds to page y `r / y_scale`.
    """
    w, h = image.size
    pw = min(w, 600)
    ph = min(h, 4096)
    y_scale = ph / h
    resized = image if (pw, ph) == (w, h) else image.resize((pw, ph), Image.Resampling.BILINEAR)
    gray_img = resized if resized.mode == "L" else resized.convert("L")
    gray = np.asarray(gray_img, dtype=np.float32)

    # Subtract smooth local row background so yellowed or shadowed regions don't look like ink.
    bg = np.asarray(_estimate_local_bg_pil(gray_img), dtype=np.float32)
    ink = np.maximum(0.0, (bg - gray) - 15.0)
    profile = ink.mean(axis=1, dtype=np.float64)

    # Small triangular smoothing prefers the center of a whitespace gutter over a row grazing ink.
    rad = max(2, min(8, ph // 300))
    kernel = np.concatenate([np.arange(1, rad + 2), np.arange(rad, 0, -1)]).astype(np.float64)
    kernel /= kernel.sum()
    smoothed = np.convolve(profile, kernel, mode="same")
    return smoothed, y_scale


def _snap_y(
    target_y: int,
    low_y: int,
    high_y: int,
    profile: np.ndarray,
    y_scale: float,
) -> int:
    """Snap `target_y` to the lowest-ink row within `[low_y, high_y]`."""
    ph = len(profile)
    lo_idx = max(0, min(ph - 1, int(round(low_y * y_scale))))
    hi_idx = max(lo_idx, min(ph - 1, int(round(high_y * y_scale))))
    target_idx = target_y * y_scale

    window = profile[lo_idx : hi_idx + 1]
    # Tiny proximity tie-breaker prefers cuts closer to target_y among equal-ink whitespace rows.
    dist = np.abs(np.arange(lo_idx, hi_idx + 1, dtype=np.float64) - target_idx)
    best_rel = int(np.argmin(window + 1e-4 * dist))
    best_idx = lo_idx + best_rel
    snapped = int(round(best_idx / y_scale))
    return max(low_y, min(high_y, snapped))


def slice_horizontal_bands(
    image: Image.Image,
    n_bands: int = 2,
    overlap_frac: float = 0.08,
    split_ys: list[int] | None = None,
) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Split a tall page image into overlapping horizontal bands without cutting text lines.

    Each interior cut line snaps to the lowest-ink horizontal whitespace gutter within
    ±4% of page height around its target y. Returns a list of `(band_image, (x0, y0, x1, y1))`
    tuples in pixel coordinates of `image`.
    """
    w, h = image.size
    if h <= 0 or w <= 0:
        return []

    if split_ys is not None:
        splits = sorted({int(y) for y in split_ys if 0 < int(y) < h})
    else:
        if n_bands <= 1:
            return [(image.crop((0, 0, w, h)), (0, 0, w, h))]
        splits = [int(round(h * i / n_bands)) for i in range(1, n_bands)]
        splits = [y for y in splits if 0 < y < h]

    if not splits:
        return [(image.crop((0, 0, w, h)), (0, 0, w, h))]

    profile, y_scale = _row_ink_profile(image)
    tol = max(1, int(round(0.04 * h)))
    k_bands = len(splits) + 1
    ov_frac = max(0.0, float(overlap_frac))
    half_overlap = int(round(0.5 * ov_frac * min(h, 2.0 * h / k_bands)))

    # Build `[y0, y1]` for every band, snapping each interior boundary to a whitespace gutter.
    starts = [0] * k_bands
    ends = [h] * k_bands

    for idx, split_y in enumerate(splits):
        prev_bound = splits[idx - 1] if idx > 0 else 0
        next_bound = splits[idx + 1] if idx + 1 < len(splits) else h
        max_half = max(0, min(split_y - prev_bound - 1, next_bound - split_y - 1) // 2)
        h_ov = min(half_overlap, max_half)

        if h_ov == 0:
            lo = max(prev_bound + 1, split_y - tol)
            hi = min(next_bound - 1, split_y + tol)
            cut = _snap_y(split_y, lo, hi, profile, y_scale)
            ends[idx] = cut
            starts[idx + 1] = cut
        else:
            # Bottom edge of band `idx`: >= split_y so bands always overlap with zero dropped rows.
            end_target = min(next_bound - 1, split_y + h_ov)
            end_lo = max(split_y, end_target - tol)
            end_hi = min(next_bound - 1, end_target + tol)
            ends[idx] = _snap_y(end_target, end_lo, end_hi, profile, y_scale)

            # Top edge of band `idx + 1`: <= split_y so bands always overlap cleanly.
            start_target = max(prev_bound + 1, split_y - h_ov)
            start_lo = max(prev_bound + 1, start_target - tol)
            start_hi = min(split_y, start_target + tol)
            starts[idx + 1] = _snap_y(start_target, start_lo, start_hi, profile, y_scale)

    bands: list[tuple[Image.Image, tuple[int, int, int, int]]] = []
    for y0, y1 in zip(starts, ends, strict=True):
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))
        box = (0, y0, w, y1)
        bands.append((image.crop(box), box))
    return bands
