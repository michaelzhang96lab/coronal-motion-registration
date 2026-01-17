"""
timelapse-motion-tracking-demo
Motion tracking on SOHO LASCO C2 (CME/coronal motion) time-lapse GIF.

Two modes are supported:

1) Global motion (phase correlation):
   - Estimates a global translation between consecutive frames using FFT phase correlation.
   - Fast, robust, but the corona is not a rigid object, so results are approximate.

2) Feature tracking (template matching):
   - Tracks a chosen rectangular ROI (template) from the first frame across time.
   - Uses normalized cross-correlation (cv2.matchTemplate).
   - Produces a visually compelling overlay GIF (moving box + trajectory trail).

This script is designed to be:
- compact and reproducible,
- readable for review (docstrings + explanatory comments),
- useful as a portfolio demonstration for research roles involving image analysis.

Author: (your name)
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import cv2
import imageio
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

# Note: The SOHO "LATEST/tinyc2.gif" is often a single-frame GIF.
# For motion tracking, you want a true multi-frame movie GIF (48 hours).
DEFAULT_GIF_URL = "https://soho.nascom.nasa.gov/data/LATEST/current_c2small.gif"


@dataclass
class MotionResult:
    """Container for one frame-to-frame motion estimate."""
    frame_i: int
    dx_pixels: float
    dy_pixels: float
    score: float  # response for phase correlation, or match score for feature tracking


# -----------------------------
# Basic utilities
# -----------------------------

def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it does not already exist."""
    os.makedirs(output_dir, exist_ok=True)


def download_file(url: str, out_path: str) -> None:
    """
    Download a file from `url` to `out_path` (streaming download).

    Streaming avoids holding the whole file in memory.
    """
    print(f"[INFO] Downloading: {url}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)

    print(f"[INFO] Saved to: {out_path}")


def load_gif_frames(gif_path: str, max_frames: int) -> List[np.ndarray]:
    """
    Load frames from an animated GIF.

    IMPORTANT:
    Some SOHO 'LATEST/*.gif' resources contain only ONE frame.
    Feature tracking and motion estimation require multiple frames.

    Returns:
        A list of frames as uint8 RGB images with shape (H, W, 3).
    """
    frames = imageio.mimread(gif_path)

    if len(frames) == 0:
        raise ValueError("No frames found in GIF.")

    if len(frames) < 2:
        raise ValueError(
            "This GIF contains only ONE frame.\n"
            "You likely downloaded a 'latest image' GIF rather than a multi-frame timelapse GIF.\n"
            "Please use the SOHO 'Real Time GIF Movies' LASCO C2 (48 hours animated GIF) instead.\n"
        )

    # Limit frames for faster demo runs
    frames = frames[:max_frames]

    rgb_frames: List[np.ndarray] = []
    for f in frames:
        # Many GIF frames are RGBA -> convert to RGB for consistency
        if f.shape[-1] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_RGBA2RGB)
        rgb_frames.append(f.astype(np.uint8))

    print(f"[INFO] Loaded {len(rgb_frames)} frames from GIF.")
    return rgb_frames


def preprocess_gray(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a frame to grayscale float32 and apply mild stabilization preprocessing.

    Steps:
    - RGB -> grayscale
    - small Gaussian blur to reduce sensor noise / compression artefacts
    - subtract mean to improve correlation stability

    Returns:
        float32 grayscale image (H, W)
    """
    gray_u8 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    gray_f32 = gray_blur.astype(np.float32)

    # Normalize for correlation-based methods
    gray_f32 = gray_f32 - np.mean(gray_f32)
    return gray_f32


def save_first_frame_png(frames_rgb: List[np.ndarray], out_path: str) -> None:
    """
    Save the first frame as a PNG for manual ROI selection.

    This is useful when you cannot (or do not want to) use interactive ROI selection.
    """
    first = frames_rgb[0]
    bgr = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    print(f"[INFO] Exported first frame for ROI selection: {out_path}")


# -----------------------------
# Motion estimation: global translation
# -----------------------------

def estimate_translation_phasecorr(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Estimate global translation between img1 and img2 using phase correlation.

    Args:
        img1: grayscale float32 image (reference)
        img2: grayscale float32 image (moving)

    Returns:
        dx, dy, response
        dx/dy are in pixels; response is a correlation quality score.
    """
    shift, response = cv2.phaseCorrelate(img1, img2)
    dx = float(shift[0])
    dy = float(shift[1])
    return dx, dy, float(response)


# -----------------------------
# Motion estimation: feature tracking via template matching
# -----------------------------

def parse_roi(roi_vals: Optional[List[int]]) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse ROI passed from CLI.

    ROI format:
        x y w h
    """
    if roi_vals is None:
        return None
    if len(roi_vals) != 4:
        raise ValueError("ROI must contain exactly four integers: x y w h")
    x, y, w, h = [int(v) for v in roi_vals]
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be > 0")
    return x, y, w, h


def clamp_roi(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Clamp ROI to be safely inside the image.
    """
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def track_feature_template_matching(
    frames_gray: List[np.ndarray],
    roi: Tuple[int, int, int, int],
    search_margin: int = 40
) -> Tuple[List[MotionResult], List[Tuple[int, int, int, int]]]:
    """
    Track a feature across frames using template matching.

    Algorithm summary:
    - Extract a template from the first frame using ROI.
    - For each subsequent frame:
        - search for the best match around the previous position (local search window)
        - update the ROI position
        - store dx/dy relative to the previous frame

    Args:
        frames_gray: list of grayscale float32 frames
        roi: (x, y, w, h) template rectangle in the first frame
        search_margin: pixels around last known position to search (smaller = faster)

    Returns:
        motion_results: list of dx/dy between frames
        roi_positions: ROI rectangle per frame (for visualization)
    """
    h_img, w_img = frames_gray[0].shape[:2]
    x, y, w, h = clamp_roi(*roi, img_w=w_img, img_h=h_img)

    # Template taken from frame 0 (float32)
    template = frames_gray[0][y:y+h, x:x+w].copy()

    # Store ROI positions for each frame so we can draw the moving box + trail
    roi_positions: List[Tuple[int, int, int, int]] = [(x, y, w, h)]

    motion_results: List[MotionResult] = []

    # Current ROI location
    cur_x, cur_y = x, y

    for i in range(len(frames_gray) - 1):
        # We will find the location of the template in the NEXT frame (i+1)
        next_frame = frames_gray[i + 1]

        # Define a local search window around current ROI to avoid false matches
        x0 = max(0, cur_x - search_margin)
        y0 = max(0, cur_y - search_margin)
        x1 = min(w_img, cur_x + w + search_margin)
        y1 = min(h_img, cur_y + h + search_margin)

        search_region = next_frame[y0:y1, x0:x1]

        # Template matching result map:
        # For TM_CCOEFF_NORMED, higher score is better (range ~[-1, 1])
        result_map = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result_map)

        # max_loc is relative to the search_region top-left
        best_x = x0 + max_loc[0]
        best_y = y0 + max_loc[1]

        dx = float(best_x - cur_x)
        dy = float(best_y - cur_y)

        # Update current ROI for the next iteration
        cur_x, cur_y = best_x, best_y
        cur_x, cur_y, w, h = clamp_roi(cur_x, cur_y, w, h, img_w=w_img, img_h=h_img)

        roi_positions.append((cur_x, cur_y, w, h))

        motion_results.append(MotionResult(
            frame_i=i,
            dx_pixels=dx,
            dy_pixels=dy,
            score=float(max_val)
        ))

        print(f"[FRAME {i:02d}] feature dx={dx:+.2f} dy={dy:+.2f} match={max_val:.3f}")

    return motion_results, roi_positions


# -----------------------------
# Visualization helpers
# -----------------------------

def draw_global_arrow(frame_rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Draw a global motion arrow on the image.
    """
    h, w = frame_rgb.shape[:2]
    start = (int(0.15 * w), int(0.85 * h))
    scale = 3.0
    end = (int(start[0] + scale * dx), int(start[1] + scale * dy))

    annotated = frame_rgb.copy()
    bgr = (0, 0, 255)  # red

    cv2.arrowedLine(annotated, start, end, bgr, thickness=2, tipLength=0.25)
    label = f"global dx={dx:+.2f}, dy={dy:+.2f} px"
    cv2.putText(annotated, label, (start[0], start[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
    return annotated


def draw_feature_box_and_trail(
    frame_rgb: np.ndarray,
    roi: Tuple[int, int, int, int],
    trail_centers: List[Tuple[int, int]],
    match_score: float
) -> np.ndarray:
    """
    Draw a tracked feature box and its trajectory trail.

    Args:
        frame_rgb: RGB image
        roi: (x, y, w, h) for current frame
        trail_centers: list of feature centers accumulated up to current frame
        match_score: template matching correlation score for current update
    """
    x, y, w, h = roi
    annotated = frame_rgb.copy()

    # Draw ROI rectangle
    bgr_box = (0, 255, 0)  # green box
    cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_box, 2)

    # Compute and store the center of the ROI
    cx = int(x + w / 2)
    cy = int(y + h / 2)

    # Draw trail as small circles and lines
    bgr_trail = (255, 0, 0)  # blue trail
    for i in range(1, len(trail_centers)):
        cv2.line(annotated, trail_centers[i - 1], trail_centers[i], bgr_trail, 2)
    for pt in trail_centers:
        cv2.circle(annotated, pt, 3, bgr_trail, -1)

    # Label
    text = f"feature match={match_score:.3f}"
    cv2.putText(annotated, text, (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_box, 1, cv2.LINE_AA)

    return annotated


def plot_motion_series(results: List[MotionResult], out_path: str, title: str) -> None:
    """
    Plot dx/dy time series and save to disk.
    """
    idx = [r.frame_i for r in results]
    dxs = [r.dx_pixels for r in results]
    dys = [r.dy_pixels for r in results]

    plt.figure(figsize=(9, 4.5))
    plt.plot(idx, dxs, label="dx (pixels)")
    plt.plot(idx, dys, label="dy (pixels)")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Frame index (i -> i+1)")
    plt.ylabel("Estimated motion (pixels)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_results_csv(results: List[MotionResult], out_path: str, mode: str) -> None:
    """
    Save results as CSV.

    Column 'score' means:
    - phasecorr: response score
    - feature: template matching correlation score
    """
    df = pd.DataFrame([{
        "mode": mode,
        "frame_i": r.frame_i,
        "dx_pixels": r.dx_pixels,
        "dy_pixels": r.dy_pixels,
        "score": r.score
    } for r in results])

    df.to_csv(out_path, index=False)


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    gif_url: str,
    max_frames: int,
    output_dir: str,
    mode: str,
    roi_vals: Optional[List[int]],
    export_first_frame: int
) -> None:
    """
    End-to-end pipeline:
    - download movie GIF
    - extract frames
    - compute motion (global or feature tracking)
    - save CSV + plot + overlay GIF
    """
    ensure_output_dir(output_dir)

    gif_path = os.path.join(output_dir, "input.gif")
    csv_path = os.path.join(output_dir, "results.csv")
    plot_path = os.path.join(output_dir, "motion_plot.png")
    overlay_gif_path = os.path.join(output_dir, "tracked_overlay.gif")
    first_frame_path = os.path.join(output_dir, "first_frame.png")

    # 1) Download data
    download_file(gif_url, gif_path)

    # 2) Load frames
    frames_rgb = load_gif_frames(gif_path, max_frames=max_frames)

    # Optional: export first frame for manual ROI choice
    if export_first_frame == 1:
        save_first_frame_png(frames_rgb, first_frame_path)

    # 3) Preprocess grayscale frames once
    frames_gray = [preprocess_gray(f) for f in frames_rgb]

    overlay_frames: List[np.ndarray] = []

    if mode == "phasecorr":
        # Global motion estimation between consecutive frames
        results: List[MotionResult] = []

        for i in range(len(frames_gray) - 1):
            dx, dy, response = estimate_translation_phasecorr(frames_gray[i], frames_gray[i + 1])
            results.append(MotionResult(frame_i=i, dx_pixels=dx, dy_pixels=dy, score=response))

            annotated = draw_global_arrow(frames_rgb[i + 1], dx, dy)
            overlay_frames.append(annotated)

            print(f"[FRAME {i:02d}] global dx={dx:+.2f} dy={dy:+.2f} response={response:.3f}")

        save_results_csv(results, csv_path, mode="phasecorr")
        plot_motion_series(results, plot_path, title="SOHO LASCO C2 Global Motion (Phase Correlation)")
        imageio.mimsave(overlay_gif_path, overlay_frames, duration=0.35)

    elif mode == "feature":
        # Feature tracking using template matching on a specified ROI
        roi = parse_roi(roi_vals)
        if roi is None:
            raise ValueError(
                "Feature tracking requires an ROI. Please pass: --roi x y w h\n"
                "Tip: run with --export_first_frame 1 to generate outputs/first_frame.png."
            )

        results, roi_positions = track_feature_template_matching(frames_gray, roi, search_margin=45)

        # Build a visually strong overlay GIF: moving box + trajectory trail
        trail_centers: List[Tuple[int, int]] = []
        for i in range(1, len(roi_positions)):
            x, y, w, h = roi_positions[i]
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            trail_centers.append((cx, cy))

            match_score = results[i - 1].score
            annotated = draw_feature_box_and_trail(frames_rgb[i], roi_positions[i], trail_centers, match_score)
            overlay_frames.append(annotated)

        save_results_csv(results, csv_path, mode="feature")
        plot_motion_series(results, plot_path, title="SOHO LASCO C2 Feature Tracking (Template Matching)")
        imageio.mimsave(overlay_gif_path, overlay_frames, duration=0.35)

    else:
        raise ValueError("Invalid mode. Use --mode phasecorr or --mode feature")

    print("\n[INFO] Done.")
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] Plot saved to: {plot_path}")
    print(f"[INFO] Overlay GIF saved to: {overlay_gif_path}")
    if export_first_frame == 1:
        print(f"[INFO] First frame exported to: {first_frame_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Motion tracking demo using SOHO LASCO C2 timelapse GIF (global phase correlation or feature tracking)."
    )
    parser.add_argument(
        "--gif_url",
        type=str,
        default=DEFAULT_GIF_URL,
        help="URL of the animated GIF movie (must contain multiple frames)."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Number of frames to use from the GIF (smaller = faster)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Folder for outputs (CSV, plot, overlay GIF)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="phasecorr",
        choices=["phasecorr", "feature"],
        help="Tracking mode: phasecorr (global) or feature (template matching)."
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=None,
        help="ROI for feature tracking: x y w h (pixels). Required if --mode feature."
    )
    parser.add_argument(
        "--export_first_frame",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, saves outputs/first_frame.png for ROI selection."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        gif_url=args.gif_url,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        mode=args.mode,
        roi_vals=args.roi,
        export_first_frame=args.export_first_frame
    )