# Coronal Motion Registration: Timelapse Alignment + Feature Tracking on SOHO LASCO C2

This repository provides a compact, reproducible pipeline for **quantitative motion estimation** in real observational timelapse imagery, using SOHO LASCO C2 coronal data as a test case.

The implementation is intentionally small, but the focus is methodological:

- correlation-based motion estimation as a **registration primitive**
- feature tracking as a **local matching strategy**
- explicit outputs (CSV + diagnostic plot + annotated animation)
- clear assumptions and limitations (non-rigid scene, intensity evolution)

Although the dataset is solar coronagraph imagery, the same workflow design applies to other research contexts where the objective is to extract **repeatable quantitative measurements** from imperfect multi-frame observations (e.g., imaging experiments, sensor time series, multi-modal registration problems).

---

## What this project demonstrates

This repository is not intended as “image processing for its own sake”. It is a small exhibition of a broader research practice:

- framing the task as a measurement / alignment problem  
- choosing methods consistent with data properties  
- implementing a reproducible pipeline with explicit diagnostics  
- reporting results numerically rather than visually only  

In my doctoral work (engineering metrology and statistical alignment), the recurring challenge was deriving reliable quantitative relationships between datasets obtained under real-world constraints (partial overlap, varying sampling, modality differences). This demo uses a different data type (timelapse imagery), but follows the same methodological logic.

---

## Data source

**SOHO LASCO C2** real-time imagery (48-hour animated GIF movies are commonly available).

Example links:
- SOHO Real-Time Images & Movies: [https://soho.nascom.nasa.gov/data/LATEST](https://soho.nascom.nasa.gov/data/LATEST)  
- A valid input must be a **multi-frame animated GIF movie** (single-frame “latest image” GIFs are not usable).

---

## Methods overview

Two complementary approaches are implemented:

### 1) Global translation estimate (Phase Correlation)
A fast FFT-based method that estimates a **global frame-to-frame shift**:
- robust to moderate illumination change
- useful as a coarse registration primitive
- does not require feature detection

Output per frame pair: `(dx_global, dy_global)` and a correlation response score.

### 2) Feature tracking (Template Matching)
A local tracking method that follows a selected patch (ROI) through time:
- produces an interpretable moving bounding box (“wow-effect” visual)
- provides local displacement `(dx_feature, dy_feature)` and match confidence score
- uses a constrained search window for stability and speed

---

## Outputs

A single run produces:

- `outputs/results.csv`  
  per-frame motion estimates (global and feature-tracking metrics)

- `outputs/motion_plot.png`  
  diagnostic plot of estimated motion across the sequence

- `outputs/tracked_overlay.gif`  
  annotated timelapse with:
  - global motion arrow
  - feature bounding box (tracking)
  - optional trajectory trail

---

## Installation

```bash
pip install -r requirements.txt
