# Methodology

This repository implements a compact pipeline for estimating apparent motion in timelapse imagery using two complementary strategies: a **global correlation-based registration primitive** and a **local patch tracking method**. The objective is to extract reproducible quantitative measurements from observational data, while exposing diagnostic indicators that help assess reliability.

Although the demonstration dataset is SOHO LASCO C2 imagery, the methodological design is transferable to other scientific settings where observations evolve over time and where rigid motion assumptions are only approximately valid.

---

## 1. Data characteristics and problem framing

SOHO LASCO C2 timelapse imagery exhibits the following properties:

- The scene is **not rigid**: coronal structures evolve, brighten, deform, or disappear.
- Intensity can vary between frames due to physical evolution and instrument effects.
- A single “true” motion field is not directly observable; measurements are therefore interpreted as **apparent displacement** under a chosen model.

Given these constraints, the motion estimation tasks are framed as:

1. **Global apparent translation**:  
   A coarse estimate of how the full frame shifts between adjacent frames.

2. **Local feature displacement**:  
   A more interpretable estimate of how a selected structure changes position over time.

The pipeline therefore emphasises **robustness, interpretability, and diagnostics**, rather than treating the output as exact physical ground truth.

---

## 2. Preprocessing

Each frame is converted to grayscale and processed with a mild stabilisation step:

1. **RGB → grayscale**
2. **Gaussian blur** (small kernel)  
   This reduces the influence of compression artefacts and high-frequency noise.
3. **Zero-mean normalisation**  
   Subtracting the mean improves the stability of correlation-based estimators under brightness drift.

These steps are intentionally lightweight to preserve interpretability and reduce sensitivity to parameter tuning.

---

## 3. Global motion estimation via phase correlation

### 3.1 Principle
Phase correlation estimates translation between two images using Fourier transforms. In brief:

- Take the Fourier transform of two images.
- Compute the normalised cross-power spectrum.
- The inverse transform of this spectrum yields a peak at the translation offset.

This approach is attractive for compact pipelines because it is:

- fast (FFT-based),
- relatively robust to moderate uniform brightness changes,
- parameter-light (no keypoint detection or complex models required).

### 3.2 Output and diagnostic
For each consecutive frame pair `(i → i+1)`, phase correlation returns:

- `dx_global, dy_global` in pixels  
- `response_global` (correlation quality response)

The response score acts as a **diagnostic indicator**. Low responses typically indicate that a pure translation model is poorly supported, or that the scene change is too strong to approximate by global shift.

### 3.3 Interpretation
Because LASCO imagery is non-rigid, global translation should be interpreted as a **registration primitive** rather than a complete physical description. It is still useful as:

- a coarse stabilisation estimate,
- a baseline motion indicator,
- a sanity check for further modelling.

---

## 4. Feature tracking via template matching

### 4.1 Principle
Template matching tracks a rectangular patch (ROI) from one frame into the next by searching for the location that maximises similarity.

Given an ROI in the previous frame, the method:

1. extracts the ROI as a template,
2. searches for the best match in a defined window of the next frame,
3. updates the ROI position to the best match.

This repository uses **normalised cross-correlation** (`TM_CCOEFF_NORMED`) since it reduces sensitivity to global scaling of intensity.

### 4.2 Search-window constraint
Instead of searching the full image, the algorithm restricts matching to a local neighbourhood around the previously tracked location. This is important because:

- it reduces false matches in repetitive structures,
- it improves stability,
- it reduces computational cost,
- it matches the physical expectation that consecutive frames are usually similar.

### 4.3 Output and diagnostic
For each consecutive frame pair `(i → i+1)`, feature tracking yields:

- `dx_feature, dy_feature` in pixels (ROI displacement)
- `match_score` (best correlation score)

The match score provides a **confidence proxy**.  
A decreasing score often indicates one of the following:

- feature deformation or disappearance,
- the ROI was placed on an unstructured region,
- motion is too large for the search window,
- significant brightness or occlusion changes.

---

## 5. Reporting and visual auditing

The pipeline produces three types of output:

1. **Numeric record (`results.csv`)**  
   Motion estimates and diagnostics per frame.

2. **Diagnostic plot (`motion_plot.png`)**  
   Time series of motion estimates to identify trends or instability.

3. **Annotated overlay (`tracked_overlay.gif`)**  
   A human-auditable representation of what the algorithm is doing:
   - a global translation arrow,
   - a moving feature ROI,
   - a trajectory trail.

The design goal is that the results are not only reproducible, but also inspectable, allowing a reviewer to connect the numeric output to the image evidence.

---

## 6. Limitations and appropriate use

This repository is intentionally compact and therefore does not attempt to solve the full physical inference problem. Key limitations include:

- the corona is non-rigid and evolves intrinsically,
- intensity patterns are not conserved,
- the motion estimates are pixel-based rather than physically calibrated.

Accordingly, the outputs should be interpreted as **apparent displacement estimates** under simplified models, suitable for:

- quick quantitative summarisation,
- registration baseline estimates,
- prototyping and methodological demonstration.

For extensions (dense motion fields, uncertainty quantification, coordinate systems), see `docs/limitations_and_extensions.md`.

---

## 7. Notes on transferability

Although the example dataset is astrophysical, the pipeline pattern is broadly applicable:

- timelapse microscopy,
- experimental imaging sequences,
- monitoring and inspection pipelines,
- engineering measurement and alignment tasks.

In particular, the idea of combining:
- a coarse global registration estimate,
- with a local feature-level displacement estimate,
is common in multi-stage alignment and validation workflows.
