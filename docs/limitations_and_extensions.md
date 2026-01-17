# Limitations and Extensions

This repository is intentionally compact and focuses on transparent implementation.

## Known limitations
1. **Non-rigid scene**
   The solar corona evolves. New structures appear, brightness changes, and flow is not rigid.
   Therefore global translation is only an approximation.

2. **Intensity evolution**
   Both phase correlation and template matching implicitly assume a degree of intensity consistency.
   Real data violates this due to instrument effects, compression and physical changes.

3. **Pixel-based motion only**
   Results are reported in pixels per frame. Converting to physical units requires:
   - timestamp metadata per frame
   - instrument pixel scale
   - careful coordinate interpretation

4. **Single-feature tracking**
   The current feature mode tracks one ROI.
   In research contexts, multi-feature or dense-field tracking may be preferred.

---

## Extensions consistent with research-grade pipelines
If this were developed into a full analysis tool, natural next steps include:

- **Orthorectification / coordinate handling**
  Incorporate instrument geometry and map frames into a common coordinate system.

- **Robust motion field estimation**
  Use optical flow (e.g., Farnebäck), PIV-style block matching, or variational methods for dense displacement fields.

- **Outlier handling**
  Add rejection logic for low match-score steps, and re-initialise ROIs when needed.

- **Uncertainty quantification**
  Estimate confidence intervals based on score landscapes or bootstrap methods.

- **Pipeline modularity**
  Expose functions as a small Python package for re-use across datasets.

The simplified implementation here is intended to highlight core mechanics
and reproducible outputs rather than exhaustively cover all of these extensions.
