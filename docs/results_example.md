# Example Results

A typical run generates:

- `outputs/results.csv` containing frame-to-frame motion estimates
- `outputs/motion_plot.png` with dx/dy series
- `outputs/tracked_overlay.gif` annotated output

## What to look for
- Global motion estimates should remain relatively small if the image sequence is stable.
- Feature tracking often shows larger motion if you select a distinct evolving coronal structure.
- Match score values help identify when a tracked feature is no longer consistent.

## Practical note
For the most interpretable overlay animation, select an ROI on a bright structured feature
outside the occulting disk edge, and avoid smooth background regions.
