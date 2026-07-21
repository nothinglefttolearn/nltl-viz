# `pulse` Motion drops the flash/centroid-color mechanic and uses a different frame-fit scale

`deform` (per-band perimeter distortion) and `rigid` (uniform scale-from-loudness) both looked wrong composited over video at the small, centered scale (`_BASE_HALF_FRACTION = 0.35`) they were originally designed for as a standalone, full-canvas visual. Rather than trying to salvage either as an overlay, we added a third `Motion` — `pulse` — that keeps the shape at a genuinely constant size and proportion (no deform, no scale animation) and expresses all audio reactivity as the fill's opacity, driven directly by the existing smoothed loudness envelope (`scale_envelope`, the same signal `rigid` already used for its scaling).

Two deliberate departures from how `deform`/`rigid` work, easy to "fix" by someone assuming they were oversights:

- **Frame-fit scale.** `pulse` inscribes the shape edge-to-edge in the canvas's shorter dimension (`_PULSE_HALF_FRACTION = 0.5`), not `deform`/`rigid`'s small centered `0.35`. This only applies to `pulse` — `_polygon_vertices` takes an explicit `half_fraction` parameter now specifically so the other two motions are untouched.
- **No flash, no bass/treble centroid color.** `deform`/`rigid` both layer an onset-triggered flash (a radial gradient blob, or a smaller inner-shape copy) using a color lerped between `bass_color`/`treble_color` by spectral centroid. `pulse` draws only the one shape, in `outline_color`, and nothing else — the design goal was a single flat color reacting through opacity alone, not an additional reactive layer with its own color mechanic.

We considered making `pulse` an overlay-only code path, but kept it inside the existing `Shape`×`Motion`×`render_frame` architecture instead — it works for standalone rendering too (on the square standalone canvas, edge-to-edge is the same thing as filling the canvas), and factoring the sizing/coloring differences into existing functions was cheaper than a parallel pipeline.
