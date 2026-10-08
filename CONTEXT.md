# NLTL Viz

Generates an audio-reactive polygon animation from a music file (or overlays it onto a video), driven entirely by Python-side audio analysis and Cairo rendering.

## Language

**Preset**:
The motion-tuning axis: a named set of amplitude, timing, onset, and post-fx numbers (plus the Motion they were tuned for), chosen via `--preset`. Independent of Palette and Layout — a Preset carries no color and no canvas-arrangement information of its own.
_Avoid_: style, look (too vague — a Preset is specifically motion tuning, not the whole visual identity)

**Palette**:
The color axis: a named or custom accent-color scheme (background, outline, and flash colors) chosen via `--palette`, independent of a Preset's motion tuning. The built-in `black` palette keeps a two-color (violet→teal) flash lerp by spectral centroid; other built-in palettes are single-accent — monochrome, no lerp.
_Avoid_: theme, colorway, scheme

**Layout**:
The canvas-arrangement axis, chosen via `--layout`: `single` (one shape centered in the frame) or `grid` (frame split into a fixed 2×2 with 4 identical copies of the shape, all driven by the same full-spectrum signal — not a spectral split per quadrant). Independent of Preset and Palette.
_Avoid_: arrangement, mode

**Motion**:
How the shape (see `nltl-viz/src/nltl_viz/render.py`'s `Shape`) responds to audio over time. Fixed per preset — not a separate flag — since a preset's amplitude/opacity fields only make sense together with the one Motion they were tuned for.
_Avoid_: animation style, effect (as generic synonyms — name the Motion)

**Pulse** (a Motion):
The shape stays a constant size and proportion — no deformation, no scaling — drawn as one solid, flat-colored silhouette. Audio reactivity is expressed entirely as that fill's opacity, floor 0%, ceiling always below 100% (`pulse_max_opacity`). Has no flash element and no bass/treble color-lerp — those belong to `deform`/`rigid` only.
_Avoid_: breathing, throbbing (as synonyms for this Motion specifically — describe the opacity mechanic, not an impression of it)
