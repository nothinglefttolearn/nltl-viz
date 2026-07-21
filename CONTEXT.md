# NLTL Viz

Generates an audio-reactive polygon animation from a music file (or overlays it onto a video), driven entirely by Python-side audio analysis and Cairo rendering.

## Language

**Motion**:
How the shape (see `nltl-viz/src/nltl_viz/render.py`'s `Shape`) responds to audio over time. Fixed per preset — not a separate flag — since a preset's amplitude/opacity fields only make sense together with the one Motion they were tuned for.
_Avoid_: animation style, effect (as generic synonyms — name the Motion)

**Pulse** (a Motion):
The shape stays a constant size and proportion — no deformation, no scaling — drawn as one solid, flat-colored silhouette. Audio reactivity is expressed entirely as that fill's opacity, floor 0%, ceiling always below 100% (`pulse_max_opacity`). Has no flash element and no bass/treble color-lerp — those belong to `deform`/`rigid` only.
_Avoid_: breathing, throbbing (as synonyms for this Motion specifically — describe the opacity mechanic, not an impression of it)
