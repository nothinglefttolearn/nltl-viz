# Motion is a Preset field, not a separate `--motion` flag

`Motion` (`deform`/`rigid`/`pulse`) used to be an independent `--motion` CLI flag, combined freely with whichever `--preset` was chosen. In practice this meant reasoning about two independently-varying things at once — and a preset's own amplitude fields (`deform_amplitude`, `scale_amplitude`, `pulse_max_opacity`) only make sense together with the one Motion they were actually tuned for; the flag let you pick a mismatched combination (e.g. an aggressive preset's `deform_amplitude` values with `--motion rigid`, silently ignoring them) with no feedback that it was mismatched.

We moved `motion` into `Preset` itself and removed `--motion` entirely. A preset now fully determines its look, motion included — consistent with how every other per-motion knob was already a preset field, not a flag. `nltl-viz presets` lists each preset's motion so it stays visible without needing `--motion` back as a discovery aid.

The tradeoff: a single preset can no longer serve two motions via a flag. If you want an existing preset's tuning (colors, grain, attack/release) under a different motion, that's a new named preset (or a `--config` override), not a flag combination. We accepted this deliberately — the combinatorial flexibility was the source of the confusion, not a feature worth preserving.

`Shape` (`--shape face|space`) stays a separate flag — a shape doesn't carry motion-specific amplitude tuning the way a preset does, so there's no equivalent mismatch to resolve.
