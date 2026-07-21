# nltl-viz

A CLI tool for generating audio-reactive music visualization video clips from a short audio file. Fully generative — no image input. The chosen shape (the NLTL face, or NLTL space — its inverse) reacts to the track according to its preset's motion style — see [Motion styles](#motion-styles) below.

Point it at a video file instead of an audio file and it switches to overlay mode: the same visualization, transparent everywhere except the shape/flash, composited on top of the video using the video's own audio track for analysis.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python version and dependencies)
- [ffmpeg](https://ffmpeg.org) in your PATH (`ffprobe`, installed alongside it, is required too — used to read a source video's resolution/frame rate for overlay mode)
- Cairo + pkg-config (pycairo builds against system Cairo): `brew install cairo pkg-config` on macOS

## Install

```bash
cd nltl-viz
uv sync
```

## Usage

```
nltl-viz [audio|video] [flags]
```

```bash
# Preview 10 seconds before committing to a full render
nltl-viz --preset industrial --preview demo.wav

# Full render
nltl-viz --preset aggressive demo.wav

# Custom preset from a config file
nltl-viz --config nltl-viz.yaml --preset my-preset demo.wav

# Render the inverse shape, NLTL space
nltl-viz --shape space demo.wav

# Overlay onto a video file instead — same flags, auto-detected by extension
nltl-viz --preset aggressive performance.mp4

# Interactive mode — prompts for audio file, preset, and shape
nltl-viz
```

Motion (`deform`/`rigid`/`pulse`) isn't a flag — it comes from whichever preset you pick. Run `nltl-viz presets` to see each preset's motion.

Output files are saved alongside the input file, named `{name}_viz_{timestamp}.mp4` for an audio input (or `{name}_viz_preview_{timestamp}.mp4` for `--preview`), and `{name}_viz-overlay_{timestamp}.mp4` for a video input (`{name}_viz-overlay_preview_{timestamp}.mp4` for `--preview`) — each render gets its own file so re-running with different settings never overwrites a previous output.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--preset`, `-p` | `industrial` | Visual preset — also determines motion (`deform`/`rigid`/`pulse`), see below |
| `--shape` | `face` | Shape to visualize: `face` or `space` (its inverse) |
| `--preview` | `false` | Render a 10s low-quality preview to check the look |
| `--output-dir`, `-o` | same as input | Where to write the output file |
| `--config`, `-c` | — | YAML file with custom presets |
| `--verbose`, `-v` | `false` | Show raw ffmpeg output during render |

## Presets

| Name | Motion | Description |
|------|--------|--------------|
| `industrial` | `deform` | Moderate deform, medium onset sensitivity |
| `subtle` | `deform` | Calmer, slower breathing, lighter grain/vignette |
| `3d-glasses` | `deform` | Red bass color, cyan treble color, light background, dark outline |
| `aggressive` | `deform` | Punchier deform and flash, faster attack, heavier grain/vignette |

Run `nltl-viz presets` to list them (with their motion). A preset's own `motion` field is what a `--config` YAML entry uses to select `deform`/`rigid`/`pulse` — see `nltl-viz.yaml.example`.

## Custom Presets

Copy `nltl-viz.yaml.example` to `nltl-viz.yaml` and edit the values — see that file for a full field reference. Custom presets fall back to built-ins if the name isn't found in the config file.

## Shapes

- `face` (default) — the NLTL face, inscribed in a bounding square with the intentional blank space to its right preserved.
- `space` — the same bounding square minus the NLTL face: the complementary shape, sharing the face's diagonal and short vertical edge as its own boundary.

## Motion styles

Motion isn't a flag — each preset's `motion` field fixes which one it uses (see `nltl-viz presets`).

- `deform` — the chosen shape's perimeter is divided into 32 frequency bands, distributed proportionally to each edge's actual length (so the long edges get more control points than the short ones), smoothed frame-to-frame with an attack/release envelope so the shape breathes rather than vibrates, and interpolated into one continuous deformed outline. The shape deforms outward/inward from its own centroid, not the canvas center, so it stays recognizable as it moves.
- `rigid` — the shape's proportions never distort; instead the whole shape scales uniformly larger and smaller around its own centroid, driven by the track's overall loudness (an RMS envelope, smoothed the same way as the band energies). The outer shape is a solid flat fill instead of a stroked outline, and the reactive element is a smaller solid-filled concentric copy of the same shape instead of a soft gradient — flat, hard-edged regions throughout, no gradients.
- `pulse` — the shape is drawn at a genuinely constant size and proportion (no deform, no scale animation), inscribed edge-to-edge in the frame's shorter dimension rather than `deform`/`rigid`'s smaller centered scale. The only reactive element is the fill's own opacity, driven directly by the same overall-loudness envelope `rigid` uses for scale — floor 0%, ceiling always below 100% (`pulse_max_opacity`). No flash, no bass/treble color blend — one flat color (`outline_color`) throughout.

## How it reacts to audio

- Each detected onset (a "hit") triggers the reactive element (a soft gradient blob in `deform`, a smaller solid copy of the shape in `rigid`) with a fixed-duration decay; a harder hit is brighter/larger, but every flash fades at the same rate. `pulse` has no onset-triggered flash at all.
- Its color is a live blend between two configurable colors, driven by the track's spectral centroid — bass-heavy moments skew toward `bass_color`, treble-heavy moments skew toward `treble_color`. `pulse` uses neither — it's one flat `outline_color` throughout.
- Grain and vignette are applied as a post-process over every frame (no image input, no desaturation pass — the palette is deliberately muted already).

## Video overlay

Passing a video file (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.webm`) instead of an audio file switches to overlay mode automatically — same flags, same presets, same shape/motion logic, no separate command:

```bash
nltl-viz --preset aggressive performance.mp4
```

What's different from audio-only mode:

- The video's audio track is what gets analyzed (extracted to a temp WAV internally, then discarded) — there's no separate audio file to supply.
- The canvas renders at the source video's own resolution rather than a fixed square, with the shape sized and centered off the frame's shorter dimension — a landscape 1920x1080 clip gets a centered square-proportioned shape, not a stretched one. (`pulse` in particular is designed for this: it fills that shorter dimension edge-to-edge, where `deform`/`rigid` stay at their smaller centered scale.)
- Analysis, rendering, and encoding all run at the video's own probed frame rate (read via `ffprobe`), not a fixed 30fps, so viz frames line up 1:1 with video frames.
- The shape/flash render fully opaque exactly as in audio-only mode; everything else is transparent, so the source video shows through untouched everywhere the shape isn't. Grain and vignette still run on every frame as in audio-only mode — including the transparent area — for one continuous texture across the whole composited frame rather than a hard dropoff at the shape's silhouette.
- The output's audio is the source video's own audio track, copied through unchanged (no re-encode) rather than re-compressed.
- Interactive mode (running `nltl-viz` with no arguments) only scans for audio files — overlay mode is invoked by passing a video path directly on the command line.

## Components

The pipeline runs as four independent stages, each owned by a different library, so audio analysis and rendering aren't constrained by what an ffmpeg filter graph can express:

| Stage | File | Library | Role |
|-------|------|---------|------|
| Audio analysis | `audio.py` | [librosa](https://librosa.org) | Decodes the audio file once, up front, and produces one value per *output video frame* (not per audio sample) for: band energy per frequency band (STFT), onset strength/timing, spectral centroid, and an RMS loudness envelope. |
| Frame rendering | `render.py` | [pycairo](https://pycairo.readthedocs.io) | Draws the shape outline/fill and flash for each frame from the arrays `audio.py` precomputed — this is the only stage that draws anything. In overlay mode it renders onto the source video's own canvas size with the background left unpainted, returning premultiplied RGBA instead of opaque RGB24. |
| Post-processing | `postprocess.py` | [numpy](https://numpy.org) | Applies grain and vignette to each rendered frame as array operations on the raw RGB buffer (alpha, when present, passes through untouched). |
| Video I/O | `video.py` | [ffmpeg](https://ffmpeg.org)/`ffprobe` (subprocess) | Overlay-mode only: probes a source video's resolution/frame rate, extracts its audio to a temp WAV for analysis, decodes its frames to raw RGB, and alpha-composites each rendered viz frame over the corresponding video frame (`fg_premultiplied + bg * (1 - alpha)`). |
| Encoding | `encode.py` | [ffmpeg](https://ffmpeg.org) (subprocess) | Muxes the piped raw RGB24 frames (composited, in overlay mode) with the audio track — the original audio file re-encoded to AAC in audio-only mode, or the source video's own audio track stream-copied unchanged in overlay mode. ffmpeg does no filtering or effects work — it's a dumb encoder/decoder, all image work already happened in Python. |

Supporting modules: `preset.py`/`config.py` define and load the numeric fields (`deform_amplitude`, `flash_decay_ms`, colors, etc.) that `audio.py` and `render.py` read; `cli.py` (built on [typer](https://typer.tiangolo.com)) wires the stages together per-frame, dispatches to audio-only or overlay mode based on the input file's extension, and exposes the flags documented above; `interactive.py` (built on [questionary](https://questionary.readthedocs.io)) is the prompt-driven entry point used when no file is passed on the command line — audio files only.

### What reacts to audio, and where it's computed

All audio reactivity is resolved once in `audio.py` before any frame is drawn — `render.py` only ever reads precomputed per-frame values, it does no audio analysis of its own:

- Shape motion — `deform` reads per-band energy; `rigid` and `pulse` both read the RMS loudness envelope (`rigid` for scale, `pulse` for fill opacity).
- Flash brightness — reads onset strength/timing (decayed per frame, combined across overlapping onsets with an elementwise `max`).
- Flash/gradient color — reads spectral centroid, blended between `bass_color` and `treble_color`.

## Development

```bash
uv sync --group dev
uv run pytest
```
