---
name: reverse-caption-style
description: Measure burned-in captions in a reference video and produce a reusable style preset for new words. Use when someone wants to match subtitle typography, inline emphasis, placement, phrase grouping or reveal timing from footage.
---

# Reverse caption style

Start with the reference clip. The default deliverable is a reusable JSON preset, a short evidence report and cropped caption samples. Preserve the target speaker's words when applying the style to different footage.

The analyzer is standalone. HyperFrames and Caption Studio are optional consumers of the preset. Running it requires a local video with visible captions, FFmpeg and ffprobe, Tesseract with the appropriate language pack, and Python 3.10+ with Pillow and NumPy. Fontconfig improves local font matching. Rendering captioned video or transparent overlays requires a separate renderer.

## Run the fast pass

With FFmpeg, ffprobe and Tesseract on PATH, run from this skill's directory:

```sh
uv run --no-project --with 'pillow>=10.1,<14' --with 'numpy>=1.26,<3' python scripts/analyze.py reference.mp4 --out caption-study
```

Alternatively install Pillow and NumPy in a virtual environment and run the same script with Python 3.10+. Fontconfig's `fc-match` supplies local font candidates. For a suspected brand typeface installed on the machine, set `CAPTION_FONT_CANDIDATES` to one or more comma-separated family names before launching the helper. Tesseract defaults to English. For another language, install that Tesseract language pack and set `CAPTION_OCR_LANG` to its code. The skill installs in another workspace by copying the entire `reverse-caption-style` folder into its agent's skills directory.

The helper seeks to at most thirteen points across the displayed video and samples a few short windows in the caption band. It reads pixels locally, skipping audio transcription and full-film rendering. Inspect `preview.jpg` before accepting `preset.json`. `report.json` includes the sampled times, fitted words, motion observations, measurement resolution and limits.

If a scene contains product labels, game UI, several subtitle languages or a persistent watermark, use `--roi x,y,width,height` with normalized values relative to the displayed frame. For example, `--roi 0.05,0.66,0.9,0.25` examines a lower band. On a miss the helper emits the sampled frames and diagnostic report. A narrower crop, a clip with more caption changes, or a better OCR language can resolve it.

## Check what actually defines the style

- Compare two ordinary phrases and the strongest emphasis moment. Measure the width and ink height of the same words in a preview. Report the font as the closest installed match. A composite video's glyphs cannot establish the source font file.
- Look for a role change *within* a line: type family, size, weight, italics, spacing or fill. Distinguish a one-off large title from a recurring spoken-word treatment. If sparse samples miss an italic aside, highlight or alternate display face, rerun with `--at 8.24` (repeat for up to three specific moments). Compare the targeted crops and revise that role alone. One preset carries three roles; describe additional variants in the report.
- Read a short run of adjacent frames at a change. A phrase cut replaces all words together. A reveal keeps preceding words in place while another appears. A pop changes the size of the same glyphs. Karaoke fills, wipes, gradients, strokes, shadows, backplates and bouncing baselines need their own measurements; record any unsupported effect and its timing in the report.
- When an effect is ambiguous, isolate the caption band around a single transition. Substitute measured pixel bounds and start time in this example:

```sh
ffmpeg -ss 4.0 -i reference.mp4 -an -vf "crop=640:140:40:480,scale=320:-1,tile=4x3" -frames:v 1 caption-transition.png
```

The twelve cells are consecutive native frames. Compare the visible switch frame and ink boxes over time. Verify a rendered test phrase beside the matching reference crop if the user asks for replication.

## Apply the preset

`x` and `y` are normalized caption-block center coordinates. Each of `normal`, `emphasis` and `hero` supplies a font, size and tracking (both fractions of canvas width), bold/italic flags and an RGB hex color. A role without visual evidence inherits ordinary text. Use `phrase`, `reveal` or `pop` for the measured animation. Keep any supplemental effect notes with the preset. A new placement explicitly requested for the target overrides the measured anchor.

For different speech, choose emphasis from the meaning as well as the pacing. Give the subject's first mention or a strong contrast the larger role if the reference uses one. Give a material, number or descriptive word the inline role where the speaker stresses it. Let connective words stay in ordinary type. Preserve a phrase's baseline when two faces share a line.

Import `preset.json` in a compatible editor or use it as a renderer-neutral recipe. The helper also exposes `analyze(path, out_dir, roi=None, at=None)`, which returns a mapping containing the preset and report.
