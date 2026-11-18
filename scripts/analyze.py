#!/usr/bin/env python3
"""Sample a video's baked-in captions and fit an editable style preset."""

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def run(argv, *, data=None, timeout=30):
    proc = subprocess.run(argv, check=False, input=data, capture_output=True, timeout=timeout)
    if proc.returncode:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError(f"{Path(argv[0]).name} failed: {' '.join(detail[-3:])}")
    return proc.stdout


def sample(path, t):
    # Fast input seeking bounds work independently of the reference duration.
    raw = run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{t:.6f}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-vf",
            "scale=w='min(iw,720)':h='min(ih,1080)':force_original_aspect_ratio=decrease",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "ppm",
            "-",
        ]
    )
    if not raw:
        return None
    return Image.open(io.BytesIO(raw)).convert("RGB")


def roi_box(im, roi):
    if roi is None:
        return (0, 0, *im.size)
    x, y, w, h = roi
    return (
        round(x * im.width),
        round(y * im.height),
        round((x + w) * im.width),
        round((y + h) * im.height),
    )


def ocr(im, *, psm=11):
    mode = "P5" if im.mode == "L" else "P6"
    pnm = f"{mode}\n{im.width} {im.height}\n255\n".encode() + im.tobytes()
    lang = os.environ.get("CAPTION_OCR_LANG", "eng")
    tsv = run(["tesseract", "stdin", "stdout", "-l", lang, "--psm", str(psm), "tsv"], data=pnm)
    rows = csv.DictReader(
        io.StringIO(tsv.decode(errors="replace")), delimiter="\t", quoting=csv.QUOTE_NONE
    )
    return [
        r
        for r in rows
        if r["level"] == "5"
        and r.get("text")
        and float(r["conf"]) >= 53
        and any(c.isalnum() for c in r["text"])
    ]


def layers(image):
    a = np.asarray(image, dtype=np.int16)
    lo, hi = a.min(2), a.max(2)
    yield "light", lo > 205
    yield "dark", hi < 62
    # This also reads e.g. a yellow highlighted word in an otherwise white line.
    yield "colour", ((hi - lo > 85) & (hi > 161)) | (lo > 215)


def words_in(im, box, frame, include_raw=True):
    crop = im.crop(box)
    words = []
    passes = [
        (name, Image.fromarray(np.where(mask, 0, 255).astype("uint8")))
        for name, mask in layers(crop)
        if 0.0004 < mask.mean() < 0.85
    ]
    if include_raw:
        passes.append(("natural", crop))
    for kind, img in passes:
        for r in ocr(img):
            x, y, w, h = [int(r[k]) for k in ("left", "top", "width", "height")]
            if not (4 <= h <= im.height * 0.24 and 2 <= w < im.width * 0.98):
                continue
            x += box[0]
            y += box[1]
            words.append(
                {
                    "frame": frame,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "text": r["text"].strip(),
                    "conf": float(r["conf"]),
                    "kind": kind,
                }
            )
    # Different OCR passes reading the same ink give it a single vote.
    kept = []
    for word in sorted(words, key=lambda v: -v["conf"]):
        if not any(
            abs(k["x"] - word["x"]) < 0.25 * word["w"] + 2
            and abs(k["y"] - word["y"]) < 0.28 * word["h"] + 2
            and abs(k["w"] - word["w"]) < 0.32 * word["w"] + 2
            for k in kept
        ):
            kept.append(word)
    return kept


def choose_band(words, H, W, count, manual=False):
    if not words:
        return []
    binning = max(5, round(H * 0.012))
    choices = {int((w["y"] + w["h"] / 2) // binning) * binning for w in words}
    candidates = []
    for center in choices:
        group = [w for w in words if abs(w["y"] + w["h"] / 2 - center) < H * 0.035]
        frames = defaultdict(list)
        for w in group:
            frames[w["frame"]].append(w)
        phrases, scores = [], []
        for line in frames.values():
            text = " ".join(w["text"].casefold() for w in sorted(line, key=lambda w: w["x"]))
            phrases.append(text)
            left = min(w["x"] for w in line)
            right = max(w["x"] + w["w"] for w in line)
            confidence = np.median([w["conf"] for w in line]) / 100
            scores.append(
                confidence
                * (1 + 0.12 * min(4, len(line)))
                * (1.1 - 0.4 * abs((left + right) / W - 1))
            )
        diversity = sum(1 for _, n in Counter(phrases).items() if n <= max(1, len(phrases) * 0.65))
        if not manual and (len(frames) < max(2, math.ceil(0.22 * count)) or diversity < 2):
            continue
        score = sum(scores) * min(1, 0.25 + 0.35 * diversity)
        candidates.append((score, center, group))
    if not candidates:
        return []
    _, center, primary = max(candidates, key=lambda g: g[0])
    axis = float(np.median([w["y"] + 0.5 * w["h"] for w in primary]))
    return [
        w
        for w in words
        if abs(w["y"] + 0.5 * w["h"] - axis) < max(H * 0.055, 0.65 * w["h"] + H * 0.026)
    ]


@lru_cache(maxsize=1)
def fonts():
    families = [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Avenir Next",
        "Georgia",
        "Times New Roman",
        "Baskerville",
        "DejaVu Serif",
        "Menlo",
        "Consolas",
        "Courier New",
        "DejaVu Sans Mono",
    ]
    extras = [name.strip() for name in os.environ.get("CAPTION_FONT_CANDIDATES", "").split(",")]
    families.extend(name for name in extras if name)
    found = {}
    for family in dict.fromkeys(families):
        for style in ("Regular", "Bold", "Italic"):
            if shutil.which("fc-match"):
                spec = (
                    run(
                        [
                            "fc-match",
                            "-f",
                            "%{family}|%{style}|%{file}|%{index}",
                            f"{family}:style={style}",
                        ]
                    )
                    .decode(errors="replace")
                    .split("|")
                )
                if len(spec) != 4 or not Path(spec[2]).is_file():
                    continue
                filename, index = spec[2], int(spec[3] or 0)
            else:
                convention = {"Regular": "", "Bold": " Bold", "Italic": " Italic"}[style]
                filename, index = family + convention + ".ttf", 0
            if (filename, index) in found:
                continue
            try:
                f = ImageFont.truetype(filename, 64, index=index)
            except OSError:
                continue
            found[filename, index] = f
    if not found:
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", 64)
        except OSError:
            f = ImageFont.load_default(size=64)
        found["default", 0] = f
    return list(found.values())


def candidates_for_word(rgb):
    # Cluster actual RGB samples. A background, outline and fill can all occur
    # inside a tight glyph box. Select the colour whose pixels best fit letters.
    vals = rgb.reshape(-1, 3).astype(np.int16)
    quant = vals // 32
    bins = quant[:, 0] * 64 + quant[:, 1] * 8 + quant[:, 2]
    ranked = np.argsort(np.bincount(bins, minlength=512))[::-1][:10]
    kept = []
    for c in ranked:
        cells = vals[bins == c]
        if len(cells) < max(2, len(vals) * 0.018):
            continue
        color = np.median(cells, axis=0)
        if any(np.linalg.norm(color - k) < 42 for k, _ in kept):
            continue
        mask = np.linalg.norm(rgb.astype(float) - color, axis=2) < 44
        if 0.08 < mask.mean() < 0.88:
            kept.append((color, mask))
    return kept


def fit_word(image, w):
    text = w["text"].strip()
    if not 2 <= len(text) <= 24:
        return None
    rgb = np.asarray(image.crop((w["x"], w["y"], w["x"] + w["w"], w["y"] + w["h"])))
    masks = candidates_for_word(rgb)
    best = None
    for f in fonts():
        box = f.getbbox(text)
        if not box or box[2] <= box[0] or box[3] <= box[1]:
            continue
        ideal = Image.new("L", (box[2] - box[0], box[3] - box[1]))
        ImageDraw.Draw(ideal).text((-box[0], -box[1]), text, font=f, fill=255)
        ink_box = ideal.getbbox()
        if not ink_box:
            continue
        ideal = ideal.crop(ink_box)
        aspect = (w["w"] / w["h"]) / (ideal.width / ideal.height)
        if not 0.56 < aspect < 1.7:
            continue
        letter = np.asarray(ideal.resize((w["w"], w["h"]), Image.Resampling.LANCZOS)) > 127
        for color, mask in masks:
            iou = (mask & letter).sum() / max(1, (mask | letter).sum())
            score = float(iou - 0.10 * abs(math.log(aspect)))
            if best is None or score > best["score"]:
                family, style = f.getname()
                best = {
                    "score": score,
                    "font": family,
                    "bold": "bold" in style.lower(),
                    "italic": any(s in style.lower() for s in ["italic", "oblique"]),
                    "size": w["h"] / (ideal.height / 64) / image.width,
                    "tracking": (w["w"] - ideal.width * w["h"] / ideal.height)
                    / max(1, len(text) - 1)
                    / image.width,
                    "color": color.tolist(),
                    "height": w["h"],
                    "text": text,
                    "frame": w["frame"],
                }
    return best if best and best["score"] > 0.34 else None


def style(group):
    # Pool several readable glyph runs to reduce background contamination.
    group = sorted(group, key=lambda s: s["score"], reverse=True)
    votes = Counter()
    for f in group:
        votes[(f["font"], f["bold"], f["italic"])] += f["score"] ** 3
    family, bold, italic = votes.most_common(1)[0][0]
    rgb = np.median([s["color"] for s in group], axis=0)
    tracks = [s["tracking"] for s in group if s["font"] == family]
    return {
        "font": family,
        "bold": bool(bold),
        "italic": bool(italic),
        "size": round(float(np.median([s["size"] for s in group])), 5),
        "color": "#" + "".join(f"{round(c):02x}" for c in rgb),
        "tracking": round(float(np.clip(np.median(tracks), -0.007, 0.02)), 5),
    }


def infer_roles(fits):
    typical = float(np.median([s["size"] for s in fits]))
    body = [s for s in fits if s["size"] < 1.20 * typical] or fits
    # Start from the most frequent ink cluster, so tinted keywords do not gray the body.
    counts = Counter(tuple(round(v / 45) for v in s["color"]) for s in body)
    base_key = counts.most_common(1)[0][0]
    base = [
        s for s in body if np.linalg.norm(np.array(s["color"]) - np.array(base_key) * 45) < 69
    ] or body
    normal = style(base)
    ink = np.median([s["color"] for s in base], axis=0)
    alt = [s for s in fits if np.linalg.norm(np.array(s["color"]) - ink) > 83 and s["score"] > 0.48]
    # A recurring second ink, supported by two real words, can drive inline emphasis.
    colors = Counter(tuple(round(v / 45) for v in s["color"]) for s in alt)
    if colors:
        k, n = colors.most_common(1)[0]
        alt = (
            [s for s in alt if np.linalg.norm(np.array(s["color"]) - np.array(k) * 45) < 65]
            if n >= 2
            else []
        )
    large = [s for s in fits if s["size"] > typical * 1.35 and s["score"] > 0.4]
    italic = [s for s in fits if s["italic"] and not normal["italic"] and s["score"] > 0.53]
    emphasis = alt or italic or [s for s in large if s["size"] < 1.8 * typical]
    return {
        "normal": normal,
        "emphasis": style(emphasis) if emphasis else normal.copy(),
        "hero": style(large) if large else normal.copy(),
    }, {
        "alternate_fill_samples": len(alt),
        "larger_type_samples": len(large),
        "italic_alternate_samples": len(italic),
        "primary_ink_samples": len(base),
    }


def motion_probe(path, duration, box, width, height):
    # Three short windows isolate changes to the measured caption band.
    hints = []

    def read_window(start):
        seen = []
        x, y, r, b = box
        raw = run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{start:.5f}",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-t",
                "0.9",
                "-vf",
                f"fps=10,scale={width}:{height},crop={r - x}:{b - y}:{x}:{y}:exact=1",
                "-frames:v",
                "9",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        n = (r - x) * (b - y) * 3
        prev = None
        for j in range(0, len(raw) - n + 1, n):
            arr = np.frombuffer(raw[j : j + n], dtype=np.uint8).reshape(b - y, r - x, 3)
            # An OCR line pass on a small native region is cheaper than a whole-frame pass.
            im = Image.fromarray(arr)
            vals = ocr(im, psm=7)
            if not vals:
                prev = None
                continue
            tokens = tuple("".join(c for c in row["text"].lower() if c.isalnum()) for row in vals)
            if sum(map(len, tokens)) < 3:
                prev = None
                continue
            l = min(int(row["left"]) for row in vals)
            rr = max(int(row["left"]) + int(row["width"]) for row in vals)
            hh = max(int(row["height"]) for row in vals)
            if prev:
                old, ww, oh = prev
                if len(tokens) > len(old) and tokens[: len(old)] == old:
                    seen.append(("reveal", round(start + j / n / 10, 2)))
                elif (
                    old == tokens
                    and 1.07 < (rr - l) / max(1, ww) < 1.5
                    and 1.06 < hh / max(1, oh) < 1.55
                ):
                    seen.append(("pop", round(start + j / n / 10, 2)))
            prev = tokens, rr - l, hh
        return seen

    starts = sorted(
        {round(max(0, min(duration - 0.9, t)), 3) for t in (0, duration * 0.36, duration * 0.72)}
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        for part in pool.map(read_window, starts):
            hints.extend(part)
    counts = Counter(k for k, _ in hints)
    winner = counts.most_common(1)[0][0] if counts and max(counts.values()) >= 2 else "phrase"
    return winner, {
        "seconds_sampled": round(sum(min(0.9, duration - s) for s in starts), 2),
        "observations": sorted(hints, key=lambda h: h[1]),
        "confidence": "limited" if winner == "phrase" else "inferred",
    }


def preview(path, frames, width, band=None):
    thumbs = []
    for t, image in frames:
        if band:
            crop = image.crop((0, band[0], width, band[1]))
            crop.thumbnail((320, 126))
        else:
            crop = image.copy()
            crop.thumbnail((132, 180))
        thumbs.append((t, crop))
    cw, ch = (332, 150) if band else (144, 204)
    cols = 3 if band else 5
    sheet = Image.new("RGB", (cols * cw, math.ceil(len(thumbs) / cols) * ch), (27, 27, 30))
    draw = ImageDraw.Draw(sheet)
    for n, (t, im) in enumerate(thumbs):
        x = n % cols * cw
        y = n // cols * ch
        sheet.paste(im, (x + (cw - im.width) // 2, y + 3))
        draw.text((x + 6, y + ch - 17), f"{t:.2f} s", fill="white")
    sheet.save(path, quality=88)


def analyze(path, out_dir, roi=None, at=None):
    started = time.monotonic()
    for tool in ["ffmpeg", "ffprobe", "tesseract"]:
        if not shutil.which(tool):
            raise ValueError(f"Install {tool} and add it to PATH.")
    if roi is not None:
        if len(roi) != 4 or not all(math.isfinite(float(v)) for v in roi):
            raise ValueError("ROI needs x,y,width,height.")
        x, y, w, h = map(float, roi)
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.00001 or y + h > 1.00001:
            raise ValueError("ROI must fit inside the frame (0 to 1).")
        roi = (x, y, w, h)
    path = Path(path).resolve()
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    metadata = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    if not metadata.get("streams"):
        raise ValueError("The reference has no video stream.")
    stream = metadata["streams"][0]
    dur = float(stream.get("duration") or metadata["format"].get("duration") or 0)
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError("The video has no readable duration.")
    end = max(0, dur - min(0.22, dur * 0.12))
    times = sorted(
        {
            round(min(end, t), 3)
            for t in [dur * 0.015, 0.22, 1.65, *[end * i / 10 for i in range(1, 11)]]
        }
    )
    focus = sorted({float(t) for t in (at or [])})
    if len(focus) > 3 or any(not math.isfinite(t) or t < 0 or t >= dur for t in focus):
        raise ValueError("Choose up to three reference timestamps inside the video duration.")
    times = sorted(set(times + focus))
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames = [
            (t, im)
            for t, im in zip(times, pool.map(lambda t: sample(path, t), times))
            if im is not None
        ]
    if not frames:
        raise ValueError("FFmpeg could not decode any sampled frames.")
    width, height = frames[0][1].size
    box = roi_box(frames[0][1], roi)
    if box[2] - box[0] < 8 or box[3] - box[1] < 6:
        raise ValueError("The selected region is too small for OCR. Include the full caption line.")
    with ThreadPoolExecutor(max_workers=3) as pool:
        found = list(pool.map(lambda item: words_in(item[1][1], box, item[0]), enumerate(frames)))
    words = choose_band(
        [w for group in found for w in group], height, width, len(frames), manual=roi is not None
    )
    report = {
        "sample_times": [t for t, _ in frames],
        "sample_width": width,
        "sample_height": height,
        "duration": dur,
        "roi": list(roi) if roi else None,
        "focus_times": focus,
    }

    def fail(message):
        preview(out / "preview.jpg", frames, width)
        report.update(
            status="review", message=message, elapsed_seconds=round(time.monotonic() - started, 2)
        )
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "preset.json").unlink(missing_ok=True)
        raise ValueError(message)

    if not words:
        fail(
            "No recurring caption band could be identified. Inspect preview.jpg and supply a caption ROI or a shorter reference with visible captions."
        )
    # Include large and normal words, and disperse sampling across the clip.
    chosen = {}
    ranked = sorted(words, key=lambda w: (-w["conf"], -len(w["text"])))
    focused = [w for w in ranked if frames[w["frame"]][0] in focus][:16]
    for seq in [focused, sorted(ranked, key=lambda w: -w["h"])[:14], ranked]:
        for w in seq:
            if len(chosen) >= 48:
                break
            if 2 <= len(w["text"]) <= 22:
                chosen.setdefault((w["text"].casefold(), round(w["h"] / 4), w["kind"]), w)
    fits = [fit_word(frames[w["frame"]][1], w) for w in chosen.values()]
    fits = [f for f in fits if f]
    if len(fits) < 2:
        fail(
            "The sampled text could not be matched to enough glyphs. Inspect preview.jpg and retry with a tighter caption ROI."
        )
    roles, support = infer_roles(fits)
    by_frame = defaultdict(list)
    for w in words:
        by_frame[w["frame"]].append(w)
    groups = []
    for ws in by_frame.values():
        # Background signage tends to have much lower OCR confidence than the
        # body text, and can otherwise drag the apparent center off to one side.
        cutoff = max(70, max(w["conf"] for w in ws) - 15)
        clean = [w for w in ws if w["conf"] >= cutoff and len(w["text"]) > 1]
        ws = clean or ws
        l = min(w["x"] for w in ws)
        r = max(w["x"] + w["w"] for w in ws)
        t = min(w["y"] for w in ws)
        b = max(w["y"] + w["h"] for w in ws)
        groups.append((l, t, r, b))
    cx = float(np.median([(l + r) / (2 * width) for l, t, r, b in groups]))
    cy = float(np.median([(t + b) / (2 * height) for l, t, r, b in groups]))
    margin = round(0.025 * height)
    band = (
        max(0, round(np.quantile([t for _, t, _, _ in groups], 0.065)) - margin),
        min(height, round(np.quantile([b for _, _, _, b in groups], 0.935)) + margin),
    )
    animation, motion = motion_probe(path, dur, (box[0], band[0], box[2], band[1]), width, height)
    preset = {
        "name": path.stem + " captions",
        "x": round(cx, 4),
        "y": round(cy, 4),
        "animation": animation,
        "roles": roles,
    }
    report.update(
        status="sampled",
        band=[round(v / height, 4) for v in band],
        frames_with_text=len(groups),
        fitted_words=len(fits),
        style_support=support,
        motion=motion,
        median_glyph_overlap=round(float(np.median([s["score"] for s in fits])), 3),
        samples=[
            {
                "time": frames[f["frame"]][0],
                "text": f["text"],
                "font_guess": f["font"],
                "size": round(f["size"], 4),
                "score": round(f["score"], 3),
            }
            for f in sorted(fits, key=lambda f: f["score"], reverse=True)[:12]
        ],
        limitations=[
            "Font families are closest installed candidates. Size and tracking are fitted approximations.",
            "A role with no observed alternate borrows the normal style. Sparse motion samples can miss short animations.",
            "Outlines, shadows, plates, karaoke masks and gradients require visual review.",
        ],
        elapsed_seconds=round(time.monotonic() - started, 2),
    )
    preview(out / "preview.jpg", frames, width, band)
    for name, value in [("preset", preset), ("report", report)]:
        (out / (name + ".json")).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return {"preset": preset, "report": report}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--at", type=float, action="append", help="additional target time in seconds (up to three)"
    )
    p.add_argument("--roi", help="normalized x,y,width,height, e.g. 0.05,0.35,0.9,0.25")
    args = p.parse_args()
    try:
        result = analyze(
            args.video,
            args.out,
            tuple(map(float, args.roi.split(","))) if args.roi else None,
            args.at,
        )
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        p.exit(1, f"{exc}\n")
    print(json.dumps(result["preset"], indent=2))
