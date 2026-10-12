"""Scorebug layouts per network, network detection, and per-frame crop extraction."""
from pathlib import Path

import numpy as np

from . import tools

FPS = 10
BUG_SCALE = 4  # downscale factor for the whole scorebug

# Regions in 1280x720 coordinates: (x, y, w, h).
# bug: static part of the scorebug (logos, scores), used to tell whether it's on screen.
# clock: the play clock.
# mode "frozen": during a play the clock freezes on its reset value (40, or 25).
# mode "hidden": during a play the clock isn't shown at all (FOX).
PRESETS = {
    "cbs": {"bug": (370, 608, 540, 77), "clock": (703, 649, 34, 34), "mode": "frozen"},
    # FOX shows the down & distance bar over the side of the team with the
    # ball, so its clock sits in one of two places: left bar end or right one.
    "fox": {"bug": (350, 604, 570, 76), "clock": [(536, 571, 60, 26), (818, 571, 60, 26)], "mode": "hidden"},
    "nbc": {"bug": (330, 634, 620, 36), "clock": (642, 673, 28, 18), "mode": "frozen"},
    "prime": {"bug": (345, 630, 590, 56), "clock": (684, 661, 34, 24), "mode": "frozen"},
    "espn": {"bug": (290, 628, 700, 70), "clock": (684, 668, 32, 24), "mode": "frozen"},
    # NFL Network's international-game feed (e.g. London games simulcast on a
    # local affiliate). Clock resets to 40/25 and holds for the play's
    # duration like CBS/NBC/Prime, just at different coordinates.
    "nflnet": {"bug": (280, 600, 700, 100), "clock": (678, 662, 50, 26), "mode": "frozen"},
}


def read_crops(src, region, scale=1, fps=FPS):
    x, y, w, h = region
    ow, oh = w // scale, h // scale
    cmd = [
        tools.ffmpeg(), "-loglevel", "error", "-i", str(src),
        "-vf", f"fps={fps},scale=1280:720,crop={w}:{h}:{x}:{y},scale={ow}:{oh}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = tools.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, oh, ow)


def sample_frames(src, duration, n=60):
    """Grayscale 1280x720 frames at n evenly spaced timestamps (fast seeks)."""
    frames = []
    for t in np.linspace(duration * 0.05, duration * 0.95, n):
        raw = tools.run([
            tools.ffmpeg(), "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
            "-vf", "scale=1280:720", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ], capture_output=True, check=True).stdout
        frames.append(np.frombuffer(raw, np.uint8).reshape(720, 1280))
    return np.stack(frames)


def bug_distance(crops, window=None, step=None):
    """Distance of each crop to the scorebug's typical look. With a window (in
    samples), the reference is a local median, so score changes don't drift it."""
    crops = crops.astype(np.float32)
    if window is None or len(crops) <= window:
        return np.abs(crops - np.median(crops, axis=0)).mean(axis=(1, 2))
    step = step or max(1, window // 12)
    out = np.empty(len(crops), np.float32)
    for a in range(0, len(crops), step):
        lo, hi = max(0, a - window // 2), min(len(crops), a + window // 2)
        ref = np.median(crops[lo:hi:3], axis=0)
        out[a:a + step] = np.abs(crops[a:a + step] - ref).mean(axis=(1, 2))
    return out


LOGO_REGION = (1040, 10, 230, 60)  # network watermark, top right
LOGO_DIR = Path(__file__).parent / "assets" / "logos"


def detect_network(src, duration):
    """Match the top-right network watermark against the stored logo templates."""
    x, y, w, h = LOGO_REGION
    frames = sample_frames(src, duration, 30)[:, y:y + h, x:x + w].astype(np.float32)
    scores = {}
    for path in sorted(LOGO_DIR.glob("*.npz")):
        t = np.load(path)
        scores[path.stem] = float(np.median([np.abs(f - t["med"])[t["mask"]].mean() for f in frames]))
    return min(scores, key=scores.get), scores
