#!/usr/bin/env python3
"""Synchronize subtitle timing to match video audio.

Given video files and their associated subtitle files (.srt or .vtt),
transcribes the audio with Whisper, matches the recognized text against
subtitle cues, and shifts timestamps in one of two modes:

  -full     (default) single median offset applied to all cues
  -partial  full bulk shift, then a per-section offset interpolated
            between anchor points applied on top of the result

Writes the shifted subtitle next to the original with a .synced suffix
and report files named <stem>.report_full.<score>.txt and
<stem>.report_partial.<score>.txt where score is 000-100.

Runs on Python 3.8+.  Missing runtime pieces (pip, ffmpeg) are
force-installed into a per-user cache without requiring root.

Author: Chun Kang <kurapa@kurapa.com>
"""

import difflib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import venv
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "subtitle_sync"
VENV_DIR = CACHE_DIR / "venv"
FFMPEG_DIR = CACHE_DIR / "ffmpeg"
VENV_MARKER = "SUBSYNC_IN_VENV"
PIP_PACKAGES = ["faster-whisper"]
MIN_PYTHON = (3, 9)

# Static ffmpeg builds for Linux (e.g. Amazon Linux 2023, which has no ffmpeg
# in its default repos).  Fully static, so they run regardless of distro/glibc
# and need no root to install.
FFMPEG_STATIC_URLS = {
    "amd64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
    "arm64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
SUBTITLE_EXTS = (".srt", ".vtt")
SYNCED_TAG = ".synced"
WHISPER_TAG = ".whisper"


# ---------- bootstrap ----------

def _venv_python():
    return VENV_DIR / "bin" / "python3"


def _has_pip(py):
    return subprocess.run(
        [str(py), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _get_pip_url():
    v = sys.version_info
    if (v[0], v[1]) < (3, 7):
        return "https://bootstrap.pypa.io/pip/{0}.{1}/get-pip.py".format(v[0], v[1])
    return "https://bootstrap.pypa.io/get-pip.py"


def _bootstrap_pip(py):
    if _has_pip(py):
        return
    print("[subsync] pip unavailable; bootstrapping via get-pip.py")
    with tempfile.TemporaryDirectory() as tmp:
        get_pip = Path(tmp) / "get-pip.py"
        urllib.request.urlretrieve(_get_pip_url(), str(get_pip))
        subprocess.check_call([str(py), str(get_pip)])


def _find_python():
    candidates = ["/usr/bin/python3"]
    for minor in range(13, 8, -1):
        candidates.append("/usr/bin/python3.{0}".format(minor))
        candidates.append(shutil.which("python3.{0}".format(minor)) or "")
    candidates.append(shutil.which("python3") or "")
    for py in candidates:
        if not py or not os.path.isfile(py):
            continue
        try:
            out = subprocess.check_output(
                [py, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                stderr=subprocess.DEVNULL, universal_newlines=True,
            ).strip()
            major, minor = (int(x) for x in out.split())
            if (major, minor) >= MIN_PYTHON:
                return py
        except Exception:
            continue
    sys.exit(
        "error: no Python >= {0}.{1} found; faster-whisper requires it.\n"
        "       Install a newer Python, e.g.: pyenv install 3.12".format(*MIN_PYTHON)
    )


def _create_venv():
    py = _find_python()
    print("[subsync] creating venv at {0} (using {1})".format(VENV_DIR, py))
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([py, "-m", "venv", str(VENV_DIR)])
    _bootstrap_pip(_venv_python())


def _venv_python_version():
    py = _venv_python()
    if not py.exists():
        return (0, 0)
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            stderr=subprocess.DEVNULL, universal_newlines=True,
        ).strip()
        return tuple(int(x) for x in out.split())
    except Exception:
        return (0, 0)


def _ensure_venv_and_reexec():
    if os.environ.get(VENV_MARKER) == "1":
        return
    if VENV_DIR.exists() and _venv_python_version() < MIN_PYTHON:
        print("[subsync] venv Python too old; rebuilding")
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        _create_venv()
    venv_python = _venv_python()
    _bootstrap_pip(venv_python)
    if PIP_PACKAGES:
        missing = [p for p in PIP_PACKAGES if subprocess.run(
            [str(venv_python), "-m", "pip", "show", p],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode != 0]
        if missing:
            print("[subsync] installing: {0}".format(", ".join(missing)))
            subprocess.check_call(
                [str(venv_python), "-m", "pip", "install", "-q"] + missing
            )
    env = dict(os.environ)
    env[VENV_MARKER] = "1"
    os.execve(
        str(venv_python),
        [str(venv_python), os.path.abspath(__file__)] + sys.argv[1:],
        env,
    )


# ---------- ffmpeg bootstrap ----------

def _cached_ffmpeg_dir():
    if (FFMPEG_DIR / "ffmpeg").exists() and (FFMPEG_DIR / "ffprobe").exists():
        return FFMPEG_DIR
    return None


def _static_ffmpeg_url():
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return FFMPEG_STATIC_URLS["amd64"]
    if machine in ("aarch64", "arm64"):
        return FFMPEG_STATIC_URLS["arm64"]
    sys.exit("error: no static ffmpeg build available for architecture {0!r}".format(machine))


def _download_static_ffmpeg():
    url = _static_ffmpeg_url()
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    print("[subsync] downloading static ffmpeg ({0})".format(platform.machine()))
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ffmpeg.tar.xz"
        urllib.request.urlretrieve(url, str(archive))
        with tarfile.open(str(archive)) as tf:
            members = tf.getmembers()
            for name in ("ffmpeg", "ffprobe"):
                member = next(
                    (m for m in members if m.name.rsplit("/", 1)[-1] == name), None
                )
                if member is None:
                    sys.exit("error: {0} not found in downloaded ffmpeg archive".format(name))
                member.name = name
                tf.extract(member, str(FFMPEG_DIR))
                (FFMPEG_DIR / name).chmod(0o755)
    return FFMPEG_DIR


def _ensure_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    cached = _cached_ffmpeg_dir()
    if cached is not None:
        os.environ["PATH"] = "{0}{1}{2}".format(cached, os.pathsep, os.environ.get("PATH", ""))
        return

    if sys.platform == "darwin":
        if shutil.which("brew"):
            print("[subsync] installing ffmpeg via Homebrew")
            subprocess.check_call(["brew", "install", "ffmpeg"])
            return
        sys.exit("error: ffmpeg/ffprobe not found and Homebrew is unavailable; install ffmpeg manually")

    if sys.platform.startswith("linux"):
        cached = _download_static_ffmpeg()
        os.environ["PATH"] = "{0}{1}{2}".format(cached, os.pathsep, os.environ.get("PATH", ""))
        return

    sys.exit("error: ffmpeg/ffprobe not found; install ffmpeg manually")


# ---------- subtitle parsing ----------

def _parse_srt_ts(ts):
    h, m, rest = ts.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_vtt_ts(ts):
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, rest = parts
    else:
        h = "0"
        m, rest = parts
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_cues(path):
    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    if ext == ".srt":
        arrow_re = re.compile(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
        )
        parse_ts = _parse_srt_ts
    else:
        arrow_re = re.compile(
            r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
            r"\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
        )
        parse_ts = _parse_vtt_ts

    cues = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = arrow_re.search(lines[i])
        if m:
            start = parse_ts(m.group(1))
            end = parse_ts(m.group(2))
            body = []
            i += 1
            while i < len(lines) and lines[i].strip():
                body.append(lines[i].strip())
                i += 1
            cues.append((start, end, " ".join(body)))
        else:
            i += 1
    return cues


# ---------- subtitle shifting ----------

def _format_srt_ts(seconds):
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(h, m, s, ms)


def _format_vtt_ts(seconds):
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return "{0:02d}:{1:02d}:{2:02d}.{3:03d}".format(h, m, s, ms)


def _interpolate_offset(cue_start, anchors):
    if len(anchors) == 1:
        return anchors[0][1]
    if cue_start <= anchors[0][0]:
        return anchors[0][1]
    if cue_start >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        t0, o0 = anchors[i]
        t1, o1 = anchors[i + 1]
        if cue_start <= t1:
            if t1 == t0:
                return o0
            frac = (cue_start - t0) / (t1 - t0)
            return o0 + frac * (o1 - o0)
    return anchors[-1][1]


def shift_subtitle(subtitle, synced, offset, anchors=None):
    cues = parse_cues(subtitle)
    ext = subtitle.suffix.lower()

    shifted = []
    for start, end, text in cues:
        off = _interpolate_offset(start, anchors) if anchors else offset
        new_end = end + off
        if new_end <= 0:
            continue
        new_start = max(0.0, start + off)
        shifted.append((new_start, max(new_start, new_end), text))

    for i in range(1, len(shifted)):
        prev_start = shifted[i - 1][0]
        start, end, text = shifted[i]
        if start < prev_start:
            dur = end - start
            shifted[i] = (prev_start, prev_start + dur, text)

    if ext == ".srt":
        lines = []
        for i, (start, end, text) in enumerate(shifted, 1):
            lines.append(str(i))
            lines.append("{0} --> {1}".format(_format_srt_ts(start), _format_srt_ts(end)))
            lines.append(text)
            lines.append("")
        synced.write_text("\n".join(lines), encoding="utf-8")
    else:
        lines = ["WEBVTT", ""]
        for start, end, text in shifted:
            lines.append("{0} --> {1}".format(_format_vtt_ts(start), _format_vtt_ts(end)))
            lines.append(text)
            lines.append("")
        synced.write_text("\n".join(lines), encoding="utf-8")


# ---------- language detection ----------

# Unicode block ranges for CJK and Hangul characters.
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),    # Fullwidth Forms
]
_HANGUL_RANGES = [
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
]
_KANA_RANGES = [
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
]


def _in_ranges(ch, ranges):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def detect_language(cues):
    sample = " ".join(text for _, _, text in cues[:50])
    counts = {"ko": 0, "ja": 0, "zh": 0}
    total = 0
    for ch in sample:
        if ch.isspace():
            continue
        total += 1
        if _in_ranges(ch, _HANGUL_RANGES):
            counts["ko"] += 1
        elif _in_ranges(ch, _KANA_RANGES):
            counts["ja"] += 1
        elif _in_ranges(ch, _CJK_RANGES):
            counts["zh"] += 1
    if total == 0:
        return "en"
    if counts["ko"] > total * 0.3:
        return "ko"
    if counts["ja"] > total * 0.1:
        return "ja"
    if counts["zh"] > total * 0.3:
        return "zh"
    return "en"


# ---------- whisper transcription ----------

def _get_duration(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, check=True,
    )
    return float(result.stdout.strip())


def transcribe_audio(video, language):
    from faster_whisper import WhisperModel

    duration = _get_duration(video)
    print("[subsync] transcribing {0} ({1}) with whisper (medium, language={2})...".format(
        video.name, _fmt_ts(duration), language))
    compute_type = "int8" if sys.platform == "darwin" else "auto"
    model = WhisperModel("medium", compute_type=compute_type)
    segments, _info = model.transcribe(str(video), language=language)
    result = []
    for seg in segments:
        result.append((seg.start, seg.end, seg.text.strip()))
        pct = min(100, int(seg.end / duration * 100)) if duration > 0 else 0
        sys.stderr.write("\r[subsync] progress: {0}% ({1} / {2})".format(
            pct, _fmt_ts(seg.end), _fmt_ts(duration)))
    sys.stderr.write("\n")
    print("[subsync] transcribed {0} segment(s)".format(len(result)))
    return result


# ---------- text matching ----------

def _normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'[\d_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _find_matches(whisper_segments, cues, max_samples=30):
    MIN_RATIO = 0.4

    candidates = []
    best_below = 0.0
    for wi, (ws, _we, wtext) in enumerate(whisper_segments):
        wn = _normalize(wtext)
        if not wn:
            continue
        for ci, (cs, _ce, ctext) in enumerate(cues):
            cn = _normalize(ctext)
            if not cn:
                continue
            ratio = difflib.SequenceMatcher(None, wn, cn).ratio()
            if ratio >= MIN_RATIO:
                candidates.append((ratio, wi, ci, ws - cs, wtext, ctext, cs))
            elif ratio > best_below:
                best_below = ratio

    if not candidates:
        return best_below, []

    candidates.sort(reverse=True)
    used_w = set()
    used_c = set()
    samples = []
    for ratio, wi, ci, off, wtext, ctext, cs in candidates:
        if wi in used_w or ci in used_c:
            continue
        used_w.add(wi)
        used_c.add(ci)
        samples.append((off, ratio, wtext, ctext, cs))
        if max_samples and len(samples) >= max_samples:
            break

    return best_below, samples


def find_offset(whisper_segments, cues):
    MIN_SAMPLES = 10
    best_below, samples = _find_matches(whisper_segments, cues, max_samples=30)

    if not samples:
        return None, best_below, []

    if len(samples) < MIN_SAMPLES:
        return None, samples[0][1] if samples else best_below, samples

    offsets = sorted(s[0] for s in samples)
    median_offset = offsets[len(offsets) // 2]

    return median_offset, samples[0][1], samples


def find_anchors(whisper_segments, cues):
    MIN_SAMPLES = 3
    best_below, samples = _find_matches(whisper_segments, cues, max_samples=0)

    if not samples:
        return None, best_below, []

    if len(samples) < MIN_SAMPLES:
        return None, samples[0][1] if samples else best_below, samples

    anchors = sorted([(cs, off) for off, _ratio, _wt, _ct, cs in samples])
    return anchors, samples[0][1], samples


# ---------- scoring & report ----------

def compute_score(cues, speech):
    if not cues:
        return 0
    total_cue = 0.0
    total_overlap = 0.0
    for cs, ce, _ in cues:
        dur = ce - cs
        if dur <= 0:
            continue
        total_cue += dur
        for ss, se in speech:
            ov_start = max(cs, ss)
            ov_end = min(ce, se)
            if ov_start < ov_end:
                total_overlap += ov_end - ov_start
    if total_cue == 0:
        return 0
    return min(100, max(0, round(total_overlap / total_cue * 100)))


def _fmt_ts(t):
    m = int(t // 60)
    s = t % 60
    return "{0:02d}:{1:05.2f}".format(m, s)


def _cue_hits_speech(cs, ce, speech):
    return any(cs < se and ce > ss for ss, se in speech)


def _clean_old_reports(video):
    pattern = re.compile(re.escape(video.stem) + r"\.report(?:_(?:full|partial))?\.\d{3}\.txt$")
    for f in video.parent.iterdir():
        if f.is_file() and pattern.match(f.name):
            f.unlink()


def write_report(video, subtitle, synced, cues, speech, score, offset, samples,
                 mode="full", anchors=None, report_type="full"):
    report = video.with_name("{0}.report_{1}.{2:03d}.txt".format(video.stem, report_type, score))

    sample_offsets = [s[0] for s in samples]
    spread = max(sample_offsets) - min(sample_offsets) if len(samples) > 1 else 0.0

    lines = [
        "subtitle sync report",
        "",
        "video:      {0}".format(video.name),
        "subtitle:   {0}".format(subtitle.name),
        "synced:     {0}".format(synced.name),
        "score:      {0}%".format(score),
        "cues:       {0}".format(len(cues)),
        "mode:       {0}".format(mode),
    ]
    if mode == "partial" and anchors:
        anchor_offsets = [o for _, o in anchors]
        lines.append("anchors:    {0} (offset range: {1:+.3f}s to {2:+.3f}s, spread {3:.3f}s)".format(
            len(anchors), min(anchor_offsets), max(anchor_offsets), spread))
    else:
        lines.append("offset:     {0:+.3f}s (median of {1} sample(s), spread {2:.3f}s)".format(
            offset, len(samples), spread))
    lines += [
        "",
        "offset samples (by confidence):",
    ]
    for i, (off, ratio, wtext, ctext, _cs) in enumerate(samples, 1):
        lines.append("  #{0}  offset: {1:+.3f}s  confidence: {2:.0%}".format(i, off, ratio))
        lines.append("      whisper: {0}".format(wtext))
        lines.append("      cue:     {0}".format(ctext))
    if mode == "partial" and anchors:
        lines.append("")
        lines.append("anchor points (chronological):")
        for i, (cs, off) in enumerate(anchors, 1):
            lines.append("  #{0}  @ {1}  offset: {2:+.3f}s".format(i, _fmt_ts(cs), off))
    lines.append("")
    for i, (start, end, text) in enumerate(cues, 1):
        hit = _cue_hits_speech(start, end, speech)
        marker = "+" if hit else "-"
        preview = text[:60]
        lines.append(
            "  {0} {1:3d}  [{2} -> {3}]  {4}".format(
                marker, i, _fmt_ts(start), _fmt_ts(end), preview
            )
        )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# ---------- input discovery ----------

def is_synced_output(p):
    return p.stem.endswith(SYNCED_TAG)


def find_subtitle_for_video(video):
    for ext in SUBTITLE_EXTS:
        sub = video.with_suffix(ext)
        if sub.exists() and not is_synced_output(sub):
            return sub
    return None


def find_pairs(directory):
    pairs = []
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
            continue
        if f.name.startswith("."):
            continue
        sub = find_subtitle_for_video(f)
        if sub:
            pairs.append((f, sub))
    return pairs


def resolve_inputs(args):
    if args:
        pairs = []
        for a in args:
            video = Path(a)
            if not video.exists():
                sys.exit("error: no such file: {0}".format(a))
            if not video.is_file():
                sys.exit("error: not a file: {0}".format(a))
            sub = find_subtitle_for_video(video)
            if not sub:
                sys.exit(
                    "error: no subtitle file found for {0}\n"
                    "       expected {1}.srt or {1}.vtt".format(video.name, video.stem)
                )
            pairs.append((video, sub))
        return pairs
    found = find_pairs(Path.cwd())
    if not found:
        sys.exit(
            "no video+subtitle pairs found in the current directory.\n"
            "usage: subtitle_sync [video ...]"
        )
    return found


# ---------- sync ----------

def _whisper_transcript_path(video, ext=".srt"):
    return video.with_name(video.stem + WHISPER_TAG + ext)


def find_whisper_transcript(video):
    for ext in SUBTITLE_EXTS:
        path = _whisper_transcript_path(video, ext)
        if path.exists():
            return path
    return None


def _write_whisper_srt(video, segments):
    path = _whisper_transcript_path(video, ".srt")
    lines = []
    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append("{0} --> {1}".format(_format_srt_ts(start), _format_srt_ts(end)))
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[subsync] wrote whisper transcript: {0}".format(path.name))


def sync_one(video, subtitle, mode="full", save_transcript=True):
    cues = parse_cues(subtitle)
    if not cues:
        print("[subsync] no cues found in {0}; skipping".format(subtitle.name))
        return

    cached = find_whisper_transcript(video)
    if cached is not None:
        whisper_segments = parse_cues(cached)
        print("[subsync] reusing whisper transcript {0} ({1} segment(s))".format(
            cached.name, len(whisper_segments)))
    else:
        language = detect_language(cues)
        print("[subsync] detected subtitle language: {0}".format(language))
        whisper_segments = transcribe_audio(video, language)
        if whisper_segments and save_transcript:
            _write_whisper_srt(video, whisper_segments)

    if not whisper_segments:
        print("[subsync] no speech recognized in {0}; skipping".format(video.name))
        return

    _clean_old_reports(video)
    speech = [(s, e) for s, e, _ in whisper_segments]
    ext = subtitle.suffix
    synced = video.with_name(video.stem + SYNCED_TAG + ext)

    # --- full bulk shift ---
    print("[subsync] full bulk shift")
    offset, confidence, full_samples = find_offset(whisper_segments, cues)
    if offset is None:
        if full_samples:
            print("[subsync] only {0} text match(es) found (need 10); skipping".format(
                len(full_samples)))
            for off, ratio, wtext, ctext, _cs in full_samples:
                print("[subsync]   [{0:.0%} {1:+.3f}s] {2}".format(ratio, off, wtext[:60]))
                print("[subsync]              <-> {0}".format(ctext[:60]))
        else:
            print("[subsync] no text match found (best similarity: {0:.0%}); skipping".format(
                confidence))
        return

    print("[subsync] {0} sample(s), median offset: {1:+.3f}s".format(
        len(full_samples), offset))
    for off, ratio, wtext, ctext, _cs in full_samples[:5]:
        print("[subsync]   [{0:.0%} {1:+.3f}s] {2}".format(ratio, off, wtext[:60]))
        print("[subsync]              <-> {0}".format(ctext[:60]))
    if len(full_samples) > 5:
        print("[subsync]   ... and {0} more".format(len(full_samples) - 5))

    shift_subtitle(subtitle, synced, offset)
    synced_cues = parse_cues(synced)
    full_score = compute_score(synced_cues, speech)
    full_report = write_report(video, subtitle, synced, synced_cues, speech,
                               full_score, offset, full_samples,
                               mode="full", report_type="full")
    print("[subsync] full: {0} (score: {1}%)".format(full_report.name, full_score))

    if mode != "partial":
        print("[subsync] wrote {0}".format(synced.name))
        return

    # --- partial correction on the full-synced result ---
    print("[subsync] partial correction")
    anchors, confidence, partial_samples = find_anchors(whisper_segments, synced_cues)
    if anchors is None:
        if partial_samples:
            print("[subsync] only {0} anchor(s) found (need 3); keeping full sync result".format(
                len(partial_samples)))
        else:
            print("[subsync] no anchors found; keeping full sync result")
        print("[subsync] wrote {0}".format(synced.name))
        return

    anchor_offsets = [o for _, o in anchors]
    print("[subsync] {0} anchor(s), offset range: {1:+.3f}s to {2:+.3f}s".format(
        len(anchors), min(anchor_offsets), max(anchor_offsets)))
    for off, ratio, wtext, ctext, _cs in partial_samples[:5]:
        print("[subsync]   [{0:.0%} {1:+.3f}s] {2}".format(ratio, off, wtext[:60]))
        print("[subsync]              <-> {0}".format(ctext[:60]))
    if len(partial_samples) > 5:
        print("[subsync]   ... and {0} more".format(len(partial_samples) - 5))

    shift_subtitle(synced, synced, None, anchors=anchors)
    synced_cues = parse_cues(synced)
    partial_score = compute_score(synced_cues, speech)
    partial_report = write_report(video, subtitle, synced, synced_cues, speech,
                                  partial_score, None, partial_samples,
                                  mode="partial", anchors=anchors,
                                  report_type="partial")
    print("[subsync] partial: {0} (score: {1}%)".format(partial_report.name, partial_score))
    print("[subsync] wrote {0}".format(synced.name))


# ---------- driver ----------

def main():
    argv = sys.argv[1:]
    if any(a in ("-h", "-help", "--help") for a in argv):
        print("usage: subtitle_sync [-full|--full | -partial|--partial] [--no-transcript] [video ...]")
        print("  Transcribe video audio with Whisper, match against subtitle")
        print("  text, and shift subtitle timestamps to match.")
        print("  With no arguments, processes every video+subtitle pair in")
        print("  the current directory.")
        print("")
        print("  modes:")
        print("    -full     (default) single median offset applied to all cues")
        print("    -partial  full bulk shift, then a per-section correction")
        print("              interpolated between anchor points on the result")
        print("")
        print("  options:")
        print("    --no-transcript  skip saving the whisper transcript")
        print("                     (saved by default as <video>.whisper.srt;")
        print("                     an existing <video>.whisper.srt/.vtt is")
        print("                     reused instead of re-transcribing)")
        return

    save_transcript = "--no-transcript" not in argv
    mode = "partial" if ("-partial" in argv or "--partial" in argv) else "full"
    flags = ("-full", "--full", "-partial", "--partial", "--no-transcript")
    args = [a for a in argv if a not in flags]

    pairs = resolve_inputs(args)
    print("[subsync] mode: {0}".format(mode))
    print("[subsync] {0} pair(s) to sync:".format(len(pairs)))
    for video, sub in pairs:
        print("        {0} + {1}".format(video.name, sub.name))
    print()

    for video, sub in pairs:
        sync_one(video, sub, mode=mode, save_transcript=save_transcript)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
