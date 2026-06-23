#!/usr/bin/env python3
"""Synchronize subtitle timing to match video audio.

Given video files and their associated subtitle files (.srt or .vtt),
transcribes the audio with Whisper, matches the recognized text against
subtitle cues, and bulk-shifts all timestamps by the detected offset.
Writes the shifted subtitle next to the original with a .synced suffix
and a report file named <stem>.report.<score>.txt where score is 000-100.

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


def _shift_srt(text, offset):
    def replace_line(match):
        t1 = _parse_srt_ts(match.group(1)) + offset
        t2 = _parse_srt_ts(match.group(2)) + offset
        return "{0} --> {1}".format(_format_srt_ts(t1), _format_srt_ts(t2))

    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
    )
    return pattern.sub(replace_line, text)


def _shift_vtt(text, offset):
    def replace_line(match):
        t1 = _parse_vtt_ts(match.group(1)) + offset
        t2 = _parse_vtt_ts(match.group(2)) + offset
        return "{0} --> {1}".format(_format_vtt_ts(t1), _format_vtt_ts(t2))

    pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
        r"\s*-->\s*"
        r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )
    return pattern.sub(replace_line, text)


def shift_subtitle(subtitle, synced, offset):
    text = subtitle.read_text(encoding="utf-8")
    ext = subtitle.suffix.lower()
    if ext == ".srt":
        shifted = _shift_srt(text, offset)
    else:
        shifted = _shift_vtt(text, offset)
    synced.write_text(shifted, encoding="utf-8")


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

def transcribe_audio(video, language):
    from faster_whisper import WhisperModel

    print("[subsync] transcribing {0} with whisper (medium, language={1})...".format(
        video.name, language))
    model = WhisperModel("medium", compute_type="int8")
    segments, _info = model.transcribe(str(video), language=language)
    result = []
    for seg in segments:
        result.append((seg.start, seg.end, seg.text.strip()))
    return result


# ---------- text matching ----------

def _normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_offset(whisper_segments, cues):
    best_ratio = 0.0
    best_offset = 0.0
    best_w = ""
    best_c = ""

    for ws, _we, wtext in whisper_segments:
        wn = _normalize(wtext)
        if not wn:
            continue
        for cs, _ce, ctext in cues:
            cn = _normalize(ctext)
            if not cn:
                continue
            ratio = difflib.SequenceMatcher(None, wn, cn).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = ws - cs
                best_w = wtext
                best_c = ctext

    if best_ratio < 0.4:
        return None, best_ratio, "", ""
    return best_offset, best_ratio, best_w, best_c


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
    pattern = re.compile(re.escape(video.stem) + r"\.report\.\d{3}\.txt$")
    for f in video.parent.iterdir():
        if f.is_file() and pattern.match(f.name):
            f.unlink()


def write_report(video, subtitle, synced, cues, speech, score, offset,
                 match_confidence, match_whisper, match_cue):
    _clean_old_reports(video)
    report = video.with_name("{0}.report.{1:03d}.txt".format(video.stem, score))

    lines = [
        "subtitle sync report",
        "",
        "video:      {0}".format(video.name),
        "subtitle:   {0}".format(subtitle.name),
        "synced:     {0}".format(synced.name),
        "score:      {0}%".format(score),
        "cues:       {0}".format(len(cues)),
        "offset:     {0:+.3f}s".format(offset),
        "confidence: {0:.0%}".format(match_confidence),
        "",
        "matched whisper: {0}".format(match_whisper),
        "matched cue:     {0}".format(match_cue),
        "",
    ]
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

def sync_one(video, subtitle):
    cues = parse_cues(subtitle)
    if not cues:
        print("[subsync] no cues found in {0}; skipping".format(subtitle.name))
        return

    language = detect_language(cues)
    print("[subsync] detected subtitle language: {0}".format(language))
    whisper_segments = transcribe_audio(video, language)
    if not whisper_segments:
        print("[subsync] no speech recognized in {0}; skipping".format(video.name))
        return

    offset, confidence, match_w, match_c = find_offset(whisper_segments, cues)
    if offset is None:
        print("[subsync] no text match found (best similarity: {0:.0%}); skipping".format(
            confidence))
        return

    ext = subtitle.suffix
    synced = video.with_name(video.stem + SYNCED_TAG + ext)
    print("[subsync] match confidence: {0:.0%}, offset: {1:+.3f}s".format(confidence, offset))
    print("[subsync]   whisper: {0}".format(match_w))
    print("[subsync]   cue:     {0}".format(match_c))
    print("[subsync] shifting {0} -> {1}".format(subtitle.name, synced.name))

    shift_subtitle(subtitle, synced, offset)
    print("[subsync] wrote {0}".format(synced.name))

    speech = [(s, e) for s, e, _ in whisper_segments]
    synced_cues = parse_cues(synced)
    score = compute_score(synced_cues, speech)
    report = write_report(video, subtitle, synced, synced_cues, speech, score,
                          offset, confidence, match_w, match_c)
    print("[subsync] report: {0} (score: {1}%)".format(report.name, score))


# ---------- driver ----------

def main():
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) != len(sys.argv[1:]):
        print("usage: subtitle_sync [video ...]")
        print("  Transcribe video audio with Whisper, match against subtitle")
        print("  text, and bulk-shift timestamps by the detected offset.")
        print("  With no arguments, processes every video+subtitle pair in")
        print("  the current directory.")
        return

    pairs = resolve_inputs(args)
    print("[subsync] {0} pair(s) to sync:".format(len(pairs)))
    for video, sub in pairs:
        print("        {0} + {1}".format(video.name, sub.name))
    print()

    for video, sub in pairs:
        sync_one(video, sub)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
