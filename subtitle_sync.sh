#!/usr/bin/env python3
"""Synchronize subtitle timing to match video audio.

Given video files and their associated subtitle files (.srt or .vtt),
re-times the subtitles to match the spoken dialogue. Writes the synced
subtitle next to the original with a .synced suffix and a report file
named <stem>.report.<score>.txt where score is 000-100.

Timing reference comes from Whisper (faster-whisper): it transcribes the
audio to locate where dialogue actually occurs, which -- unlike energy or
generic voice-activity detection -- ignores music, laughter, and sound
effects. That speech timeline is rasterized into a reference signal and
ffsubsync aligns the subtitle (offset + framerate) against it. The
subtitle's own text is never changed; only its timing.

Runs on Python 3.6+ so it works on older distros (e.g. CentOS 7's EPEL
python3) as well as current ones. Missing runtime pieces (pip, ffmpeg)
are force-installed into a per-user cache without requiring root.

Author: Chun Kang <kurapa@kurapa.com>
"""

import os

# Quiet the "unauthenticated requests to the HF Hub / set a HF_TOKEN" advisory
# faster-whisper emits while downloading models. Set before any huggingface_hub
# import so it takes effect; downloads work fine without a token.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import bisect
import hashlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import venv
from difflib import SequenceMatcher
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "subtitle_sync"
VENV_DIR = CACHE_DIR / "venv"
FFMPEG_DIR = CACHE_DIR / "ffmpeg"
TRANSCRIPT_CACHE = CACHE_DIR / "transcripts"
VENV_MARKER = "SUBSYNC_IN_VENV"
PIP_PACKAGES = ["ffsubsync", "faster-whisper", "numpy", "langdetect"]

# Static ffmpeg builds for Linux (e.g. Amazon Linux 2023, which has no ffmpeg
# in its default repos, and CentOS 7, whose glibc is too old for many wheels).
# These are fully static, so they run regardless of distro/glibc and need no
# root to install.
FFMPEG_STATIC_URLS = {
    "amd64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
    "arm64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
SUBTITLE_EXTS = (".srt", ".vtt")
SYNCED_TAG = ".synced"

# Whisper model used to locate dialogue. We only keep each segment's start/end
# timing (the transcribed text is discarded), so we don't need large-v3's text
# accuracy -- medium's speech *boundaries* track dialogue nearly as well while
# running noticeably faster on CPU. tiny < base < small < medium < large-v3
# trade speed for accuracy; override with SUBSYNC_WHISPER_MODEL (e.g. "small"
# to go faster, "large-v3" for a noisy mix).
WHISPER_MODEL = os.environ.get("SUBSYNC_WHISPER_MODEL", "medium")

# ffsubsync samples speech at 100 Hz (see ffsubsync.constants.SAMPLE_RATE); the
# reference array we hand it must use the same rate.
REFERENCE_HZ = 100

# How far a framerate scale may stray from 1.0 before we treat the alignment as
# bogus. Real broadcast mismatches (PAL 25fps vs film 23.976fps = 0.959, and its
# inverse 1.042) deviate ~0.04, so 0.06 admits those while rejecting the spurious
# ~0.10 stretches a free framerate search produces on dense/ambiguous audio. We
# also pass it to ffsubsync as --max-framerate-deviation so it never explores
# beyond the plausible band in the first place.
MAX_SCALE_DEVIATION = 0.06

# Alignment passes tried against the Whisper-derived reference, in order. The
# reference (the expensive Whisper step) is built once; these only re-run the
# near-instant ffsubsync offset/framerate search.
#
# Offset-only comes first because same-source rips (the common case) need only a
# constant shift; correcting a framerate that doesn't actually differ is what
# wrecks the sync. The second pass lets ffsubsync infer a *standard* framerate
# ratio for genuine PAL/NTSC mismatches, capped to the plausible band. We do NOT
# use --gss (golden-section search): its continuous, unconstrained search readily
# finds a spurious ~0.9x ratio that maximizes correlation against dense dialogue
# while stretching the timeline into nonsense.
ALIGN_STRATEGIES = [
    ("offset-only", ["--no-fix-framerate"]),
    ("offset+framerate", ["--max-framerate-deviation", "{0:.3f}".format(MAX_SCALE_DEVIATION)]),
]
# Our score (0-100) at/above which a pass is accepted without trying the rest,
# and below which the final result is flagged as unreliable.
EARLY_ACCEPT_SCORE = 90
LOW_QUALITY_SCORE = 75
# ffsubsync's alignment score is not comparable across files, but its sign is
# meaningful: a value at or below this is anti-correlated, i.e. a wrong sync.
MIN_FFSUBSYNC_SCORE = 0.0

# Top-anchored ASS position tags ({\an7}, {\an8}, {\an9}) mark on-screen-text
# captions -- translations of on-screen text, sound-effect labels, song lyrics --
# that are pinned to the top of the frame and do NOT track spoken dialogue. In
# dense variety/reality content these can outnumber the actual dialogue cues, so
# feeding them to the aligner is feeding it mostly noise. We strip them when
# building the alignment reference (the final file still keeps every cue).
CAPTION_RE = re.compile(r"\{\\an[789]\}")
# If filtering would leave too little to align against, we don't trust the split
# and fall back to using every cue.
MIN_DIALOGUE_CUES = 30
MIN_DIALOGUE_FRACTION = 0.2

# ----- content alignment (Tier 2) -----
# When the subtitle's language matches what Whisper can produce for the audio --
# the spoken language itself, or English via Whisper's translate task -- we can
# match actual words to find anchor points (subtitle_time <-> audio_time) and fit
# a piecewise-linear time map. Unlike a single global offset, that map corrects
# drift and survives the ad-break gaps and re-cuts common in broadcast content.
#
# Two kinds of anchors feed the time map, because a machine translation rarely
# reproduces the professional subtitle's exact wording:
#   - word-run anchors: a run of this many consecutive shared words. Short runs
#     survive paraphrasing better, and the monotonic/slope filters downstream
#     discard the coincidental ones, so 2 maximizes recall safely.
#   - fuzzy cue anchors: a subtitle cue and a transcript segment that share
#     enough content words (set overlap), even reordered or reworded. This is
#     what recovers anchors when the translations diverge but name the same
#     people, numbers, and nouns.
MIN_MATCH_BLOCK = 2
# A fuzzy cue<->segment pair anchors when it shares at least this many content
# words AND their Jaccard overlap clears the floor (either condition strong
# enough on its own also qualifies, see fuzzy_anchors).
FUZZY_MIN_SHARED = 2
FUZZY_MIN_JACCARD = 0.30
FUZZY_STRONG_SHARED = 4
# Don't trust a content map built from fewer clean anchors than this...
MIN_ANCHORS = 12
# ...or one whose anchors span less than this fraction of the subtitle timeline
# (a map extrapolated from one clump drifts badly outside it).
MIN_ANCHOR_COVERAGE = 0.5
# A clean anchor's local slope (d audio / d subtitle) must stay within this of
# the median; a same-source recording maps at ~1.0, so large local deviations
# are mismatched words, not real timing.
ANCHOR_SLOPE_TOL = 0.05
# The overall slope across all anchors must stay in this band, else the match is
# bogus regardless of how clean it looks locally.
MIN_GLOBAL_SLOPE = 0.9
MAX_GLOBAL_SLOPE = 1.111
# Tokens shorter than this are dropped before matching: single letters and most
# stopword fragments match by coincidence and dilute real anchors.
MIN_TOKEN_LEN = 2


# ---------- bootstrap ----------

def _venv_python():
    return VENV_DIR / "bin" / "python3"


def _has_pip(py):
    return subprocess.run(
        [str(py), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _get_pip_url():
    # PyPA keeps version-pinned installers for interpreters that current pip
    # has dropped (CentOS 7 ships Python 3.6).
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


def _create_venv():
    print("[subsync] creating venv at {0}".format(VENV_DIR))
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Fails on minimal interpreters (e.g. CentOS 7) that lack ensurepip.
        venv.create(VENV_DIR, with_pip=True)
    except Exception as exc:
        print("[subsync] ensurepip unavailable ({0}); creating venv without pip".format(exc))
        venv.create(VENV_DIR, with_pip=False)
    _bootstrap_pip(_venv_python())


def _ensure_venv_and_reexec():
    if os.environ.get(VENV_MARKER) == "1":
        return
    if not VENV_DIR.exists():
        _create_venv()
    venv_python = _venv_python()
    # A pre-existing venv may predate the pip bootstrap; make sure pip is there.
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
                member.name = name  # flatten into FFMPEG_DIR, ignore archive subdir
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

SRT_ARROW_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)
VTT_ARROW_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
)


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


def _fmt_srt_ts(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(h, m, s, ms)


def _fmt_vtt_ts(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "{0:02d}:{1:02d}:{2:02d}.{3:03d}".format(h, m, s, ms)


def _arrow_re_for(ext):
    return SRT_ARROW_RE if ext == ".srt" else VTT_ARROW_RE


def _ts_funcs_for(ext):
    # (parse, format) for the given subtitle extension.
    if ext == ".srt":
        return _parse_srt_ts, _fmt_srt_ts
    return _parse_vtt_ts, _fmt_vtt_ts


def parse_cues(path):
    # utf-8-sig transparently strips a leading BOM (common in subtitles
    # exported from Windows tools) while still decoding plain UTF-8.
    text = path.read_text(encoding="utf-8-sig")
    ext = path.suffix.lower()
    arrow_re = _arrow_re_for(ext)
    parse_ts = _parse_srt_ts if ext == ".srt" else _parse_vtt_ts

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


def validate_subtitle(path):
    """Sanity-check that a subtitle file is well-formed before syncing.

    Returns the parsed cues on success and raises ValueError (with a
    human-readable reason) otherwise. ffsubsync happily emits a garbage
    "synced" file when handed an empty, wrong-encoding, or malformed
    subtitle, so we'd rather fail loudly here than after the slow audio
    analysis has run.
    """
    ext = path.suffix.lower()
    if ext not in SUBTITLE_EXTS:
        raise ValueError("unsupported subtitle extension {0!r}".format(ext))
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("not valid UTF-8 text ({0})".format(exc))
    if not text.strip():
        raise ValueError("file is empty")
    if ext == ".vtt" and not text.lstrip().upper().startswith("WEBVTT"):
        raise ValueError("missing 'WEBVTT' header")
    if "-->" not in text:
        raise ValueError("no subtitle timing lines ('-->') found")
    cues = parse_cues(path)
    if not cues:
        raise ValueError("no cues could be parsed (malformed timestamps?)")
    bad = sum(1 for cs, ce, _ in cues if ce <= cs or cs < 0)
    if bad * 2 > len(cues):
        raise ValueError(
            "{0} of {1} cues have invalid timing (end <= start or negative)".format(
                bad, len(cues)
            )
        )
    return cues


def is_caption_cue(text):
    """True for top-anchored on-screen-text captions (see CAPTION_RE)."""
    return bool(CAPTION_RE.search(text))


def dialogue_cues(cues):
    """Cues likely to be spoken dialogue (captions removed).

    Returns the full list unchanged if filtering would leave too little to align
    against -- some sources tag everything, or nothing, and in those cases the
    split is meaningless and we're better off using all cues.
    """
    dialogue = [c for c in cues if not is_caption_cue(c[2])]
    if len(dialogue) < MIN_DIALOGUE_CUES or (
        cues and len(dialogue) < MIN_DIALOGUE_FRACTION * len(cues)
    ):
        return cues, False
    return dialogue, True


def write_alignment_srt(cues, path):
    """Write cues as a minimal SRT for ffsubsync to align against.

    Only the timings matter to the aligner (it derives a speech track from cue
    presence), so the body is a placeholder and the output is always SRT
    regardless of the source format.
    """
    lines = []
    for i, (start, end, _) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append("{0} --> {1}".format(_fmt_srt_ts(start), _fmt_srt_ts(end)))
        lines.append(".")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def rewrite_timestamps(in_path, out_path, map_fn):
    """Re-time every cue through map_fn (seconds -> seconds), writing out_path.

    Rewrites only the timestamps on each '-->' line and leaves everything else
    (indices, cue settings, body text, blank lines, VTT header) byte-for-byte
    intact, so the synced file is a faithful re-timing of the original rather
    than a reserialization. Times that would go negative are clamped to 0; the
    count of such cues is returned so the caller can warn.
    """
    ext = in_path.suffix.lower()
    arrow_re = _arrow_re_for(ext)
    parse_ts, fmt_ts = _ts_funcs_for(ext)
    text = in_path.read_text(encoding="utf-8-sig")
    clamped = [0]

    def repl(m):
        new_start = map_fn(parse_ts(m.group(1)))
        new_end = map_fn(parse_ts(m.group(2)))
        if new_start < 0 or new_end < 0:
            clamped[0] += 1
        return "{0} --> {1}".format(fmt_ts(new_start), fmt_ts(new_end))

    out_path.write_text(arrow_re.sub(repl, text), encoding="utf-8")
    return clamped[0]


def apply_time_transform(in_path, out_path, scale, offset):
    """Faithfully re-time a subtitle by the linear map new_t = scale*t + offset."""
    return rewrite_timestamps(in_path, out_path, lambda t: scale * t + offset)


# ---------- speech detection ----------

def _get_duration(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, check=True,
    )
    return float(result.stdout.strip())


def real_media_end(video):
    """Timestamp of the last decodable audio packet, or None if unknown.

    A truncated or incompletely-downloaded file keeps the original duration in
    its container header while its actual streams stop early. ffprobe reads the
    packet index (fast -- it doesn't decode), so the last audio packet's PTS is
    the real end of the audio, which can be far short of the claimed duration.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(video)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, check=True,
        )
    except subprocess.CalledProcessError:
        return None
    last = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                last = float(line)
            except ValueError:
                pass
    return last


def warn_truncated_media(video, cues, container_dur, real_end):
    """Warn if the file is truncated or the subtitle outruns the media.

    Either condition means a chunk of the subtitle has no audio to sync against,
    so the user should know before trusting (or distrusting) the result.
    """
    if real_end is None:
        return
    if real_end < 0.97 * container_dur:
        print("[subsync] WARNING: {0} looks truncated -- its audio ends at {1} but "
              "the container claims {2}. The file may be incomplete or still "
              "downloading.".format(video.name, _fmt_ts(real_end),
                                    _fmt_ts(container_dur)))
    beyond = [c for c in cues if c[0] >= real_end + 5.0]
    if cues and len(beyond) > 0.03 * len(cues):
        print("[subsync] WARNING: {0} of {1} subtitle cue(s) ({2:.0f}%) start after "
              "the audio ends ({3}); the subtitle appears to cover a longer cut "
              "than this video. Those cues cannot be synced.".format(
                  len(beyond), len(cues), 100.0 * len(beyond) / len(cues),
                  _fmt_ts(real_end)))


def transcribe_dialogue(video, task="transcribe"):
    """Transcribe a video's dialogue with Whisper.

    Returns (segments, audio_lang) where segments is a list of dicts with keys
    start, end, text, and words (a list of (word, time) with word-level
    timestamps). Whisper's VAD filter keeps the segments on actual speech and off
    music / laughter / sound effects -- the failure mode of energy- or
    webrtc-based detection on dense content.

    task="transcribe" yields text in the spoken language; task="translate" yields
    English regardless of the spoken language. We use the timings for the Tier 1
    speech reference and the text/words for Tier 2 content anchoring; audio_lang
    is Whisper's detected spoken language either way.

    The result is cached on disk keyed by the video's identity and the model/task,
    so re-syncing (e.g. after tuning) skips the slow transcription entirely.
    """
    cache = _transcript_cache_path(video, task)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            print("[subsync] using cached transcript: {0}".format(cache.name))
            return data["segments"], data.get("audio_lang")
        except (ValueError, KeyError):
            pass  # corrupt cache; re-transcribe

    from faster_whisper import WhisperModel

    print("[subsync] loading whisper model: {0} (downloads on first use)".format(
        WHISPER_MODEL))
    model = WhisperModel(
        WHISPER_MODEL, device="cpu", compute_type="int8",
        cpu_threads=os.cpu_count() or 0,
    )
    print("[subsync] transcribing {0} ({1}) to locate dialogue (this can take a "
          "while)".format(video.name, task))
    segments, info = model.transcribe(
        str(video),
        task=task,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    # faster-whisper yields segments lazily as the audio is processed, so we can
    # report progress by comparing each segment's end against the total media
    # duration. info.duration is the portion Whisper will actually scan (after
    # VAD), which is what the timestamps run up to.
    total = float(getattr(info, "duration", 0.0)) or 0.0
    audio_lang = getattr(info, "language", None)
    out = []
    last_pct = -1
    stderr_tty = sys.stderr.isatty()
    for seg in segments:
        if seg.text and seg.text.strip():
            words = []
            for w in (seg.words or []):
                if w.word and w.word.strip():
                    words.append((w.word, (float(w.start) + float(w.end)) / 2.0))
            out.append({"start": float(seg.start), "end": float(seg.end),
                        "text": seg.text, "words": words})
        if total > 0:
            pct = min(100, int(seg.end / total * 100))
            if pct != last_pct:
                last_pct = pct
                msg = "[subsync] transcribing... {0:3d}% ({1} / {2})".format(
                    pct, _fmt_ts(seg.end), _fmt_ts(total))
                if stderr_tty:
                    sys.stderr.write("\r" + msg)
                    sys.stderr.flush()
                else:
                    print(msg)
    if stderr_tty and last_pct >= 0:
        sys.stderr.write("\n")
        sys.stderr.flush()
    print("[subsync] found {0} dialogue segment(s); detected audio language: "
          "{1}".format(len(out), audio_lang or "unknown"))
    try:
        TRANSCRIPT_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"segments": out, "audio_lang": audio_lang}),
                         encoding="utf-8")
    except OSError:
        pass  # caching is best-effort
    return out, audio_lang


def _transcript_cache_path(video, task):
    st = video.stat()
    key = "{0}|{1}|{2}|{3}|{4}".format(
        video.resolve(), st.st_mtime_ns, st.st_size, WHISPER_MODEL, task)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return TRANSCRIPT_CACHE / "{0}.{1}.{2}.json".format(video.stem, task, digest)


def speech_ranges(segments):
    """[(start, end)] for cues with text -- the Tier 1 timing reference."""
    return [(s["start"], s["end"]) for s in segments if s["text"].strip()]


def build_reference(speech, duration, path):
    """Rasterize speech ranges into ffsubsync's reference format.

    ffsubsync accepts a .npz holding a 'speech' array sampled at REFERENCE_HZ,
    where frames >= 1.0 are speech. Aligning against this lets ffsubsync do its
    robust offset + framerate search using the Whisper timeline as ground truth.
    """
    import numpy as np

    n = max(1, int(round(duration * REFERENCE_HZ)))
    arr = np.zeros(n, dtype=np.float32)
    for s, e in speech:
        a = max(0, int(round(s * REFERENCE_HZ)))
        b = min(n, int(round(e * REFERENCE_HZ)))
        if b > a:
            arr[a:b] = 1.0
    np.savez(str(path), speech=arr)


# ---------- content alignment (Tier 2) ----------

# ASS override blocks ({\an8}, {\i1}...) and HTML-ish tags (<i>, <font ...>)
# carry no spoken words and must be stripped before tokenizing.
_TAG_RE = re.compile(r"\{[^}]*\}|<[^>]*>")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _norm_lang(code):
    return code.split("-")[0].lower() if code else None


def detect_subtitle_language(cues):
    """Best-effort ISO 639-1 language of the subtitle text, or None.

    Determines whether content matching is possible: Whisper can produce the
    spoken language (transcribe) or English (translate), so we can only match
    words when the subtitle is in the spoken language or in English.
    """
    try:
        from langdetect import detect, DetectorFactory
    except Exception:
        return None
    DetectorFactory.seed = 0  # make detection deterministic
    sample = " ".join(_TAG_RE.sub(" ", t) for _, _, t in cues[:1000]).strip()
    if len(sample) < 20:
        return None
    try:
        return _norm_lang(detect(sample))
    except Exception:
        return None


def _tokens(text):
    text = _TAG_RE.sub(" ", text).lower()
    return [t for t in _WORD_RE.findall(text) if len(t) >= MIN_TOKEN_LEN]


def _interp_word_stream(cues_or_segs):
    """[(token, time)] spreading each cue/segment's words across its span.

    Used for subtitle cues (which have no word timings) and as a fallback for
    transcript segments lacking word-level timestamps.
    """
    stream = []
    for start, end, text in cues_or_segs:
        toks = _tokens(text)
        if not toks:
            continue
        span = end - start
        for k, tok in enumerate(toks):
            stream.append((tok, start + span * (k + 0.5) / len(toks)))
    return stream


def asr_word_stream(segments):
    """[(token, time)] from the transcript, preferring word-level timestamps."""
    stream = []
    for seg in segments:
        if seg["words"]:
            for word, t in seg["words"]:
                for tok in _tokens(word):
                    stream.append((tok, t))
        else:
            stream.extend(_interp_word_stream(
                [(seg["start"], seg["end"], seg["text"])]))
    return stream


def find_anchors(sub_stream, asr_stream):
    """Pair subtitle word-times to transcript word-times via shared word runs.

    Aligns the two token sequences with difflib and keeps runs of at least
    MIN_MATCH_BLOCK matching words; each matched word contributes one
    (subtitle_time, audio_time) anchor. Order is preserved, so insertions and
    deletions (translation paraphrasing, skipped on-screen text) are tolerated.
    """
    a = [w for w, _ in sub_stream]
    b = [w for w, _ in asr_stream]
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    anchors = []
    for i, j, size in sm.get_matching_blocks():
        if size >= MIN_MATCH_BLOCK:
            for k in range(size):
                anchors.append((sub_stream[i + k][1], asr_stream[j + k][1]))
    return anchors


def fuzzy_anchors(align_cues, segments):
    """Anchor subtitle cues to transcript segments by content-word overlap.

    For each dialogue cue, find the transcript segment sharing the most content
    words and accept it as an anchor when the overlap is convincing. Unlike
    word-run matching this tolerates reordering and rewording, so it recovers
    anchors from divergent translations that still mention the same names,
    numbers, and nouns. Coincidental matches at the wrong time are pruned later
    by the monotonic/slope filters.
    """
    seg_sets = [((s["start"] + s["end"]) / 2.0, set(_tokens(s["text"])))
                for s in segments]
    seg_sets = [(t, toks) for t, toks in seg_sets if toks]
    anchors = []
    for cs, ce, text in align_cues:
        cset = set(_tokens(text))
        if len(cset) < FUZZY_MIN_SHARED:
            continue
        best_time, best_shared, best_j = None, 0, 0.0
        for st, sset in seg_sets:
            shared = len(cset & sset)
            if shared < FUZZY_MIN_SHARED or shared < best_shared:
                continue
            j = shared / len(cset | sset)
            if shared > best_shared or j > best_j:
                best_time, best_shared, best_j = st, shared, j
        if best_time is not None and (
                best_shared >= FUZZY_STRONG_SHARED or best_j >= FUZZY_MIN_JACCARD):
            anchors.append(((cs + ce) / 2.0, best_time))
    return anchors


def _lis_increasing(anchors):
    """Longest subsequence (by index order) with non-decreasing audio time.

    Anchors come pre-sorted by subtitle time; a correct mapping is monotonic, so
    keeping the longest non-decreasing run in audio time discards the crossings
    that coincidental word matches produce.
    """
    if not anchors:
        return []
    ys = [y for _, y in anchors]
    tails = []        # tails[k] = index into anchors of smallest tail of an LIS of length k+1
    prev = [-1] * len(ys)
    tail_idx = []
    for i, y in enumerate(ys):
        pos = bisect.bisect_right([ys[t] for t in tail_idx], y)
        if pos == len(tail_idx):
            tail_idx.append(i)
        else:
            tail_idx[pos] = i
        prev[i] = tail_idx[pos - 1] if pos > 0 else -1
    result = []
    i = tail_idx[-1]
    while i != -1:
        result.append(anchors[i])
        i = prev[i]
    result.reverse()
    return result


def _slope_filter(anchors):
    """Drop anchors whose local slope strays from the median (mismatched words)."""
    if len(anchors) < 3:
        return anchors
    slopes = [(y1 - y0) / (x1 - x0)
              for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]) if x1 > x0]
    if not slopes:
        return anchors
    med = statistics.median(slopes)
    kept = [anchors[0]]
    for x1, y1 in anchors[1:]:
        x0, y0 = kept[-1]
        if x1 <= x0:
            continue
        s = (y1 - y0) / (x1 - x0)
        if abs(s - med) <= ANCHOR_SLOPE_TOL or abs(s - 1.0) <= ANCHOR_SLOPE_TOL:
            kept.append((x1, y1))
    return kept


def clean_anchors(anchors):
    """Reduce raw anchors to a monotonic, locally consistent set."""
    if not anchors:
        return []
    # Aggregate anchors sharing a subtitle time (rounded) to their median audio
    # time, so a single instant maps to one place before enforcing monotonicity.
    buckets = {}
    for x, y in anchors:
        buckets.setdefault(round(x, 1), []).append(y)
    merged = sorted((x, statistics.median(ys)) for x, ys in buckets.items())
    return _slope_filter(_lis_increasing(merged))


def build_time_map(anchors):
    """Piecewise-linear map subtitle_time -> audio_time from clean anchors.

    Interpolates between bracketing anchors and extrapolates beyond the ends with
    the overall slope. Returns (map_fn, global_slope).
    """
    xs = [x for x, _ in anchors]
    ys = [y for _, y in anchors]
    gslope = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if xs[-1] > xs[0] else 1.0

    def f(t):
        if t <= xs[0]:
            return ys[0] + (t - xs[0]) * gslope
        if t >= xs[-1]:
            return ys[-1] + (t - xs[-1]) * gslope
        i = bisect.bisect_right(xs, t)
        x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)

    return f, gslope


def content_align(subtitle, segments, all_cues, speech):
    """Try to build a content-based time map; return a result dict or None.

    Returns None (so the caller falls back to timing alignment) whenever the
    match is too sparse, doesn't span the timeline, implies an implausible global
    rate, or doesn't actually land the cues on speech.
    """
    align_cues, _ = dialogue_cues(all_cues)
    sub_stream = _interp_word_stream(align_cues)
    asr_stream = asr_word_stream(segments)
    if len(sub_stream) < MIN_ANCHORS or len(asr_stream) < MIN_ANCHORS:
        return None
    raw = find_anchors(sub_stream, asr_stream) + fuzzy_anchors(align_cues, segments)
    anchors = clean_anchors(raw)
    if len(anchors) < MIN_ANCHORS:
        print("[subsync]   content match: only {0} clean anchor(s) "
              "(need {1})".format(len(anchors), MIN_ANCHORS))
        return None

    f, gslope = build_time_map(anchors)
    sub_span = align_cues[-1][0] - align_cues[0][0]
    coverage = (anchors[-1][0] - anchors[0][0]) / sub_span if sub_span > 0 else 0.0
    if coverage < MIN_ANCHOR_COVERAGE:
        print("[subsync]   content match: anchors span only {0:.0f}% of the "
              "timeline (need {1:.0f}%)".format(
                  coverage * 100, MIN_ANCHOR_COVERAGE * 100))
        return None
    if not (MIN_GLOBAL_SLOPE <= gslope <= MAX_GLOBAL_SLOPE):
        print("[subsync]   content match: implausible rate {0:.3f}; "
              "rejecting".format(gslope))
        return None

    mapped = [(f(cs), f(ce), t) for cs, ce, t in align_cues]
    overlap = compute_score(mapped, speech)
    onset = onset_score(mapped, speech)
    if overlap < LOW_QUALITY_SCORE:
        print("[subsync]   content match: mapped cues only {0}% on speech; "
              "rejecting".format(overlap))
        return None
    return {"map": f, "anchors": anchors, "slope": gslope, "coverage": coverage,
            "overlap": overlap, "onset": onset}


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


def onset_score(cues, speech, tol=0.75):
    """Fraction of cue starts (0-100) that begin near a speech onset.

    compute_score (time overlap) is nearly useless when dialogue is near
    continuous: almost any placement overlaps *some* speech, so a badly stretched
    timeline still scores high. Onset agreement is discriminative -- a correct
    sync has cues starting right as someone begins speaking, while a wrong global
    scale smears cue starts away from the speech edges it should hug.
    """
    if not cues or not speech:
        return 0
    import bisect
    starts = sorted(ss for ss, _ in speech)
    hit = 0
    for cs, _ce, _t in cues:
        i = bisect.bisect_left(starts, cs)
        nearest = min(
            (abs(cs - starts[j]) for j in (i - 1, i) if 0 <= j < len(starts)),
            default=float("inf"),
        )
        if nearest <= tol:
            hit += 1
    return round(hit / len(cues) * 100)


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


def write_report(video, subtitle, synced, cues, speech, score, strategy="default",
                 ff_score=None, ff_offset=None, trustworthy=True,
                 dropped=0, orig_count=None, overlap=None, coverage=None):
    _clean_old_reports(video)
    report = video.with_name("{0}.report.{1:03d}.txt".format(video.stem, score))

    quality = "ok" if trustworthy else "LOW - sync may be wrong"
    if orig_count is None:
        orig_count = len(cues)
    if overlap is None:
        overlap = score
    if coverage is None:
        coverage = 1.0
    ff_offset_desc = "n/a" if ff_offset is None else "{0:.3f}s".format(ff_offset)
    cues_desc = "{0}".format(len(cues))
    if dropped > 0:
        cues_desc += " ({0} of {1} clamped to 0:00)".format(dropped, orig_count)
    lines = [
        "subtitle sync report",
        "",
        "video:     {0}".format(video.name),
        "subtitle:  {0}".format(subtitle.name),
        "synced:    {0}".format(synced.name),
        "reference: whisper {0}, {1} dialogue segment(s)".format(
            WHISPER_MODEL, len(speech)),
        "strategy:  {0}".format(strategy),
        "ffsubsync: score {0}, offset {1}".format(_fmt_ff_score(ff_score), ff_offset_desc),
        "score:     {0}% ({1})".format(score, quality),
        "           overlap {0}% x coverage {1:.0f}%".format(overlap, coverage * 100),
        "cues:      {0}".format(cues_desc),
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
            try:
                validate_subtitle(sub)
            except ValueError as exc:
                sys.exit("error: {0}: {1}".format(sub.name, exc))
            pairs.append((video, sub))
        return pairs
    found = find_pairs(Path.cwd())
    if not found:
        sys.exit(
            "no video+subtitle pairs found in the current directory.\n"
            "usage: subtitle_sync [video ...]"
        )
    pairs = []
    for video, sub in found:
        try:
            validate_subtitle(sub)
        except ValueError as exc:
            print("[subsync] skipping {0}: {1}".format(sub.name, exc))
            continue
        pairs.append((video, sub))
    if not pairs:
        sys.exit("no valid video+subtitle pairs to sync.")
    return pairs


# ---------- sync ----------

_FF_SCORE_RE = re.compile(r"\bscore:\s*(-?\d+(?:\.\d+)?)")
_FF_OFFSET_RE = re.compile(r"offset seconds:\s*(-?\d+(?:\.\d+)?)")
_FF_SCALE_RE = re.compile(r"framerate scale factor:\s*(-?\d+(?:\.\d+)?)")


def _run_ffsubsync(reference, subtitle, out_path, extra_args):
    """Run ffsubsync, returning (ff_score, ff_offset, ff_scale) from its output.

    `reference` is the .npz speech array built from the Whisper timeline.
    ffsubsync ships a console entry point but no runnable __main__, so invoke
    the binary installed alongside the (venv) interpreter rather than -m. The
    final timing it computes is new_t = ff_scale*t + ff_offset; we recover both
    so the transform can be re-applied to the full subtitle. Its score is not
    comparable across files, but the sign matters: negative means the alignment
    is anti-correlated (wrong). Any value may be None if it could not be parsed.
    """
    ffsubsync_bin = Path(sys.executable).with_name("ffsubsync")
    result = subprocess.run(
        [str(ffsubsync_bin), str(reference), "-i", str(subtitle), "-o", str(out_path)]
        + list(extra_args),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, check=True,
    )
    out = result.stdout or ""
    scores = _FF_SCORE_RE.findall(out)
    offsets = _FF_OFFSET_RE.findall(out)
    scales = _FF_SCALE_RE.findall(out)
    ff_score = float(scores[-1]) if scores else None
    ff_offset = float(offsets[-1]) if offsets else None
    ff_scale = float(scales[-1]) if scales else None
    return ff_score, ff_offset, ff_scale


def _trust_tier(ff_score, ff_scale=None):
    # 2 = ffsubsync trusts it, 1 = unknown (unparsed), 0 = anti-correlated.
    # A framerate scale outside the plausible band is bogus regardless of score.
    if ff_scale is not None and abs(ff_scale - 1.0) > MAX_SCALE_DEVIATION:
        return 0
    if ff_score is None:
        return 1
    return 2 if ff_score > MIN_FFSUBSYNC_SCORE else 0


def _fmt_ff_score(ff_score):
    return "n/a" if ff_score is None else "{0:.0f}".format(ff_score)


def sync_one(video, subtitle, speech):
    ext = subtitle.suffix
    synced = video.with_name(video.stem + SYNCED_TAG + ext)
    print("[subsync] syncing {0} -> {1}".format(subtitle.name, synced.name))
    if not speech:
        sys.exit("error: no dialogue detected in {0}; cannot sync".format(video.name))

    all_cues = parse_cues(subtitle)
    orig_count = len(all_cues)
    duration = _get_duration(video)

    # Align against dialogue cues only; on-screen-text captions don't track
    # speech and would just add noise. The recovered transform is applied to the
    # full file, so captions are re-timed too -- they're only held out of the fit.
    align_cues, filtered = dialogue_cues(all_cues)
    if filtered:
        print("[subsync]   aligning on {0} dialogue cue(s) ({1} caption(s) held "
              "out of the fit)".format(len(align_cues), orig_count - len(align_cues)))

    best = None  # candidate dict; ranked by (tier, effective, onset)
    tmp_dir = Path(tempfile.mkdtemp(prefix="subsync_"))
    try:
        reference = tmp_dir / "reference.npz"
        build_reference(speech, duration, reference)
        align_srt = tmp_dir / "align.srt"
        write_alignment_srt(align_cues, align_srt)
        for label, extra in ALIGN_STRATEGIES:
            print("[subsync]   trying strategy: {0}".format(label))
            ff_out = tmp_dir / ("ff_out.srt")  # ffsubsync's own output, unused
            try:
                ff_score, ff_offset, ff_scale = _run_ffsubsync(
                    reference, align_srt, ff_out, extra)
            except subprocess.CalledProcessError as exc:
                print("[subsync]   strategy {0} failed ({1}); skipping".format(label, exc))
                continue
            if ff_offset is None:
                print("[subsync]   strategy {0}: could not parse an offset; "
                      "skipping".format(label))
                continue
            scale = ff_scale if ff_scale is not None else 1.0
            # Re-apply the recovered transform to the *full* subtitle.
            candidate = tmp_dir / ("candidate" + ext)
            clamped = apply_time_transform(subtitle, candidate, scale, ff_offset)
            cues = parse_cues(candidate)
            dcues, _ = dialogue_cues(cues)
            overlap = compute_score(dcues, speech)
            onset = onset_score(dcues, speech)
            # Cues clamped to 0:00 were shifted before the start; treat them as a
            # coverage penalty so a wrong, large negative offset can't score well.
            coverage = 1.0 - (clamped / float(orig_count) if orig_count else 0.0)
            effective = int(round(overlap * coverage))
            tier = _trust_tier(ff_score, ff_scale)
            print("[subsync]   strategy {0}: score {1}% (overlap {2}%, onset {3}%, "
                  "coverage {4:.0f}%), ffsubsync score {5}, offset {6}, scale {7}".format(
                      label, effective, overlap, onset, coverage * 100,
                      _fmt_ff_score(ff_score),
                      "n/a" if ff_offset is None else "{0:.1f}s".format(ff_offset),
                      "{0:.3f}".format(scale)))
            cand = {"overlap": overlap, "onset": onset, "coverage": coverage,
                    "effective": effective, "cues": cues, "label": label,
                    "tier": tier, "ff_score": ff_score, "ff_offset": ff_offset,
                    "ff_scale": scale, "clamped": clamped, "path": None}
            # Prefer an alignment we trust; break ties by effective, then onset.
            if best is None or (tier, effective, onset) > (
                    best["tier"], best["effective"], best["onset"]):
                kept = tmp_dir / ("best" + ext)
                shutil.copyfile(str(candidate), str(kept))
                cand["path"] = kept
                best = cand
            # Only stop early when ffsubsync trusts it AND nearly all cues survived.
            if tier == 2 and effective >= EARLY_ACCEPT_SCORE:
                break
        if best is None:
            sys.exit("error: all alignment strategies failed for {0}".format(subtitle.name))
        shutil.copyfile(str(best["path"]), str(synced))
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    effective = best["effective"]
    clamped = best["clamped"]
    trustworthy = best["tier"] == 2 and effective >= LOW_QUALITY_SCORE
    print("[subsync] wrote {0} (strategy: {1}, score {2}%, overlap {3}%, onset {4}%, "
          "scale {5:.3f}, ffsubsync score {6})".format(
              synced.name, best["label"], effective, best["overlap"], best["onset"],
              best["ff_scale"], _fmt_ff_score(best["ff_score"])))
    if not trustworthy:
        print(
            "[subsync] WARNING: alignment for {0} looks unreliable (score {1}%, "
            "scale {2:.3f}, ffsubsync score {3}); the sync may be wrong.".format(
                subtitle.name, effective, best["ff_scale"],
                _fmt_ff_score(best["ff_score"]))
        )
    if clamped > 0:
        print(
            "[subsync] WARNING: {0} of {1} cue(s) shifted before 0:00 and were "
            "clamped to the start.".format(clamped, orig_count)
        )

    report = write_report(video, subtitle, synced, best["cues"], speech, effective,
                          best["label"], best["ff_score"], best["ff_offset"],
                          trustworthy, clamped, orig_count, best["overlap"],
                          best["coverage"])
    print("[subsync] report: {0} (score {1}%)".format(report.name, effective))


def write_content_sync(video, subtitle, result, speech, all_cues):
    """Write the synced file from a content-based time map and report it."""
    ext = subtitle.suffix
    synced = video.with_name(video.stem + SYNCED_TAG + ext)
    print("[subsync] syncing {0} -> {1}".format(subtitle.name, synced.name))
    orig_count = len(all_cues)
    clamped = rewrite_timestamps(subtitle, synced, result["map"])

    overlap = result["overlap"]
    onset = result["onset"]
    coverage = 1.0 - (clamped / float(orig_count) if orig_count else 0.0)
    score = int(round(overlap * coverage))
    label = "content ({0} anchors, slope {1:.3f}, span {2:.0f}%)".format(
        len(result["anchors"]), result["slope"], result["coverage"] * 100)
    trustworthy = score >= LOW_QUALITY_SCORE
    print("[subsync] wrote {0} (strategy: {1}, score {2}%, overlap {3}%, "
          "onset {4}%)".format(synced.name, label, score, overlap, onset))
    if clamped > 0:
        print("[subsync] WARNING: {0} of {1} cue(s) shifted before 0:00 and were "
              "clamped to the start.".format(clamped, orig_count))

    report = write_report(video, subtitle, synced, parse_cues(synced), speech,
                          score, label, None, None, trustworthy, clamped,
                          orig_count, overlap, coverage)
    print("[subsync] report: {0} (score {1}%)".format(report.name, score))


# ---------- driver ----------

def sync_pair(video, subtitle):
    """Sync one video+subtitle pair, choosing content vs timing alignment.

    Content matching (Tier 2) is attempted only when the subtitle is in a
    language Whisper can produce -- the spoken language (transcribe) or English
    (translate) -- and is built from matched words into a drift-correcting
    piecewise map. When that isn't possible or isn't confident, we fall back to
    Tier 1: aligning the dialogue cues' timing against the Whisper speech track.
    """
    all_cues = parse_cues(subtitle)

    # A truncated/incomplete video, or a subtitle for a longer cut, leaves part of
    # the subtitle with no audio to match -- warn before spending time on Whisper.
    warn_truncated_media(video, all_cues, _get_duration(video), real_media_end(video))

    sub_lang = detect_subtitle_language(all_cues)
    # If the subtitle is English we can always get matching English text via
    # Whisper's translate task; otherwise we transcribe and can match only if the
    # spoken language turns out to be the subtitle's language.
    task = "translate" if sub_lang == "en" else "transcribe"
    print("[subsync] subtitle language: {0}; whisper task: {1}".format(
        sub_lang or "unknown", task))

    segments, audio_lang = transcribe_dialogue(video, task=task)
    speech = speech_ranges(segments)
    if not speech:
        sys.exit("error: no dialogue detected in {0}; cannot sync".format(video.name))

    can_match = sub_lang == "en" or (sub_lang is not None and sub_lang == _norm_lang(audio_lang))
    if can_match:
        print("[subsync] attempting content-based alignment")
        result = content_align(subtitle, segments, all_cues, speech)
        if result is not None:
            write_content_sync(video, subtitle, result, speech, all_cues)
            return
        print("[subsync] content match not confident; falling back to timing "
              "alignment")
    else:
        print("[subsync] subtitle language not matchable to audio; using timing "
              "alignment")
    sync_one(video, subtitle, speech)


def main():
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) != len(sys.argv[1:]):
        print("usage: subtitle_sync [video ...]")
        print("  Re-time subtitle files to their video's dialogue (located with")
        print("  Whisper). With no arguments, processes every video+subtitle pair")
        print("  in the current directory.")
        return

    pairs = resolve_inputs(args)
    print("[subsync] {0} pair(s) to sync:".format(len(pairs)))
    for video, sub in pairs:
        print("        {0} + {1}".format(video.name, sub.name))
    print()

    for video, sub in pairs:
        sync_pair(video, sub)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
