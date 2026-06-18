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
PIP_PACKAGES = ["ffsubsync", "faster-whisper", "numpy"]

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

# Whisper model used to locate dialogue. large-v3 is the most accurate (and
# slowest); tiny/base/small/medium trade accuracy for speed.
WHISPER_MODEL = "large-v3"

# ffsubsync samples speech at 100 Hz (see ffsubsync.constants.SAMPLE_RATE); the
# reference array we hand it must use the same rate.
REFERENCE_HZ = 100

# Alignment passes tried against the Whisper-derived reference, in order. The
# reference (the expensive Whisper step) is built once; these only re-run the
# near-instant ffsubsync offset/framerate search. gss adds a golden-section
# search for a framerate-ratio mismatch (common with PAL/NTSC broadcast subs).
ALIGN_STRATEGIES = [
    ("whisper-ref", []),
    ("whisper-ref+gss", ["--gss"]),
]
# Our overlap score (0-100) at/above which a pass is accepted without trying
# the rest, and below which the final result is flagged as unreliable. Overlap
# is the fraction of cue time landing on Whisper-detected dialogue.
EARLY_ACCEPT_SCORE = 90
LOW_QUALITY_SCORE = 75
# ffsubsync's alignment score is not comparable across files, but its sign is
# meaningful: a value at or below this is anti-correlated, i.e. a wrong sync.
MIN_FFSUBSYNC_SCORE = 0.0


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
    # utf-8-sig transparently strips a leading BOM (common in subtitles
    # exported from Windows tools) while still decoding plain UTF-8.
    text = path.read_text(encoding="utf-8-sig")
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


# ---------- speech detection ----------

def _get_duration(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, check=True,
    )
    return float(result.stdout.strip())


def detect_speech_ranges(video):
    """Locate spoken dialogue in a video, returning [(start, end)] in seconds.

    Whisper transcribes the audio with its own VAD filter, so the segments it
    returns track actual speech and skip music / laughter / sound effects --
    the failure mode of energy- or webrtc-based detection on dense content. We
    keep only the segment timings; the transcribed text is discarded (we sync
    the user's existing subtitle, we don't replace it).
    """
    from faster_whisper import WhisperModel

    print("[subsync] loading whisper model: {0} (downloads on first use)".format(
        WHISPER_MODEL))
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("[subsync] transcribing {0} to locate dialogue (this can take a "
          "while)".format(video.name))
    segments, info = model.transcribe(
        str(video),
        vad_filter=True,
        condition_on_previous_text=False,
    )
    # faster-whisper yields segments lazily as the audio is processed, so we can
    # report progress by comparing each segment's end against the total media
    # duration. info.duration is the portion Whisper will actually scan (after
    # VAD), which is what the timestamps run up to.
    total = float(getattr(info, "duration", 0.0)) or 0.0
    speech = []
    last_pct = -1
    stderr_tty = sys.stderr.isatty()
    for seg in segments:
        if seg.text and seg.text.strip():
            speech.append((float(seg.start), float(seg.end)))
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
    print("[subsync] found {0} dialogue segment(s)".format(len(speech)))
    return speech


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
        cues_desc += " ({0} of {1} dropped before 0:00)".format(dropped, orig_count)
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


def _run_ffsubsync(reference, subtitle, out_path, extra_args):
    """Run ffsubsync, returning (ff_score, ff_offset) parsed from its output.

    `reference` is the .npz speech array built from the Whisper timeline.
    ffsubsync ships a console entry point but no runnable __main__, so invoke
    the binary installed alongside the (venv) interpreter rather than -m. Its
    score is not comparable across files, but the sign matters: negative means
    the alignment is anti-correlated (wrong). Either value may be None if it
    could not be parsed.
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
    ff_score = float(scores[-1]) if scores else None
    ff_offset = float(offsets[-1]) if offsets else None
    return ff_score, ff_offset


def _trust_tier(ff_score):
    # 2 = ffsubsync trusts it, 1 = unknown (unparsed), 0 = anti-correlated.
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

    orig_count = len(parse_cues(subtitle))
    duration = _get_duration(video)

    best = None  # candidate dict; ranked by (tier, overlap)
    tmp_dir = Path(tempfile.mkdtemp(prefix="subsync_"))
    try:
        reference = tmp_dir / "reference.npz"
        build_reference(speech, duration, reference)
        for label, extra in ALIGN_STRATEGIES:
            candidate = tmp_dir / ("candidate" + ext)
            print("[subsync]   trying strategy: {0}".format(label))
            try:
                ff_score, ff_offset = _run_ffsubsync(reference, subtitle, candidate, extra)
            except subprocess.CalledProcessError as exc:
                print("[subsync]   strategy {0} failed ({1}); skipping".format(label, exc))
                continue
            if not candidate.exists():
                print("[subsync]   strategy {0} produced no output; skipping".format(label))
                continue
            cues = parse_cues(candidate)
            overlap = compute_score(cues, speech)
            # Cues shifted before 0:00 are dropped by ffsubsync; scoring only the
            # survivors would reward a wildly wrong offset that keeps just one
            # well-placed cue. Fold coverage in so losing cues tanks the score.
            coverage = len(cues) / float(orig_count) if orig_count else 0.0
            effective = int(round(overlap * coverage))
            tier = _trust_tier(ff_score)
            print("[subsync]   strategy {0}: score {1}% (overlap {2}%, coverage "
                  "{3:.0f}%), ffsubsync score {4}, offset {5}".format(
                      label, effective, overlap, coverage * 100,
                      _fmt_ff_score(ff_score),
                      "n/a" if ff_offset is None else "{0:.1f}s".format(ff_offset)))
            cand = {"overlap": overlap, "coverage": coverage, "effective": effective,
                    "cues": cues, "label": label, "tier": tier,
                    "ff_score": ff_score, "ff_offset": ff_offset, "path": None}
            # Prefer an alignment ffsubsync trusts; break ties by effective score.
            if best is None or (tier, effective) > (best["tier"], best["effective"]):
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
    dropped = orig_count - len(best["cues"])
    trustworthy = best["tier"] == 2 and effective >= LOW_QUALITY_SCORE
    print("[subsync] wrote {0} (strategy: {1}, score {2}%, overlap {3}%, "
          "ffsubsync score {4})".format(
              synced.name, best["label"], effective, best["overlap"],
              _fmt_ff_score(best["ff_score"])))
    if not trustworthy:
        print(
            "[subsync] WARNING: alignment for {0} looks unreliable (score {1}%, "
            "ffsubsync score {2}); the sync may be wrong.".format(
                subtitle.name, effective, _fmt_ff_score(best["ff_score"]))
        )
    if dropped > 0:
        print(
            "[subsync] WARNING: {0} of {1} cue(s) fell before 0:00 after shifting "
            "and were dropped.".format(dropped, orig_count)
        )

    report = write_report(video, subtitle, synced, best["cues"], speech, effective,
                          best["label"], best["ff_score"], best["ff_offset"],
                          trustworthy, dropped, orig_count, best["overlap"],
                          best["coverage"])
    print("[subsync] report: {0} (score {1}%)".format(report.name, effective))


# ---------- driver ----------

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
        speech = detect_speech_ranges(video)
        sync_one(video, sub, speech)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
