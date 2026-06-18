#!/usr/bin/env python3
"""Synchronize subtitle timing to match video audio.

Given video files and their associated subtitle files (.srt or .vtt),
uses ffsubsync to re-time the subtitles to match the audio. Writes the
synced subtitle next to the original with a .synced suffix and a report
file named <stem>.report.<score>.txt where score is 000-100.

Runs on Python 3.6+ so it works on older distros (e.g. CentOS 7's EPEL
python3) as well as current ones. Missing runtime pieces (pip, ffmpeg)
are force-installed into a per-user cache without requiring root.

Author: Chun Kang <kurapa@kurapa.com>
"""

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
PIP_PACKAGES = ["ffsubsync"]

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

# Alignment strategies tried in order. We score each result against our own
# speech detection and keep the best, stopping early once one is good enough.
#   - default (subs_then_webrtc) uses any embedded subtitle track in the video
#     as the reference, which is great when it exists and correctly timed but
#     misleads when it doesn't (a common cause of "synced but totally off").
#   - webrtc / auditok force pure audio-based detection, sidestepping that.
#   - gss adds golden-section search for a framerate ratio mismatch.
ALIGN_STRATEGIES = [
    ("default", []),
    ("vad=webrtc", ["--vad", "webrtc"]),
    ("vad=auditok", ["--vad", "auditok"]),
    ("gss", ["--gss"]),
]
# Our overlap score (0-100) at/above which a strategy is accepted without
# trying the rest, and below which the final result is flagged as unreliable.
EARLY_ACCEPT_SCORE = 90
LOW_QUALITY_SCORE = 75


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
    print("[subsync] detecting speech ranges in {0}".format(video.name))
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video),
         "-af", "silencedetect=noise=-30dB:d=0.5",
         "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    silence_starts = []
    silence_ranges = []
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start:\s*(\S+)", line)
        if m:
            silence_starts.append(float(m.group(1)))
        m = re.search(r"silence_end:\s*(\S+)", line)
        if m and silence_starts:
            silence_ranges.append((silence_starts.pop(0), float(m.group(1))))

    duration = _get_duration(video)
    speech = []
    prev = 0.0
    for ss, se in sorted(silence_ranges):
        if ss > prev:
            speech.append((prev, ss))
        prev = se
    if prev < duration:
        speech.append((prev, duration))
    return speech


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


def write_report(video, subtitle, synced, cues, speech, score, strategy="default"):
    _clean_old_reports(video)
    report = video.with_name("{0}.report.{1:03d}.txt".format(video.stem, score))

    quality = "ok" if score >= LOW_QUALITY_SCORE else "LOW - sync may be unreliable"
    lines = [
        "subtitle sync report",
        "",
        "video:    {0}".format(video.name),
        "subtitle: {0}".format(subtitle.name),
        "synced:   {0}".format(synced.name),
        "strategy: {0}".format(strategy),
        "score:    {0}% ({1})".format(score, quality),
        "cues:     {0}".format(len(cues)),
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

def _run_ffsubsync(video, subtitle, out_path, extra_args):
    # ffsubsync ships a console entry point but no runnable __main__, so invoke
    # the binary installed alongside the (venv) interpreter rather than -m.
    ffsubsync_bin = Path(sys.executable).with_name("ffsubsync")
    subprocess.run(
        [str(ffsubsync_bin), str(video), "-i", str(subtitle), "-o", str(out_path)]
        + list(extra_args),
        check=True,
    )


def sync_one(video, subtitle, speech):
    ext = subtitle.suffix
    synced = video.with_name(video.stem + SYNCED_TAG + ext)
    print("[subsync] syncing {0} -> {1}".format(subtitle.name, synced.name))

    best = None  # {"score", "cues", "label", "path"}
    tmp_dir = Path(tempfile.mkdtemp(prefix="subsync_"))
    try:
        for label, extra in ALIGN_STRATEGIES:
            candidate = tmp_dir / ("candidate" + ext)
            print("[subsync]   trying strategy: {0}".format(label))
            try:
                _run_ffsubsync(video, subtitle, candidate, extra)
            except subprocess.CalledProcessError as exc:
                print("[subsync]   strategy {0} failed ({1}); skipping".format(label, exc))
                continue
            if not candidate.exists():
                print("[subsync]   strategy {0} produced no output; skipping".format(label))
                continue
            cues = parse_cues(candidate)
            score = compute_score(cues, speech)
            print("[subsync]   strategy {0} scored {1}%".format(label, score))
            if best is None or score > best["score"]:
                # Stash a copy; the next strategy overwrites `candidate`.
                kept = tmp_dir / ("best" + ext)
                shutil.copyfile(str(candidate), str(kept))
                best = {"score": score, "cues": cues, "label": label, "path": kept}
            if score >= EARLY_ACCEPT_SCORE:
                break
        if best is None:
            sys.exit("error: all alignment strategies failed for {0}".format(subtitle.name))
        shutil.copyfile(str(best["path"]), str(synced))
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    score = best["score"]
    label = best["label"]
    print("[subsync] wrote {0} (strategy: {1}, score: {2}%)".format(synced.name, label, score))
    if score < LOW_QUALITY_SCORE:
        print(
            "[subsync] WARNING: best alignment for {0} scored only {1}%; the sync "
            "may be unreliable.".format(subtitle.name, score)
        )

    report = write_report(video, subtitle, synced, best["cues"], speech, score, label)
    print("[subsync] report: {0} (score: {1}%)".format(report.name, score))


# ---------- driver ----------

def main():
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) != len(sys.argv[1:]):
        print("usage: subtitle_sync [video ...]")
        print("  Sync subtitle files to their video's audio using ffsubsync.")
        print("  With no arguments, processes every video+subtitle pair in the")
        print("  current directory.")
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
