#!/usr/bin/env python3
"""Synchronize subtitle timing to match video audio.

Given video files and their associated subtitle files (.srt or .vtt),
uses ffsubsync to re-time the subtitles to match the audio. Writes the
synced subtitle next to the original with a .synced suffix and a report
file named <stem>.report.<score>.txt where score is 000-100.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path.home() / ".cache" / "subtitle_sync" / "venv"
VENV_MARKER = "SUBSYNC_IN_VENV"
PIP_PACKAGES: list[str] = ["ffsubsync"]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
SUBTITLE_EXTS = (".srt", ".vtt")
SYNCED_TAG = ".synced"


# ---------- bootstrap ----------

def _ensure_venv_and_reexec() -> None:
    if os.environ.get(VENV_MARKER) == "1":
        return
    if not VENV_DIR.exists():
        print(f"[subsync] creating venv at {VENV_DIR}")
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        venv.create(VENV_DIR, with_pip=True)
    venv_python = VENV_DIR / "bin" / "python3"
    if PIP_PACKAGES:
        pip = VENV_DIR / "bin" / "pip"
        missing = [p for p in PIP_PACKAGES if subprocess.run(
            [str(pip), "show", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode != 0]
        if missing:
            print(f"[subsync] installing: {', '.join(missing)}")
            subprocess.check_call([str(pip), "install", "-q", *missing])
    env = {**os.environ, VENV_MARKER: "1"}
    os.execve(str(venv_python), [str(venv_python), os.path.abspath(__file__), *sys.argv[1:]], env)


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if shutil.which("brew"):
        print("[subsync] installing ffmpeg via Homebrew")
        subprocess.check_call(["brew", "install", "ffmpeg"])
        return
    sys.exit("error: ffmpeg/ffprobe not found and Homebrew is unavailable; install ffmpeg manually")


# ---------- subtitle parsing ----------

def _parse_srt_ts(ts: str) -> float:
    h, m, rest = ts.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_vtt_ts(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, rest = parts
    else:
        h = "0"
        m, rest = parts
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_cues(path: Path) -> list[tuple[float, float, str]]:
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

    cues: list[tuple[float, float, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = arrow_re.search(lines[i])
        if m:
            start = parse_ts(m.group(1))
            end = parse_ts(m.group(2))
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip():
                body.append(lines[i].strip())
                i += 1
            cues.append((start, end, " ".join(body)))
        else:
            i += 1
    return cues


# ---------- speech detection ----------

def _get_duration(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def detect_speech_ranges(video: Path) -> list[tuple[float, float]]:
    print(f"[subsync] detecting speech ranges in {video.name}")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video),
         "-af", "silencedetect=noise=-30dB:d=0.5",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    silence_starts: list[float] = []
    silence_ranges: list[tuple[float, float]] = []
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start:\s*(\S+)", line)
        if m:
            silence_starts.append(float(m.group(1)))
        m = re.search(r"silence_end:\s*(\S+)", line)
        if m and silence_starts:
            silence_ranges.append((silence_starts.pop(0), float(m.group(1))))

    duration = _get_duration(video)
    speech: list[tuple[float, float]] = []
    prev = 0.0
    for ss, se in sorted(silence_ranges):
        if ss > prev:
            speech.append((prev, ss))
        prev = se
    if prev < duration:
        speech.append((prev, duration))
    return speech


# ---------- scoring & report ----------

def compute_score(cues: list[tuple[float, float, str]],
                  speech: list[tuple[float, float]]) -> int:
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


def _fmt_ts(t: float) -> str:
    m = int(t // 60)
    s = t % 60
    return f"{m:02d}:{s:05.2f}"


def _cue_hits_speech(cs: float, ce: float,
                     speech: list[tuple[float, float]]) -> bool:
    return any(cs < se and ce > ss for ss, se in speech)


def _clean_old_reports(video: Path) -> None:
    pattern = re.compile(re.escape(video.stem) + r"\.report\.\d{3}\.txt$")
    for f in video.parent.iterdir():
        if f.is_file() and pattern.match(f.name):
            f.unlink()


def write_report(video: Path, subtitle: Path, synced: Path,
                 cues: list[tuple[float, float, str]],
                 speech: list[tuple[float, float]],
                 score: int) -> Path:
    _clean_old_reports(video)
    report = video.with_name(f"{video.stem}.report.{score:03d}.txt")

    lines = [
        "subtitle sync report",
        "",
        f"video:    {video.name}",
        f"subtitle: {subtitle.name}",
        f"synced:   {synced.name}",
        f"score:    {score}%",
        f"cues:     {len(cues)}",
        "",
    ]
    for i, (start, end, text) in enumerate(cues, 1):
        hit = _cue_hits_speech(start, end, speech)
        marker = "+" if hit else "-"
        preview = text[:60]
        lines.append(
            f"  {marker} {i:3d}  [{_fmt_ts(start)} -> {_fmt_ts(end)}]  {preview}"
        )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# ---------- input discovery ----------

def is_synced_output(p: Path) -> bool:
    return p.stem.endswith(SYNCED_TAG)


def find_subtitle_for_video(video: Path) -> Path | None:
    for ext in SUBTITLE_EXTS:
        sub = video.with_suffix(ext)
        if sub.exists() and not is_synced_output(sub):
            return sub
    return None


def find_pairs(directory: Path) -> list[tuple[Path, Path]]:
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


def resolve_inputs(args: list[str]) -> list[tuple[Path, Path]]:
    if args:
        pairs = []
        for a in args:
            video = Path(a)
            if not video.exists():
                sys.exit(f"error: no such file: {a}")
            if not video.is_file():
                sys.exit(f"error: not a file: {a}")
            sub = find_subtitle_for_video(video)
            if not sub:
                sys.exit(
                    f"error: no subtitle file found for {video.name}\n"
                    f"       expected {video.stem}.srt or {video.stem}.vtt"
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

def sync_one(video: Path, subtitle: Path,
             speech: list[tuple[float, float]]) -> None:
    ext = subtitle.suffix
    synced = video.with_name(video.stem + SYNCED_TAG + ext)
    print(f"[subsync] syncing {subtitle.name} -> {synced.name}")
    subprocess.run(
        [sys.executable, "-m", "ffsubsync",
         str(video), "-i", str(subtitle), "-o", str(synced)],
        check=True,
    )
    print(f"[subsync] wrote {synced.name}")

    cues = parse_cues(synced)
    score = compute_score(cues, speech)
    report = write_report(video, subtitle, synced, cues, speech, score)
    print(f"[subsync] report: {report.name} (score: {score}%)")


# ---------- driver ----------

def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) != len(sys.argv[1:]):
        print("usage: subtitle_sync [video ...]")
        print("  Sync subtitle files to their video's audio using ffsubsync.")
        print("  With no arguments, processes every video+subtitle pair in the")
        print("  current directory.")
        return

    pairs = resolve_inputs(args)
    print(f"[subsync] {len(pairs)} pair(s) to sync:")
    for video, sub in pairs:
        print(f"        {video.name} + {sub.name}")
    print()

    for video, sub in pairs:
        speech = detect_speech_ranges(video)
        sync_one(video, sub, speech)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
