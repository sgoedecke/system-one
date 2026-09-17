#!/usr/bin/env python3
"""Compose captured Doom frames and recorded categorical decisions into an MP4.

Requires Pillow and ffmpeg with libx264 (or imageio_ffmpeg). No GPU required.
--fps changes output frame rate by resampling, never playback speed.
--limit-frames limits source frames, before output-rate conversion.
Optional audio.wav is muxed as AAC, padded or trimmed to the video duration.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it in your rendering environment: "
        "python -m pip install Pillow"
    ) from exc


WIDTH, HEIGHT = 1920, 1080
GAME_BOX = (40, 232, 1040, 982)
HEAD_ORDER = ("goal", "target", "dodge", "move", "turn", "fire", "weapon", "use")
BG = "#090f19"
PANEL = "#111e2d"
STROKE = "#24394b"
WHITE = "#edf7ff"
MUTED = "#8ca4b9"
CYAN = "#66d9f0"
MINT = "#8bf2c6"


def finite_number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return value


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def read_capture(root, limit):
    with (root / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    source_fps = finite_number(metadata.get("fps"), "metadata fps", 0.001)
    for key in ("width", "height"):
        if integer(metadata.get(key), f"metadata {key}") == 0:
            raise ValueError(f"metadata {key} must be positive")
    if (metadata["width"], metadata["height"]) != (640, 480):
        raise ValueError("This renderer expects 640x480 source images")
    if metadata.get("model") not in ("Qwen/Qwen3-8B", "Qwen3-8B"):
        raise ValueError("metadata model must be Qwen/Qwen3-8B")
    frames = sorted((root / "frames").glob("[0-9][0-9][0-9][0-9][0-9][0-9].jpg"))
    if limit is not None:
        frames = frames[:limit]
    if not frames:
        raise ValueError(f"No numbered JPEG frames found in {root / 'frames'}")
    for index, path in enumerate(frames):
        if path.name != f"{index:06d}.jpg":
            raise ValueError(f"Missing source frame {index:06d}.jpg; found {path.name}")
    decisions = []
    with (root / "decisions.jsonl").open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            label = f"decisions.jsonl line {line_number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                integer(row.get("frame"), "frame")
                integer(row.get("batch"), "batch")
                if "episode" in row:
                    integer(row["episode"], "episode")
                target = row.get("active_target")
                if target is not None and not isinstance(target, str):
                    if not isinstance(target, dict) or not isinstance(target.get("name"), str):
                        raise ValueError("active_target must be a name or an object with a name")
                observation = row.get("observation", {})
                if not isinstance(observation, dict):
                    raise ValueError("observation must be an object")
                for key in ("armor", "shells", "bullets", "visited_cells"):
                    if key in observation:
                        finite_number(observation[key], f"observation.{key}")
                finite_number(row.get("latency_ms"), "latency_ms")
                if decisions and row["frame"] <= decisions[-1]["frame"]:
                    raise ValueError("decision frames must be strictly increasing")
                answers = row.get("answers")
                if not isinstance(answers, dict):
                    raise ValueError("answers must be an object")
                if not 1 <= len(answers) <= 8:
                    raise ValueError("answers must contain 1–8 question heads")
                for key, answer in answers.items():
                    if not isinstance(answer, dict):
                        raise ValueError(f"{key} must be an answer object")
                    probabilities = answer.get("probabilities")
                    if not isinstance(probabilities, dict) or not 1 <= len(probabilities) <= 10:
                        raise ValueError(f"{key}.probabilities must have 1–10 named options")
                    if any(not isinstance(option, str) or not option for option in probabilities):
                        raise ValueError(f"{key} options must be nonempty strings")
                    if answer.get("choice") not in probabilities:
                        raise ValueError(f"{key}.choice must occur in its probabilities")
                    for option, probability in probabilities.items():
                        finite_number(probability, f"{key}.{option}")
                        if probability > 1:
                            raise ValueError(f"{key}.{option} must be <= 1")
                    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.03):
                        raise ValueError(f"{key} probabilities must sum to 1 (±0.03)")
                stats = row.get("stats", {})
                if not isinstance(stats, dict):
                    raise ValueError("stats must be an object")
                for key in ("health", "ammo", "kills"):
                    value = row.get(key, stats.get(key))
                    if value is not None:
                        finite_number(value, key, minimum=-1_000_000)
                decisions.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{label}: {exc}") from exc
    keys = set().union(*(row["answers"] for row in decisions)) if decisions else set()
    if len(keys) > 8:
        raise ValueError("Capture contains more than eight distinct question heads")
    return source_fps, frames, decisions, metadata


def load_fonts():
    families = (
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    )
    for regular, bold in families:
        if Path(regular).is_file() and Path(bold).is_file():
            return {
                "title": ImageFont.truetype(bold, 44),
                "question": ImageFont.truetype(bold, 28),
                "answer": ImageFont.truetype(bold, 24),
                "body": ImageFont.truetype(regular, 23),
                "small": ImageFont.truetype(regular, 20),
                "label": ImageFont.truetype(bold, 18),
                "compact": ImageFont.truetype(regular, 18),
                "tiny": ImageFont.truetype(regular, 16),
            }
    raise RuntimeError(
        "No supported font family found. Install DejaVu Sans or Liberation Sans "
        "under /usr/share/fonts/truetype, or use macOS Arial."
    )


class Composer:
    """Cache planning and control panels until a recorded decision changes."""

    def __init__(self, metadata, decisions):
        self.fonts = load_fonts()
        self.metadata = metadata
        questions = metadata.get("questions", {})
        self.questions = questions if isinstance(questions, dict) else {}
        self.row_questions = {}
        actual = list(dict.fromkeys(key for row in decisions for key in row["answers"]))
        keys = actual or list(self.questions) or list(HEAD_ORDER)
        self.keys = [key for key in HEAD_ORDER if key in keys]
        self.keys += [key for key in keys if key not in self.keys]
        if len(self.keys) > 8:
            raise ValueError("The planning layout supports at most eight heads")
        self.planning = [key for key in ("goal", "target") if key in self.keys]
        self.controls = [key for key in self.keys if key not in self.planning]
        if len(self.controls) > 6:
            raise ValueError("Expected goal/target planning heads and at most six control heads")
        self.base = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(self.base)
        for y in range(HEIGHT):
            draw.line((0, y, WIDTH, y), fill=(9, 17 + y * 6 // HEIGHT, 27 + y * 9 // HEIGHT))
        draw.rectangle((40, 38, 46, 85), fill=MINT)
        self.text(draw, (66, 35), "SYSTEM ONE / DOOM", "title")
        draw.rounded_rectangle((1600, 36, 1880, 88), 12, fill="#173239", outline="#2c6269")
        self.text(draw, (1640, 49), "Qwen3-8B", "answer", MINT)
        draw.rounded_rectangle((40, 112, 1880, 188), 12, fill="#12232e", outline=STROKE)
        self.text(draw, (58, 130), "STANDING ORDER", "label", CYAN)
        mission = metadata.get("standing_order") or "Awaiting recorded mission"
        for index, line in enumerate(self.wrap(str(mission), "body", 1600, 2)):
            self.text(draw, (252, 124 + index * 27), line, "body")
        self.text(draw, (40, 204), "01 / LEVEL", "label", MUTED)
        self.text(draw, (1080, 204), "02 / PLANNING", "label", MUTED)
        self.text(draw, (1880, 204), "SELECTED ANSWER", "label", MINT, "ra")
        x0, y0, x1, y1 = GAME_BOX
        draw.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), fill=STROKE)
        self.dashboard = None

    def fit(self, text, font, width):
        text = str(text).replace("\n", " ")
        if self.fonts[font].getlength(text) <= width:
            return text
        while text and self.fonts[font].getlength(text + "…") > width:
            text = text[:-1]
        return text.rstrip() + "…"

    def wrap(self, text, font, width, lines):
        words = text.split()
        output = []
        while words and len(output) < lines:
            line = words.pop(0)
            if len(output) == lines - 1:
                output.append(self.fit(" ".join([line] + words), font, width))
                break
            while words and self.fonts[font].getlength(line + " " + words[0]) <= width:
                line += " " + words.pop(0)
            output.append(self.fit(line, font, width))
        return output

    def text(self, draw, xy, text, font="body", fill=WHITE, anchor=None):
        draw.text(xy, str(text), font=self.fonts[font], fill=fill, anchor=anchor)

    def description(self, key, choice):
        question = self.row_questions.get(key, self.questions.get(key, ""))
        if isinstance(question, dict):
            criteria = question.get("criteria")
            if isinstance(criteria, dict):
                question = criteria.get(choice) or question.get("instructions") or ""
            else:
                question = criteria or question.get("instructions") or ""
        if isinstance(question, list):
            question = " / ".join(map(str, question))
        return str(question)

    def card(self, draw, key, answer, box, prominent):
        x, y, w, h = box
        draw.rounded_rectangle((x, y, x + w, y + h), 14, fill=PANEL, outline=STROKE)
        draw.rounded_rectangle((x, y + 14, x + 3, y + h - 14), 2, fill=CYAN)
        label = "MOVEMENT" if key == "move" else key.upper().replace("_", " ")
        self.text(draw, (x + 16, y + 12), label, "label", CYAN)
        chosen = answer["choice"] if answer else "Deciding..."
        if prominent:
            self.text(draw, (x + 16, y + 38), self.fit(self.description(key, chosen), "tiny", w - 32),
                      "tiny", MUTED)
            self.text(draw, (x + 16, y + 61), self.fit(chosen, "answer", w - 32),
                      "answer", MINT if answer else MUTED)
        else:
            self.text(draw, (x + w - 16, y + 10),
                      self.fit(chosen, "compact", w - 155),
                      "compact", MINT if answer else MUTED, "ra")
        if not answer:
            return
        options = list(answer["probabilities"].items())
        columns = 1 if prominent else 2
        rows = math.ceil(len(options) / columns)
        top = y + (98 if prominent else 42)
        pitch = (h - (110 if prominent else 50)) / rows
        column_width = (w - 32 - (columns - 1) * 14) / columns
        font = "small" if prominent and pitch >= 26 else "compact"
        if pitch < 22:
            font = "tiny"
        for index, (option, probability) in enumerate(options):
            column, row = (0, index) if prominent else (index % columns, index // columns)
            left = x + 16 + column * (column_width + 14)
            line_y = top + row * pitch
            selected = option == answer["choice"]
            color = MINT if selected else MUTED
            self.text(draw, (left, line_y), self.fit(option, font, column_width - 45),
                      font, color, "lt")
            self.text(draw, (left + column_width, line_y), f"{probability * 100:.0f}%",
                      font, color, "rt")
            bottom = line_y + min(pitch - 3, self.fonts[font].size + 3)
            draw.rectangle((left, bottom, left + column_width, bottom + 2), fill="#293c4a")
            if probability > 0:
                draw.rectangle((left, bottom, left + column_width * probability, bottom + 2),
                               fill=MINT if selected else "#407888")

    def update(self, decision):
        canvas = self.base.copy()
        draw = ImageDraw.Draw(canvas)
        status = ("Waiting for first decision" if decision is None else
                  f"BATCH {decision['batch']}  /  {decision['latency_ms']:,.0f} ms")
        self.text(draw, (1550, 64), status, "small", MUTED, "rm")
        answers = decision["answers"] if decision else {}
        self.row_questions = decision.get("questions", {}) if decision else {}
        for index, key in enumerate(self.planning):
            self.card(draw, key, answers.get(key), (1080 + index * 408, 232, 392, 304), True)
        self.text(draw, (1080, 540), "CONTROL CONTEXT / COMMITTED PLAN", "label", CYAN)
        goal = decision.get("active_goal") if decision else None
        target = decision.get("active_target") if decision else None
        if isinstance(target, dict):
            target = target["name"]
        context = f"Goal: {goal if goal is not None else '—'}   /   Target: {target if target is not None else '—'}"
        for index, line in enumerate(self.wrap(context, "compact", 800, 2)):
            self.text(draw, (1080, 565 + index * 21), line, "compact", WHITE)
        for index, key in enumerate(self.controls):
            self.card(draw, key, answers.get(key),
                      (1080 + index % 2 * 408, 616 + index // 2 * 148, 392, 144), False)
        if decision:
            episode = decision.get("episode")
            if episode is not None:
                self.text(draw, (300, 204), f"EPISODE {episode}", "label", CYAN)
            stats = decision.get("stats", {})
            values = [
                f"{key.upper()} {decision.get(key, stats.get(key)):g}"
                for key in ("health", "ammo", "kills")
                if decision.get(key, stats.get(key)) is not None
            ]
            self.text(draw, (40, 1004), "    /    ".join(values), "body")
        level = (self.metadata.get("map") or self.metadata.get("level")
                 or self.metadata.get("gamelevel") or "")
        self.text(draw, (1040, 204), self.fit(level, "label", 350), "label", MUTED, "ra")
        observation = decision.get("observation", {}) if decision else {}
        inventory = [
            f"{label} {observation[key]:g}"
            for key, label in (("shells", "SHELLS"), ("bullets", "BULLETS"),
                               ("armor", "ARMOR"), ("visited_cells", "CELLS VISITED"))
            if key in observation
        ]
        self.text(draw, (40, 1045), "  /  ".join(inventory) or "ViZDoom / Freedoom",
                  "compact", MUTED)
        self.dashboard = canvas

    def frame(self, game, index, count, source_fps):
        canvas = self.dashboard.copy()
        x0, y0, x1, y1 = GAME_BOX
        canvas.paste(game.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), (x0, y0))
        draw = ImageDraw.Draw(canvas)
        tenths = round(index / source_fps * 10)
        stamp = f"{tenths // 600:02d}:{(tenths % 600) / 10:04.1f}"
        self.text(draw, (1040, 1045), f"{stamp}  /  FRAME {index:05d}", "small", MUTED, "ra")
        return canvas


def find_ffmpeg():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    homebrew = Path("/opt/homebrew/bin/ffmpeg")
    if homebrew.is_file():
        return str(homebrew)
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg is missing; install ffmpeg or imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def render(root, output, fps=None, limit=None):
    source_fps, frames, decisions, metadata = read_capture(root, limit)
    output_fps = source_fps if fps is None else finite_number(fps, "--fps", 0.001)
    output_count = math.ceil(len(frames) * output_fps / source_fps - 1e-9)
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must have an .mp4 extension")
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    ffmpeg = find_ffmpeg()
    composer = Composer(metadata, decisions)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a distinct sibling and publish only after a successful encoder exit.
    partial = output.with_name(f".{output.stem}.{os.getpid()}.rendering.mp4")
    with partial.open("xb"):
        pass
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(output_fps), "-i", "pipe:0",
    ]
    audio = root / "audio.wav"
    if audio.is_file():
        command += [
            "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-c:a", "aac", "-b:a", "192k",
            "-af", f"apad,atrim=duration={output_count / output_fps:.9f}",
        ]
    else:
        command += ["-an"]
    command += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
    ]
    process = None
    stderr_lines = deque(maxlen=80)
    reader = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        def drain_errors():
            for line in iter(process.stderr.readline, b""):
                stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())

        reader = threading.Thread(target=drain_errors, daemon=True)
        reader.start()
        decision_index = -1
        previous_frame_index = -1
        game = None
        composer.update(None)
        for output_index in range(output_count):
            frame_index = min(
                int(output_index * source_fps / output_fps + 1e-9), len(frames) - 1)
            updated = False
            while (decision_index + 1 < len(decisions)
                   and decisions[decision_index + 1]["frame"] <= frame_index):
                decision_index += 1
                updated = True
            if updated:
                composer.update(decisions[decision_index])
            if frame_index != previous_frame_index:
                with Image.open(frames[frame_index]) as source:
                    if source.size != (640, 480):
                        raise ValueError(f"{frames[frame_index]} is not 640x480")
                    game = source.convert("RGB")
                previous_frame_index = frame_index
            image = composer.frame(game, frame_index, len(frames), source_fps)
            process.stdin.write(image.tobytes())
            if output_index and output_index % max(1, int(output_fps * 10)) == 0:
                print(f"Rendered {output_index}/{output_count} frames", file=sys.stderr)
        process.stdin.close()
        code = process.wait()
        reader.join()
        if code:
            raise RuntimeError(f"ffmpeg exited {code}:\n" + "\n".join(stderr_lines))
        if not partial.stat().st_size:
            raise RuntimeError("ffmpeg returned success but produced an empty file")
        # A hard link avoids clobbering an output created concurrently.
        os.link(partial, output)
        print(f"Rendered {output_count} frames, {output_count / output_fps:.2f}s, "
              f"{WIDTH}x{HEIGHT} at {output_fps:g} fps: {output}")
    except BrokenPipeError as exc:
        process.wait()
        reader.join()
        raise RuntimeError(
            f"ffmpeg closed its input (exit {process.returncode}):\n"
            + "\n".join(stderr_lines)
        ) from exc
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if reader:
                reader.join()
            for stream in (process.stdin, process.stderr):
                if stream and not stream.closed:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        pass
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Capture directory")
    parser.add_argument("--output", required=True, type=Path, help="New output .mp4 path")
    parser.add_argument("--fps", type=float, help="Output fps; default: metadata fps")
    parser.add_argument("--limit-frames", type=int, help="Maximum number of source frames")
    args = parser.parse_args()
    if args.limit_frames is not None and args.limit_frames < 1:
        parser.error("--limit-frames must be positive")
    try:
        render(args.input, args.output, args.fps, args.limit_frames)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Render error: {exc}\n")


if __name__ == "__main__":
    main()
