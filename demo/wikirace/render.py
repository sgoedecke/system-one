#!/usr/bin/env python3
"""Render the genuine 100-way tournament at 1x with a load-paused race clock.

Only completed group answers are shown. A bounded top-five view is explicitly
local to one group, never a global ranking of every outgoing article link.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

from .render_support import BG, BORDER, H, MINT, MUTED, PANEL, W, WHITE, fit, lines, text, title, PausedRenderer


class Tournament100Renderer(PausedRenderer):
    def __init__(self, root):
        super().__init__(root)
        if self.meta.get("group_size") != 100:
            raise ValueError("This renderer requires an actual 100-way tournament capture")
        latest, round_record, pools, completed = None, None, [], 0
        for event, state in zip(self.events, self.states):
            if event["phase"] == "page_loaded":
                latest, round_record, pools, completed = None, None, [], 0
            if event["phase"] in ("selecting", "model_batch"):
                hop = (self.trace["hops"][event["hop"]]
                       if event["hop"] < len(self.trace["hops"]) else None)
                pools = [r["candidate_count"] for r in hop["rounds"]] + [1] if hop else []
            if event["phase"] == "model_batch" and hop:
                round_record = next(r for r in hop["rounds"] if r["round"] == event["round"])
                completed = event["batch_start"] + event["questions"]
                latest = round_record["groups"][completed - 1]
                if not 1 <= len(latest["candidates"]) <= 100:
                    raise ValueError("Completed group has an invalid cardinality")
                if latest["winner"] != latest["answer"]["choice"]:
                    raise ValueError("Group winner and recorded answer disagree")
            state.update(_group=latest, _round_record=round_record,
                         _pools=pools, _completed_groups=completed)
        self.intervals = self.meta["loading_intervals"]
        if any(not 0 <= i["start_t"] <= i["end_t"] <= self.meta["elapsed_seconds"]
               for i in self.intervals):
            raise ValueError("Loading intervals must lie inside the capture")
        if any(a["end_t"] > b["start_t"] for a, b in zip(self.intervals, self.intervals[1:])):
            raise ValueError("Loading intervals must not overlap")
        if abs(sum(i["end_t"] - i["start_t"] for i in self.intervals) -
               self.meta["loading_seconds"]) > 1e-6:
            raise ValueError("Loading intervals must match the recorded loading total")

    def active_clock(self, t):
        if t >= self.meta["elapsed_seconds"]:
            return self.meta["active_elapsed_seconds"], False
        loading = sum(max(0, min(t, i["end_t"]) - i["start_t"]) for i in self.intervals)
        paused = any(i["start_t"] <= t < i["end_t"] for i in self.intervals)
        return max(0, t - loading), paused

    def base(self, index):
        image = super().base(index)
        draw = ImageDraw.Draw(image)
        draw.rectangle((242, 35, 1180, 77), fill=BG)
        text(draw, (246, 43), "100-WAY WIKIPEDIA TOURNAMENT", 20, MUTED, "bold")
        draw.rectangle((42, 175, 1174, 225), fill=BG)
        text(draw, (44, 179), "Qwen3-8B  /  100-link groups  /  1× original footage", 24, MUTED)
        return image

    def draw_choice(self, draw, state):
        x, width = 1256, 576
        phase = state["phase"]
        group = state.get("_group")
        heading = {
            "fetching": "LOADING NEXT ARTICLE", "page_loaded": "SAVING ARTICLE",
            "loading_complete": "ARTICLE READY", "selecting": "100-WAY TOURNAMENT",
            "model_batch": "100-WAY TOURNAMENT", "navigating": "FOLLOWING CHOSEN LINK",
            "finished": "RACE COMPLETE",
        }.get(phase, phase.upper())
        text(draw, (x, 279), heading, 18, MINT, "bold")
        count = state.get("eligible_count")
        text(draw, (x, 321), f"{count:,} eligible outgoing links" if isinstance(count, int)
             else "Real Wikipedia links", 25, WHITE, "bold")
        pools = state.get("_pools", [])
        if pools:
            text(draw, (x, 362), "Round pools: " + " → ".join(f"{n:,}" for n in pools), 21, MUTED)
        record = state.get("_round_record")
        if record:
            text(draw, (x, 398), f"Round {record['round'] + 1} / {len(pools) - 1}"
                 f"   ·   {state['_completed_groups']} / {len(record['groups'])} groups completed",
                 19, MUTED)
        if not group:
            y = 444
            message = "Page loading pauses the race clock.\nEvery second of footage is retained." \
                if state.get("paused", True) else \
                "Choosing one winner per group.\nWaiting for the next recorded response."
            for row in message.split("\n"):
                text(draw, (x, y), row, 23, MUTED)
                y += 39
            return
        cardinality = len(group["candidates"])
        draw.rounded_rectangle((x - 12, 439, x + width + 10, 759), 11, fill="#192b40")
        text(draw, (x + 4, 453), f"GROUP RESULT · {cardinality} LINKS", 18, MINT, "bold")
        text(draw, (x + 4, 480), f"Last completed group: {group['group'] + 1} / {len(record['groups'])}",
             16, MUTED)
        winner = group["winner"]
        size = 27
        while len(lines(winner, width - 16, size, "bold")) > 2 and size > 18:
            size -= 1
        for i, row in enumerate(lines(winner, width - 16, size, "bold")[:2]):
            text(draw, (x + 4, 509 + i * 31), row, size, WHITE, "bold")
        text(draw, (x + 4, 576), f"Winning label: {group['selected_label']} · one token", 16, MUTED)
        probabilities = group["answer"]["probabilities"]
        top = sorted(probabilities.items(), key=lambda pair: -pair[1])[:5]
        text(draw, (x + 4, 607), f"TOP {len(top)} OF {cardinality} SHOWN · GROUP-LOCAL PROBABILITIES",
             14, MUTED, "bold")
        for i, (name, probability) in enumerate(top):
            y = 632 + i * 24
            color = MINT if name == winner else MUTED
            text(draw, (x + 4, y), fit(name, width - 110, 19), 19, color)
            text(draw, (x + width - 94, y), f"{probability * 100:6.2f}%", 18, color, "mono")

    def finish(self, image):
        image = Image.alpha_composite(image.convert("RGBA"),
                                      Image.new("RGBA", (W, H), (3, 9, 18, 170))).convert("RGB")
        draw = ImageDraw.Draw(image)
        meta = self.meta
        color = MINT if meta["success"] else "#ffbd9e"
        draw.rounded_rectangle((275, 266, 1645, 913), 24, fill=PANEL, outline=BORDER, width=2)
        text(draw, (328, 309), "TARGET REACHED · 100-WAY TOURNAMENT" if meta["success"]
             else "TARGET NOT REACHED · 100-WAY TOURNAMENT", 23, color, "bold")
        text(draw, (328, 351), f"{meta['start']} → {meta['target']}", 52, WHITE, "bold")
        text(draw, (332, 438), "RACE TIME · PAGE LOADS EXCLUDED", 18, MUTED, "bold")
        text(draw, (328, 464), f"{meta['active_elapsed_seconds']:.2f}s", 90, color, "mono")
        text(draw, (332, 579), f"{meta['active_elapsed_seconds']:.6f} seconds active", 21, MUTED)
        text(draw, (1060, 441), f"{meta['hops']} HOPS", 46, WHITE, "bold")
        text(draw, (1060, 511), f"Full wall time  {meta['elapsed_seconds']:.2f}s", 25, WHITE)
        text(draw, (1060, 552), f"Model time      {meta['model_seconds']:.2f}s", 25, WHITE)
        text(draw, (1060, 593), f"Page loading    {meta['loading_seconds']:.2f}s", 22, MUTED)
        draw.line((328, 646, 1592, 646), fill=BORDER, width=2)
        text(draw, (328, 676), "ACTUAL RECORDED ROUTE", 17, MUTED, "bold")
        route = "  →  ".join(title(p) for p in meta["route"])
        size = 34
        while len(lines(route, 1250, size, "bold")) > 2 and size > 20:
            size -= 1
        for i, row in enumerate(lines(route, 1250, size, "bold")):
            text(draw, (328, 719 + i * 43), row, size, WHITE, "bold")
        text(draw, (328, 861), "1× full footage · race clock pauses on loads · 6s frozen finish hold",
             20, MUTED)
        page = self.pages.get(title(meta["route"][-1]), {}) if meta["route"] else {}
        text(draw, (43, 1021), fit(f"Source: {page.get('url', 'Wikipedia')}", 1160, 17), 17, MUTED)
        text(draw, (43, 1048), "Wikipedia contributors · CC BY-SA 4.0 · creativecommons.org/licenses/by-sa/4.0/",
             17, MUTED)
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Captured tournament directory")
    parser.add_argument("--output", type=Path, required=True, help="New MP4 path, or image path with --still")
    parser.add_argument("--still", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    if args.still is not None and (not math.isfinite(args.still) or args.still < 0):
        parser.error("--still must be finite and nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = Tournament100Renderer(args.input.resolve())
    if args.still is not None:
        renderer.frame(args.still).save(args.output, quality=94)
        return
    fps = 30
    frames = math.ceil((renderer.meta["elapsed_seconds"] + 6) * fps - 1e-9)
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-metadata", "title=System One — Qwen3-8B 100-way Wikipedia tournament",
        "-metadata", "comment=Full original 1x wall timeline; race clock excludes page loading. Group-local probabilities only. Six-second frozen finish hold. Wikipedia contributors, CC BY-SA 4.0.",
        str(args.output),
    ], stdin=subprocess.PIPE)
    try:
        for i in range(frames):
            process.stdin.write(renderer.frame(i / fps).tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    print(json.dumps({"output": str(args.output), "duration": frames / fps,
                      "wall_elapsed": renderer.meta["elapsed_seconds"],
                      "active_elapsed": renderer.meta["active_elapsed_seconds"],
                      "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
