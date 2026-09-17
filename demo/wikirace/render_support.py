"""Offline article layout and paused-clock frame composition."""
from bisect import bisect_right
from functools import lru_cache
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
BG, PANEL, BORDER = "#0b1220", "#131f31", "#2b3c52"
WHITE, MUTED, MINT, BLUE = "#f3f6fb", "#a7b7ce", "#83edc4", "#245cac"


@lru_cache(maxsize=40)
def font(size, style="regular"):
    choices = {
        "regular": ["/System/Library/Fonts/Supplemental/Arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "serif": ["/System/Library/Fonts/Supplemental/Georgia.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
        "mono": ["/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    }
    for name in choices[style]:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    raise RuntimeError(f"No local {style} font found; install DejaVu fonts")


def text(draw, xy, value, size=24, color=WHITE, style="regular"):
    draw.text(xy, str(value), font=font(size, style), fill=color)


def fit(value, width, size=24, style="regular"):
    value = str(value)
    if font(size, style).getlength(value) <= width:
        return value
    while value and font(size, style).getlength(value + "…") > width:
        value = value[:-1]
    return value + "…"


def lines(value, width, size, style="regular"):
    result, row = [], ""
    for word in str(value).split():
        trial = (row + " " + word).strip()
        if row and font(size, style).getlength(trial) > width:
            result.append(row)
            row = word
        else:
            row = trial
    return result + ([row] if row else [])


def clock(seconds):
    ticks = int(max(0, seconds) * 10 + 1e-7)
    return f"{ticks // 600:02d}:{ticks // 10 % 60:02d}.{ticks % 10}"


def title(value):
    if isinstance(value, dict):
        return str(value.get("title", value.get("label", value.get("name", ""))))
    return str(value or "")


class Paragraphs(HTMLParser):
    """Keep only saved paragraph text and actual article-anchor styling."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row = [], None
        self.link, self.skip = False, 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "sup"):
            self.skip += 1
        if tag == "p" and self.row is None:
            self.row = []
        if tag == "a":
            self.link = attrs.get("href", "").startswith(("/wiki/", "./", "https://en.wikipedia.org/wiki/"))

    def handle_endtag(self, tag):
        if tag in ("script", "style", "sup"):
            self.skip = max(0, self.skip - 1)
        if tag == "a":
            self.link = False
        if tag == "p" and self.row is not None:
            if len("".join(t for t, _ in self.row).strip()) > 60:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.row is not None and not self.skip:
            self.row.append((data, self.link))


def read_trace(root):
    trace = json.loads((root / "trace.json").read_text())
    meta, events = trace["metadata"], trace["events"]
    elapsed = meta["elapsed_seconds"]
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Capture must have a positive finite elapsed_seconds")
    if not events or any(not isinstance(e.get("t"), (int, float)) or
                         not math.isfinite(e["t"]) or e["t"] < 0 for e in events):
        raise ValueError("Events require finite, nonnegative elapsed t values")
    if any(a["t"] > b["t"] for a, b in zip(events, events[1:])):
        raise ValueError("Events must be chronological")
    if events[-1]["phase"] != "finished" or abs(events[-1]["t"] - elapsed) > 0.1:
        raise ValueError("Finished event must agree with actual total elapsed time")
    if not isinstance(meta.get("success"), bool):
        raise ValueError("Capture must explicitly record success or failure")
    if meta.get("model") not in ("Qwen/Qwen3-8B", "Qwen3-8B"):
        raise ValueError("This renderer requires Qwen3-8B")
    if not 0 <= meta["model_seconds"] <= elapsed + 0.1:
        raise ValueError("Recorded model time must be within total elapsed time")
    route = meta["route"]
    if route and title(route[0]) != meta["start"]:
        raise ValueError("Recorded route must begin at start")
    if meta["success"] and (not route or title(route[-1]) != meta["target"]):
        raise ValueError("Successful route must actually reach target")
    return trace


class PausedRenderer:
    def __init__(self, root):
        self.root = root
        self.trace = read_trace(root)
        self.meta = self.trace["metadata"]
        self.events = self.trace["events"]
        self.times = [e["t"] for e in self.events]
        self.pages = {p["title"]: p for p in self.trace["pages"]}
        self.pages.update({str(p["id"]): p for p in self.trace["pages"]})
        for key in ("active_elapsed_seconds", "loading_seconds"):
            value = self.meta.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Capture requires finite nonnegative {key}")
        if abs(self.meta["active_elapsed_seconds"] + self.meta["loading_seconds"]
               - self.meta["elapsed_seconds"]) > .02:
            raise ValueError("Active time plus loading time must equal wall time")
        self.states = []
        state = {"route": [self.meta["start"]], "current_page": self.meta["start"],
                 "hop": 0, "model_seconds": 0, "phase": "fetching"}
        previous_active = 0
        for event in self.events:
            active = event.get("active_t")
            if not isinstance(active, (int, float)) or not math.isfinite(active) or active < 0:
                raise ValueError("Each event requires nonnegative active_t")
            if active + .002 < previous_active or active > event["t"] + .002:
                raise ValueError("Active clock must be monotonic and never exceed wall time")
            if not isinstance(event.get("paused"), bool):
                raise ValueError("Each event requires an explicit paused flag")
            previous_active = active
            if event["phase"] == "page_loaded":
                for key in ("selected_link", "eligible_count"):
                    state.pop(key, None)
            if event["phase"] == "selecting":
                state.pop("selected_link", None)
            state.update(event)
            self.states.append(dict(state))
        self.cached_index, self.cached_image = None, None

    @lru_cache(maxsize=40)
    def article(self, page_name):
        page = self.pages.get(page_name, {})
        paragraphs = []
        if page.get("html_path"):
            path = (self.root / page["html_path"]).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("Article snapshots must be inside the capture directory")
            if path.is_file():
                parser = Paragraphs()
                parser.feed(path.read_text(errors="replace"))
                paragraphs = parser.rows
        if not paragraphs:
            paragraphs = [[(p, False)] for p in page.get("paragraphs", [page.get("leadtext", "")])]
        return page, paragraphs

    def draw_article(self, draw, state):
        x, y, width = 78, 278, 1080
        name = title(state["current_page"])
        page, paragraphs = self.article(str(state.get("page_id", name)))
        text(draw, (x, y), "WIKIPEDIA  /  SAVED ARTICLE PREVIEW", 18, "#66768a", "bold")
        y += 42
        for row in lines(page.get("title", name), width, 43, "serif")[:2]:
            text(draw, (x, y), row, 43, "#142337", "serif")
            y += 54
        draw.line((x, y + 8, x + width, y + 8), fill="#cbd4df", width=2)
        y += 31
        text(draw, (x, y), "From Wikipedia, the free encyclopedia", 19, "#69798c")
        y += 46
        if state["phase"] == "fetching" and not state.get("page_id"):
            text(draw, (x, y), "Fetching this article…", 29, "#49617c")
            text(draw, (x, y + 48), "The race clock is paused until page loading completes.", 22, "#69798c")
            return
        if not any(any(t.strip() for t, _ in p) for p in paragraphs):
            text(draw, (x, y), "No saved article excerpt available.", 27, "#69798c")
            return
        for paragraph in paragraphs:
            cursor = x
            tokens = [(m.group(), linked) for value, linked in paragraph
                      for m in re.finditer(r"\S+|\s+", value)]
            for value, linked in tokens:
                value = " " if value.isspace() else value
                if cursor == x and value == " ":
                    continue
                advance = font(26, "serif").getlength(value)
                if cursor + advance > x + width:
                    cursor, y = x, y + 39
                    if value == " ":
                        continue
                if y + 34 > 934:
                    text(draw, (x, 958), "Excerpt continues on Wikipedia", 19, BLUE)
                    return
                text(draw, (cursor, y), value, 26, BLUE if linked else "#283648", "serif")
                cursor += advance
            y += 63

    def draw_route(self, draw, state):
        x, y = 1256, 798
        draw.line((x, 774, 1834, 774), fill=BORDER, width=2)
        route = state.get("route", [self.meta["start"]])
        text(draw, (x, y), f"PATH SO FAR   /   {state.get('hop', max(0, len(route) - 1))} HOPS",
             17, MUTED, "bold")
        y += 34
        for index, name in list(enumerate(route))[-4:]:
            text(draw, (x, y), f"{index:02d}", 18, MUTED, "mono")
            text(draw, (x + 43, y), fit(title(name), 522, 22), 22,
                 MINT if index == len(route) - 1 else WHITE)
            y += 32

    def base(self, index):
        state = self.states[index] if index >= 0 else {
            "phase": "fetching", "current_page": self.meta["start"],
            "route": [self.meta["start"]], "paused": True}
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((40, 35, 225, 73), 9, fill="#21473e")
        text(draw, (57, 44), "SYSTEM ONE", 18, MINT, "bold")
        text(draw, (42, 100), f"{self.meta['start']}  →  {self.meta['target']}", 54, WHITE, "bold")
        text(draw, (1280, 42), "RACE TIME · PAGE LOADS EXCLUDED", 18, MINT, "bold")
        draw.rounded_rectangle((40, 244, 1198, 1005), 16, fill="#f8fafc")
        draw.rounded_rectangle((1220, 244, 1880, 1005), 16, fill=PANEL, outline=BORDER, width=2)
        self.draw_article(draw, state)
        self.draw_choice(draw, state)
        self.draw_route(draw, state)
        page = self.pages.get(str(state.get("page_id", title(state["current_page"]))), {})
        source = page.get("url", "")
        text(draw, (43, 1021), fit(f"Source: {source}" if source else "Source: Wikipedia · request pending",
                                 1160, 17), 17, MUTED)
        text(draw, (43, 1048), "Wikipedia contributors · CC BY-SA 4.0 · creativecommons.org/licenses/by-sa/4.0/",
             17, MUTED)
        text(draw, (1300, 1026), "Saved article preview · not a browser recording", 18, MUTED)
        return image

    def frame(self, t):
        elapsed = self.meta["elapsed_seconds"]
        t = min(t, elapsed)
        index = bisect_right(self.times, t) - 1
        if self.cached_image is None or self.cached_index != index:
            self.cached_image = self.base(index)
            self.cached_index = index
        image = self.cached_image.copy()
        draw = ImageDraw.Draw(image)
        active, paused = self.active_clock(t)
        text(draw, (1384, 79), clock(active), 72, WHITE, "mono")
        state = self.states[index] if index >= 0 else {}
        if paused:
            draw.rounded_rectangle((1280, 175, 1430, 210), 7, fill="#554624")
            text(draw, (1298, 183), "PAUSED", 19, "#ffdb8e", "bold")
            text(draw, (1447, 183), "Page loading", 22, MUTED)
        else:
            model = self.meta["model_seconds"] if t >= elapsed else state.get("model_seconds", 0)
            text(draw, (1280, 183), f"Model time: {model:.2f}s", 24, MUTED)
        return self.finish(image) if t >= elapsed else image
