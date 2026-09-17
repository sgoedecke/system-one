"""Renderer-only verification; synthetic fixtures are removed after every run."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import uuid
import wave
from unittest.mock import patch

from PIL import Image
from demo.doom import render


class PlanningRendererTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / f".renderer-fixture-{uuid.uuid4().hex}"
        (self.root / "frames").mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.metadata = {
            "fps": 35, "width": 640, "height": 480, "model": "Qwen/Qwen3-8B",
            "standing_order": "Explore the level, replenish supplies, and find the exit.",
            "questions": {"goal": {"criteria": "Upgrade / scout / fight / resupply / survive / exit"},
                          "target": {"instructions": "Choose a named item or enemy"}},
            "map": "TEST MAP — NOT GAMEPLAY",
        }
        self.row = {
            "frame": 2, "batch": 0, "latency_ms": 215.4, "episode": 1,
            "active_goal": "Scout", "active_target": "North corridor",
            "answers": {}, "health": 100, "ammo": 2, "kills": 0,
        }
        options = {
            "goal": ["Upgrade weapon", "Scout", "Kill enemies", "Stock ammo",
                     "Restore health", "Add armor", "Exit"],
            "target": ["Shotgun", "Shells", "Medikit", "Green armor", "Imp",
                       "Zombieman", "Door", "North corridor", "Exit switch", "None"],
            "dodge": ["Carry on", "Left", "Right", "Back"],
            "move": ["Forward", "Stay", "Backward"],
            "turn": ["Hard left", "Left", "Stay", "Right", "Hard right"],
            "fire": ["Yes", "No"],
            "weapon": ["Keep", "Pistol", "Shotgun"],
            "use": ["Yes", "No"],
        }
        for key, names in options.items():
            self.row["answers"][key] = {
                "choice": names[0],
                "probabilities": {name: .7 if i == 0 else .3 / (len(names) - 1)
                                  for i, name in enumerate(names)},
            }
        for index in range(7):
            Image.new("RGB", (640, 480), (index * 10, 30, 60)).save(
                self.root / "frames" / f"{index:06d}.jpg")
        self.write_capture()

    def write_capture(self):
        (self.root / "metadata.json").write_text(json.dumps(self.metadata))
        (self.root / "decisions.jsonl").write_text(json.dumps(self.row) + "\n")

    def test_dynamic_schema_and_ten_target_choices(self):
        fps, frames, rows, metadata = render.read_capture(self.root, None)
        self.assertEqual((fps, len(frames)), (35, 7))
        self.assertEqual(len(rows[0]["answers"]["target"]["probabilities"]), 10)
        composer = render.Composer(metadata, rows)
        self.assertEqual(composer.planning, ["goal", "target"])
        self.assertEqual(len(composer.controls), 6)
        composer.update(None)
        waiting = composer.dashboard.tobytes()
        composer.update(self.row)
        self.assertNotEqual(waiting, composer.dashboard.tobytes())
        self.assertEqual(composer.dashboard.size, (1920, 1080))
        x0, y0, x1, y1 = render.GAME_BOX
        self.assertEqual((x1-x0)/(y1-y0), 4/3)

    def test_committed_context_distinct_from_new_plan(self):
        composer = render.Composer(self.metadata, [self.row])
        with patch.object(composer, "text", wraps=composer.text) as text:
            composer.update(self.row)
        strings = [call.args[2] for call in text.call_args_list]
        self.assertIn("Upgrade weapon", strings)
        self.assertIn("Goal: Scout / Target: North corridor", strings)

    def test_six_heads_and_ten_control_options(self):
        del self.row["answers"]["fire"]
        del self.row["answers"]["use"]
        names = [f"Direction {index}" for index in range(10)]
        self.row["answers"]["move"] = {
            "choice": names[-1], "probabilities": {name: .1 for name in names}}
        self.write_capture()
        _, _, rows, metadata = render.read_capture(self.root, None)
        composer = render.Composer(metadata, rows)
        self.assertEqual(len(composer.controls), 4)
        composer.update(self.row)
        for name in names:
            self.assertLessEqual(composer.fonts["tiny"].getlength(
                composer.fit(name, "tiny", 128)), 128)

    def test_capture_target_objects_and_dynamic_criteria(self):
        self.row["active_target"] = {"id": "exit-1", "name": "Campaign exit", "x": 12}
        self.row["questions"] = {
            "goal": {"criteria": {"Upgrade weapon": "Acquire an available shotgun."}}}
        self.row["observation"] = {"shells": 0, "bullets": 25, "armor": 80, "visited_cells": 171}
        self.write_capture()
        _, _, rows, metadata = render.read_capture(self.root, None)
        composer = render.Composer(metadata, rows)
        with patch.object(composer, "text", wraps=composer.text) as text:
            composer.update(rows[0])
        strings = [call.args[2] for call in text.call_args_list]
        self.assertIn("Goal: Scout / Target: Campaign exit", strings)
        self.assertIn("Acquire an available shotgun.", strings)
        self.assertIn("SHELLS 0  /  BULLETS 25  /  ARMOR 80  /  CELLS VISITED 171", strings)

    def test_rejects_invalid_choice_and_excess_options(self):
        self.row["answers"]["goal"]["choice"] = "Invented"
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "choice must occur"):
            render.read_capture(self.root, None)
        self.row["answers"]["goal"]["probabilities"] = {str(i): 1/11 for i in range(11)}
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "1–10"):
            render.read_capture(self.root, None)

    def test_plan_commit_visible_before_controls_and_retained_freshness(self):
        self.metadata.update(schema_version=2, plan_every=3)
        plan = dict(self.row, kind="plan", plan_id=1, plan_updated=True,
                    controls_since_plan=0, controls_plan_id=None,
                    evaluated_heads=["goal", "target"],
                    answers={key: self.row["answers"][key] for key in ("goal", "target")})
        self.row = plan
        self.write_capture()
        _, _, rows, metadata = render.read_capture(self.root, None)
        with patch.object(render.Composer, "text", autospec=True) as text:
            render.Composer(metadata, rows)
        self.assertIn("Plan every 3 control updates", [call.args[3] for call in text.call_args_list])
        composer = render.Composer(metadata, rows)
        self.assertEqual(composer.keys, list(render.LEGACY_HEAD_ORDER))
        with patch.object(composer, "text", wraps=composer.text) as text:
            composer.update(plan)
        strings = [call.args[2] for call in text.call_args_list]
        self.assertEqual(strings.count("Deciding..."), 6)
        self.assertEqual(strings.count("NEW PLAN #1"), 2)
        control = dict(plan, kind="control", plan_updated=False, controls_since_plan=1,
                       controls_plan_id=1, evaluated_heads=list(render.HEAD_ORDER[2:]))
        with patch.object(composer, "text", wraps=composer.text) as text:
            composer.update(control)
        strings = [call.args[2] for call in text.call_args_list]
        self.assertEqual(strings.count("RETAINED PLAN #1"), 2)
        self.assertIn("Upgrade weapon", strings)
        self.assertIn("1/3 control updates since plan  /  Controls held from plan #1", strings)
        reset = dict(plan, kind="reset", answers={}, plan_updated=False,
                     evaluated_heads=[], controls_since_plan=0)
        self.row = reset
        self.write_capture()
        render.read_capture(self.root, None)
        with patch.object(composer, "text", wraps=composer.text) as text:
            composer.update(reset)
        self.assertEqual([call.args[2] for call in text.call_args_list].count("Deciding..."), 8)

    def test_rejects_falsely_fresh_plan_metadata(self):
        self.metadata.update(schema_version=2, plan_every=3)
        self.row.update(kind="control", plan_id=1, plan_updated=True, controls_since_plan=1,
                        evaluated_heads=list(render.LEGACY_HEAD_ORDER[2:]))
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "plan_updated"):
            render.read_capture(self.root, None)

    def test_native_tool_calls_have_no_fabricated_probabilities(self):
        self.metadata["controller"] = "tool_agent"
        self.row["questions"] = {
            key: {"criteria": dict.fromkeys(answer["probabilities"], "Tool option")}
            for key, answer in self.row["answers"].items()}
        for answer in self.row["answers"].values():
            answer["probabilities"] = None
        self.write_capture()
        _, _, rows, metadata = render.read_capture(self.root, None)
        with patch.object(render.Composer, "text", autospec=True) as text:
            composer = render.Composer(metadata, rows)
            composer.update(rows[0])
        strings = [call.args[3] for call in text.call_args_list]
        self.assertIn("TOOL-CALL AGENT / DOOM", strings)
        self.assertIn("TOOL CALL / NO PROBABILITY SCORES", strings)
        self.assertFalse(any("%" in str(value) for value in strings))
        with patch.object(composer, "text", wraps=composer.text) as text:
            composer.update(None)
        self.assertIn("Not called", [call.args[2] for call in text.call_args_list])
        self.row["answers"]["goal"]["probabilities"] = {"Upgrade weapon": 1.0}
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "must not claim choice probabilities"):
            render.read_capture(self.root, None)
        self.row["answers"]["goal"].update(choice="Invented", probabilities=None)
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "must occur in its tool options"):
            render.read_capture(self.root, None)

    def test_real_encoder_shape_count_audio_and_cache(self):
        self.metadata.update(schema_version=2, plan_every=3,
                             control_heads=list(render.HEAD_ORDER[2:]))
        self.row.update(kind="control", plan_id=1, plan_updated=False,
                        controls_since_plan=1, controls_plan_id=1,
                        evaluated_heads=list(render.HEAD_ORDER[2:]))
        self.row["answers"]["strafe"] = {
            "choice": "Strafe left",
            "probabilities": {"Hold": .1, "Strafe left": .8, "Strafe right": .1}}
        turns = ["Hard left", "Left", "Fine left", "Hold", "Fine right", "Right", "Hard right"]
        self.row["answers"]["turn"] = {
            "choice": "Hold", "probabilities": {name: float(name == "Hold") for name in turns}}
        self.write_capture()
        _, _, rows, metadata = render.read_capture(self.root, None)
        composer = render.Composer(metadata, rows)
        self.assertEqual(composer.controls, list(render.HEAD_ORDER[2:]))
        with patch.object(composer, "card", wraps=composer.card) as card:
            composer.update(self.row)
        for call in card.call_args_list:
            x, y, width, height = call.args[3]
            self.assertLessEqual(x + width, render.WIDTH)
            self.assertLessEqual(y + height, render.HEIGHT)
        self.assertEqual(len(card.call_args_list), 9)
        with wave.open(str(self.root / "audio.wav"), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(44100)
            audio.writeframes(b"\0" * 441 * 4)
        output = self.root / "fixture.mp4"
        original = render.Composer.update
        updates = []
        def update(composer, row):
            updates.append(None if row is None else row["frame"])
            original(composer, row)
        with patch.object(render.Composer, "update", update), contextlib.redirect_stdout(io.StringIO()):
            render.render(self.root, output)
        self.assertEqual(updates, [None, 2])
        probe = subprocess.run([
            shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe", "-v", "error",
            "-count_frames", "-show_streams", "-of", "json", str(output),
        ], check=True, capture_output=True, text=True)
        video, audio = json.loads(probe.stdout)["streams"]
        self.assertEqual((video["width"], video["height"]), (1920, 1080))
        self.assertEqual((video["codec_name"], video["pix_fmt"]), ("h264", "yuv420p"))
        self.assertEqual(int(video["nb_read_frames"]), 7)
        self.assertEqual((audio["codec_name"], audio["channels"]), ("aac", 2))
        subprocess.run([render.find_ffmpeg(), "-v", "error", "-i", str(output),
                        "-f", "null", "-"], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
