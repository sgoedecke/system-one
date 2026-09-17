"""Deterministic scheduling and prompt tests without ViZDoom, weights or a GPU."""
import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import MagicMock, patch

import numpy as np

from demo.labels import LabelSystemOne
from demo.doom import render
from demo.tests.test_labels import Model, Tokenizer

with patch.dict(sys.modules, {"vizdoom": MagicMock()}):
    capture = importlib.import_module("demo.doom.capture")


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def observation(goal="Reach exit", target=None):
    exit_target = {"id": "exit", "name": "Campaign exit", "kind": "Exit",
                   "x": 300, "y": 0, "distance": 300}
    armor = {"id": "armor", "name": "Green armor", "kind": "Armor",
             "x": 100, "y": 0, "distance": 100}
    pools = {name: capture.targets_for_goal(name, [], [armor, exit_target], [])
             for name in capture.QUESTIONS["goal"].criteria}
    return {
        "enemies": [], "selected_weapon": "shotgun", "selected_weapon_ammo": 2,
        "own_shotgun": True, "health": 100, "shells": 2, "bullets": 30,
        "armor": 0, "nearest_armor_distance": 100,
        "clearance": {"left": 80, "right": 80, "backward": 80, "forward": 80},
        "aim_bearing": 0, "aim_direction": "centered", "route_bearing": 0,
        "target_distance": 300, "stuck": False, "interactions": [],
        "active_goal": goal, "active_target": target or exit_target,
        "targets": pools[goal], "target_pools": pools, "visited_cells": 1,
    }


class Engine:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fake"))

    def system_one(self, text, questions, cache_prefix=False):
        self.calls.append((text, questions, cache_prefix))
        choices = {"goal": "Add armor", "target": "Green armor", "dodge": "Carry on",
                   "move": "Hold", "strafe": "Hold", "turn": "Hold", "fire": "Hold fire",
                   "weapon": "Shotgun", "use": "Wait"}
        self.clock.advance(.4 if "goal" in questions else .6 if "target" in questions else .2)
        return SimpleNamespace(answers={
            key: SimpleNamespace(choice=choices[key],
                                 probabilities={name: float(name == choices[key])
                                                for name in question.criteria})
            for key, question in questions.items()})


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.engine = Engine(self.clock)
        self.cadence = capture.PlanCadence(3)

    def run_request(self, obs=None, episode=None):
        request = {"kind": self.cadence.next_kind, "episode": episode or self.cadence.episode,
                   "plan_id": self.cadence.plan_id, "frame": 11,
                   "observation": obs or observation()}
        return capture.infer_request(self.engine, request, True, self.clock)

    def test_initial_plan_three_applied_controls_then_replan(self):
        result = self.run_request()
        self.assertEqual(self.cadence.next_kind, "plan")
        self.cadence.commit(result, self.clock(), 1)
        self.assertEqual(self.cadence.controls_since_plan, 0)
        obs = observation(result["active_goal"], result["active_target"])
        for index in range(3):
            result = self.run_request(obs)
            self.clock.advance(100)  # Neither elapsed time nor 35 Hz ticks count.
            self.assertEqual(self.cadence.controls_since_plan, index)
            self.cadence.commit(result, self.clock(), index + 2)
            self.assertEqual(self.cadence.controls_since_plan, index + 1)
        self.assertEqual(self.cadence.next_kind, "plan")
        result = self.run_request(obs)
        self.cadence.commit(result, self.clock(), 5)
        self.assertEqual((self.cadence.plan_id, self.cadence.controls_since_plan), (2, 0))
        self.assertEqual([tuple(call[1]) for call in self.engine.calls],
                         [("goal",), ("target",)] + [capture.CONTROL_HEADS] * 3 +
                         [("goal",), ("target",)])
        self.assertEqual(len(self.cadence.answers), 9)

    def test_strafe_combines_with_movement_and_turning_but_dodge_has_priority(self):
        choices = {"move": "Forward", "turn": "Left", "fire": "Hold fire",
                   "weapon": "Shotgun", "use": "Wait"}
        for dodge in capture.QUESTIONS["dodge"].criteria:
            for strafe in capture.QUESTIONS["strafe"].criteria:
                with self.subTest(dodge=dodge, strafe=strafe):
                    answers = {key: {"choice": value} for key, value in
                               dict(choices, dodge=dodge, strafe=strafe).items()}
                    action = capture.controls(answers)
                    expected = {"Dodge left": (1, 0), "Dodge right": (0, 1),
                                "Dodge back": (0, 0)}.get(
                                    dodge, {"Hold": (0, 0), "Strafe left": (1, 0),
                                            "Strafe right": (0, 1)}[strafe])
                    self.assertEqual(tuple(action[2:4]), expected)
                    self.assertEqual(action[0], 14)
                    self.assertEqual(action[4], -.7)
                    self.assertEqual(action[1], int(dodge == "Dodge back"))

    def test_strafe_prompt_exposes_stuck_without_changing_other_controls(self):
        obs = observation()
        obs.update(stuck=True)
        obs["clearance"].update(left=90, right=20)
        question = capture.questions_for(obs)["strafe"]
        self.assertIn("RECOVERY STATUS: JAMMED", question.instructions)
        self.assertIn("Left CLEAR; right BLOCKED", question.instructions)
        self.assertEqual(list(question.criteria), ["Hold", "Strafe left", "Strafe right"])

    def test_clear_sides_and_blocked_forward_do_not_imply_strafe_when_not_stuck(self):
        for forward in (20, 90):
            for left in (70, 71):
                for right in (70, 71):
                    obs = observation()
                    obs["clearance"].update(forward=forward, left=left, right=right)
                    question = capture.questions_for(obs)["strafe"]
                    self.assertIn("RECOVERY STATUS: NORMAL", question.instructions)
                    self.assertIn(f"Left {'CLEAR' if left > 70 else 'BLOCKED'}", question.instructions)
                    self.assertIn(f"right {'CLEAR' if right > 70 else 'BLOCKED'}", question.instructions)
                    self.assertIn("choose Hold regardless of side clearance", question.instructions)
                    self.assertEqual(len(question.criteria), 3)

    def test_target_is_model_selected_conditioned_on_new_goal(self):
        result = self.run_request()
        self.assertEqual(result["kind"], "plan")
        self.assertEqual(result["active_goal"], "Add armor")
        self.assertEqual(result["active_target"]["id"], "armor")
        self.assertIn("Stuck: False", self.engine.calls[0][0])
        self.assertIn("final destination distance 300", self.engine.calls[0][0])
        text, questions, cache = self.engine.calls[1]
        self.assertIn(capture.STANDING_ORDER, text)
        self.assertIn("Equipped shotgun", text)
        self.assertIn("NEW SELECTED PLANNING GOAL: Add armor", text)
        self.assertNotIn("Reach exit", text)
        self.assertEqual(list(questions["target"].criteria), ["Green armor"])
        self.assertIn("Selected planning goal: Add armor", questions["target"].instructions)
        self.assertTrue(cache)
        self.assertAlmostEqual(result["goal_latency_ms"], 400)
        self.assertAlmostEqual(result["target_latency_ms"], 600)
        self.assertAlmostEqual(result["planning_latency_ms"], 1000)
        self.assertEqual(result["observation_frame"], 11)
        self.assertEqual(observation()["active_goal"], "Reach exit")

    def test_warmups_unapplied_results_and_episode_discard_do_not_count(self):
        self.run_request()  # Completed warmup: never committed.
        self.assertEqual((self.cadence.plan_id, self.cadence.controls_since_plan), (0, 0))
        plan = self.run_request()
        self.cadence.commit(plan, self.clock(), 1)
        pending = self.run_request()
        self.assertEqual(self.cadence.controls_since_plan, 0)
        self.cadence.reset(2)
        self.assertFalse(self.cadence.commit(pending, self.clock(), 2))
        self.assertFalse(self.cadence.commit(plan, self.clock(), 2))
        self.assertEqual(self.cadence.next_kind, "plan")
        self.assertEqual(self.cadence.answers, {})
        self.assertIsNone(self.cadence.last_control_applied)
        new_plan = self.run_request()
        self.cadence.commit(new_plan, self.clock(), 3)
        self.assertEqual((self.cadence.plan_id, self.cadence.controls_since_plan), (2, 0))

    def test_both_planning_stages_and_all_controls_support_labels(self):
        self.engine.label_map = LabelSystemOne(Model(), Tokenizer()).label_map
        plan = self.run_request()
        self.cadence.commit(plan, self.clock(), 1)
        self.run_request(observation(plan["active_goal"], plan["active_target"]))
        for _, questions, _ in self.engine.calls:
            for question in questions.values():
                self.assertIn("two-letter option label", question.instructions)
                self.assertNotIn("index digit", question.instructions)
                self.assertNotIn("choice_index:", question.instructions)

    def test_standing_order_reaches_planning_and_control_calls(self):
        plan = self.run_request()
        self.cadence.commit(plan, self.clock(), 1)
        self.run_request(observation(plan["active_goal"], plan["active_target"]))
        for text, _, _ in self.engine.calls:
            self.assertIn("kill all enemies in the way", text)
            self.assertIn(capture.STANDING_ORDER, text)

    def test_tool_backend_uses_choose_and_preserves_unscored_answers(self):
        obs = observation("Add armor")
        response = self.engine.system_one("", capture.questions_for(obs))
        for answer in response.answers.values():
            answer.probabilities = None
        tool = SimpleNamespace(inference_mode="tool_agent",
                               choose=MagicMock(return_value=response))
        result = capture.infer(tool, obs, False, tuple(capture.QUESTIONS), self.clock)
        tool.choose.assert_called_once()
        self.assertTrue(all(answer["probabilities"] is None for answer in result["answers"].values()))

    def test_tool_backend_rejects_single_token_options(self):
        for option in ("--labels", "--cache-prefix"):
            with patch.object(sys, "argv", ["capture", "--output", "unused",
                                           "--controller", "tool-agent", option]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    capture.main()
                self.assertEqual(caught.exception.code, 2)

    def test_invalid_answers_and_model_errors_propagate(self):
        for head in ("goal", "target"):
            original = self.engine.system_one

            def invalid(text, questions, cache_prefix=False):
                result = original(text, questions, cache_prefix)
                if head in result.answers:
                    result.answers[head].choice = "Invented"
                return result

            with patch.object(self.engine, "system_one", side_effect=invalid):
                with self.assertRaisesRegex(ValueError, f"Invalid {head} choice"):
                    self.run_request()
        with patch.object(self.engine, "system_one", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.run_request()

    def test_plan_every_must_be_positive_integer(self):
        for value in ("0", "-1", "1.5", "no"):
            with self.assertRaises(argparse.ArgumentTypeError):
                capture.positive_integer(value)
        self.assertEqual(capture.positive_integer("3"), 3)
        with patch.object(sys, "argv", ["capture", "--output", "unused", "--plan-every", "0"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                capture.main()
        self.assertEqual(caught.exception.code, 2)

    def capture_fixture(self, reset_at=None, pending_at=None, cache_prefix=False, tool_agent=False):
        root = Path(__file__).parent / f".doom-fixture-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        (root / "freedoom2.wad").write_bytes(b"fake wad")
        game = MagicMock()
        game.is_episode_finished.return_value = False
        state = {"ticks": 0, "dead": False}
        game.is_player_dead.side_effect = lambda: state["dead"]
        def action(*args):
            state["ticks"] += 1
            state["dead"] = state["ticks"] == reset_at
        game.make_action.side_effect = action
        game.get_game_variable.return_value = 0
        game.get_state.return_value = SimpleNamespace(
            screen_buffer=np.zeros((480, 640, 3), dtype=np.uint8),
            audio_buffer=np.zeros((1260, 2), dtype=np.int16))
        nav = SimpleNamespace(items=[observation()["active_target"]], valid=[0],
                              visited={0}, collected=set())
        inv = {"health": 100, "shells": 2, "bullets": 30, "selected_weapon_ammo": 2}
        original = capture.infer_request
        if tool_agent:
            self.engine.inference_mode = "tool_agent"
            self.engine.trace = []
            self.engine.reset = MagicMock()
            turns = []

            def choose(text, questions):
                result = self.engine.system_one(text, questions)
                if tuple(questions) == capture.CONTROL_HEADS:
                    head = "move" if len(turns) % 2 == 0 else "strafe"
                    turns.append(head)
                    result.answers = {head: result.answers[head]}
                    result.answers[head].choice = "Forward" if head == "move" else "Strafe left"
                for answer in result.answers.values():
                    answer.probabilities = None
                return result

            self.engine.choose = choose

        def inference(engine, request, cache):
            return original(engine, request, cache, self.clock)

        def submit(function, *args):
            result = function(*args)
            return SimpleNamespace(done=lambda: args[1]["frame"] != pending_at, result=lambda: result)

        def start(*args):
            state["dead"] = False
            return inv

        worker = MagicMock()
        worker.__enter__.return_value.submit.side_effect = submit
        with patch.object(sys, "argv", ["capture", "--output", str(root / "capture"),
                                       "--seconds", str(12 / 35), "--device", "cpu"] +
                                      (["--cache-prefix"] if cache_prefix else []) +
                                      (["--controller", "tool-agent"] if tool_agent else [])), \
                patch.object(capture, "create_game", return_value=game), \
                patch.object(capture, "CampaignMap", return_value=nav), \
                patch.object(capture, "start_episode", side_effect=start), \
                patch.object(capture, "inventory", return_value=inv), \
                patch.object(capture, "observe", side_effect=lambda g, s, n, goal, target, memory:
                             observation(goal, target)), \
                patch.object(capture.SystemOne, "from_pretrained", return_value=self.engine), \
                patch("demo.doom.tool_agent.ToolAgent.from_pretrained", return_value=self.engine), \
                patch.object(capture, "ThreadPoolExecutor", return_value=worker), \
                patch.object(capture, "infer_request", side_effect=inference), \
                patch.object(capture.time, "perf_counter", self.clock), \
                patch.object(capture.time, "sleep", self.clock.advance), \
                patch.object(capture.vzd, "__file__", str(root / "vizdoom.py"), create=True), \
                patch.object(capture.vzd, "__version__", "fake", create=True), \
                patch.object(capture, "BUTTONS", [SimpleNamespace(name=str(i)) for i in range(9)]), \
                patch("builtins.print"):
            capture.main()
        rows = [json.loads(line) for line in (root / "capture/decisions.jsonl").read_text().splitlines()]
        metadata = json.loads((root / "capture/metadata.json").read_text())
        events = [json.loads(line) for line in (root / "capture/events.jsonl").read_text().splitlines()]
        _, _, parsed, _ = render.read_capture(root / "capture", None)
        self.assertEqual(parsed, rows)
        game.close.assert_called_once()
        return rows, metadata, events

    def test_capture_loop_warmup_cadence_and_recorded_gaps(self):
        rows, metadata, _ = self.capture_fixture()
        self.assertEqual([row["kind"] for row in rows],
                         ["plan", "control", "control", "control",
                          "plan", "control", "control", "control", "plan", "control", "control"])
        self.assertEqual([row["controls_since_plan"] for row in rows], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2])
        self.assertEqual(set(rows[0]["answers"]), set(capture.PLAN_HEADS))
        self.assertEqual(set(rows[1]["answers"]), set(capture.QUESTIONS))
        self.assertEqual(rows[1]["active_goal"], rows[1]["observation"]["active_goal"])
        self.assertEqual(rows[1]["active_target"], rows[1]["observation"]["active_target"])
        self.assertEqual(rows[4]["controls_plan_id"], 1)
        self.assertEqual(rows[4]["plan_id"], 2)
        self.assertFalse(rows[5]["plan_updated"])
        self.assertAlmostEqual(rows[5]["control_gap_ms"], 1200)
        self.assertAlmostEqual(rows[2]["control_gap_ms"], 200)
        self.assertEqual((metadata["plans"], metadata["control_updates"]), (3, 8))
        self.assertAlmostEqual(metadata["warmup_model_ms"], 2400)
        self.assertEqual(metadata["plan_every"], 3)
        self.assertEqual(metadata["forward_passes_per_model_call"], {"goal": 1, "target": 1, "control": 1})
        self.assertEqual(metadata["forward_passes_per_plan"], 2)

    def test_tool_turns_apply_partial_controls_without_waiting_for_other_heads(self):
        rows, metadata, _ = self.capture_fixture(tool_agent=True)
        controls = [row for row in rows if row["kind"] == "control"]
        self.assertEqual(controls[0]["evaluated_heads"], ["move"])
        self.assertEqual(controls[0]["action"][:4], [14, 0, 0, 0])
        self.assertEqual(controls[1]["evaluated_heads"], ["strafe"])
        self.assertEqual(controls[1]["action"][:4], [14, 0, 1, 0])
        self.assertNotIn("fire", controls[1]["answers"])
        self.assertEqual([row["controls_since_plan"] for row in controls[:4]], [1, 2, 3, 1])
        self.assertEqual(metadata["controller"], "tool_agent")
        self.assertIsNone(metadata["forward_passes_per_model_call"])
        self.engine.reset.assert_called_once()

    def test_cached_forward_pass_metadata_distinguishes_single_planning_heads(self):
        _, metadata, _ = self.capture_fixture(cache_prefix=True)
        self.assertEqual(metadata["forward_passes_per_model_call"], {"goal": 1, "target": 1, "control": 2})
        self.assertEqual(metadata["forward_passes_per_plan"], 2)

    def test_capture_loop_discards_old_episode_and_clears_renderer(self):
        rows, metadata, events = self.capture_fixture(reset_at=6, pending_at=4)
        reset = next(row for row in rows if row["kind"] == "reset")
        self.assertEqual(reset["answers"], {})
        self.assertEqual(reset["controls_since_plan"], 0)
        episode_two = [row for row in rows if row["episode"] == 2 and row["kind"] != "reset"]
        self.assertEqual(episode_two[0]["kind"], "plan")
        self.assertEqual(set(episode_two[0]["answers"]), set(capture.PLAN_HEADS))
        self.assertEqual(episode_two[1]["controls_since_plan"], 1)
        self.assertIsNone(episode_two[1]["control_gap_ms"])
        discards = [event for event in events if event["event"] == "discarded_episode_boundary_inference"]
        self.assertEqual(len(discards), 1)
        self.assertEqual(discards[0]["result"]["kind"], "control")
        self.assertEqual(discards[0]["result"]["episode"], 1)
        self.assertAlmostEqual(metadata["unapplied_model_ms"], 200)

    def test_capture_loop_final_pending_result_never_advances_count(self):
        rows, metadata, events = self.capture_fixture(pending_at=4)
        self.assertEqual([row["kind"] for row in rows], ["plan", "control", "control"])
        self.assertEqual(rows[-1]["controls_since_plan"], 2)
        self.assertEqual(metadata["control_updates"], 2)
        self.assertEqual(metadata["plans"], 1)
        self.assertEqual(events[-1]["event"], "unapplied_final_inference")
        self.assertEqual(events[-1]["result"]["kind"], "control")
        self.assertAlmostEqual(metadata["unapplied_model_ms"], 200)


if __name__ == "__main__":
    unittest.main()
