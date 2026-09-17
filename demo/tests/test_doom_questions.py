import importlib
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

from demo.labels import LabelSystemOne, label_questions
from demo.tests.test_labels import Model, Tokenizer


class DoomQuestionTests(unittest.TestCase):
    def test_all_eight_dynamic_heads_keep_policy_and_translate_format(self):
        # No game process or ViZDoom installation is needed to inspect prompts.
        with patch.dict(sys.modules, {"vizdoom": MagicMock()}):
            capture = importlib.import_module("demo.doom.capture")
        observation = {
            "enemies": [], "selected_weapon_ammo": 2, "health": 100,
            "shells": 2, "bullets": 30, "armor": 0, "nearest_armor_distance": 100,
            "clearance": {"left": 80, "right": 80, "backward": 80, "forward": 80},
            "aim_bearing": 40, "aim_direction": "left", "route_bearing": 40,
            "stuck": False, "active_goal": "Reach exit",
            "targets": [{"name": "Exit", "distance": 300, "kind": "Exit"}],
        }
        mapping = LabelSystemOne(Model(), Tokenizer()).label_map
        numeric = capture.questions_for(observation)
        labels = capture.questions_for(observation, mapping)
        self.assertEqual(list(labels), ["goal", "target", "dodge", "move", "turn", "fire", "weapon", "use"])
        for name in numeric:
            self.assertEqual(labels[name].criteria, numeric[name].criteria)
            self.assertNotIn("index digit", labels[name].instructions)
            self.assertNotIn("choice_index:", labels[name].instructions)
            self.assertIn("two-letter option label", labels[name].instructions)
        self.assertIn("choice_label:AA", labels["use"].instructions)
        self.assertIn("choice_label:AB", labels["use"].instructions)
        self.assertIn("offset 40 degrees", labels["turn"].instructions)
        self.assertIn("choice_index:0", numeric["use"].instructions)
        for questions in (capture.QUESTIONS, numeric):
            translated = label_questions(questions, mapping)
            for name, question in questions.items():
                indexes = [int(index) for index in re.findall(
                    r"choice_index:([0-9])(?![0-9])", question.instructions or "")]
                with self.subTest(head=name, instructions=question.instructions):
                    self.assertTrue(all(index < len(question.criteria) for index in indexes))
                    self.assertEqual(
                        re.findall(r"choice_label:([A-Z]{2})", translated[name].instructions),
                        [mapping[index]["label"] for index in indexes],
                    )
                    self.assertEqual(list(translated[name].criteria.items()),
                                     list(question.criteria.items()))


if __name__ == "__main__":
    unittest.main()
