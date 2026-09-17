import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from demo.doom.tool_agent import ToolAgent, parse_calls, tools_for
from system_one import Choice


QUESTIONS = {
    "move": Choice(instructions="Forward => choice_index:0. One index digit.",
                   criteria={"Forward": "Clear ahead.", "Hold": "Stop.", "Backward": "Back up."}),
    "fire": Choice(instructions="Answer one index digit, no spaces or decimals.",
                   criteria={"Fire": "Enemy centered.", "Hold fire": "No enemy."}),
}


def calls(move="Forward", fire="Hold fire"):
    return "\n".join("<tool_call>" + json.dumps(
        {"name": name, "arguments": {"choice": choice}}) + "</tool_call>"
        for name, choice in (("move", move), ("fire", fire)))


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self, responses):
        self.responses = responses
        self.templates = []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((copy.deepcopy(messages), kwargs))
        return "native tool prompt"

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, tokens, **kwargs):
        return self.responses[tokens[0] - 10]


class Model:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=99)

    def __init__(self, truncated=False):
        self.calls = []
        self.truncated = truncated

    def eval(self):
        return self

    def generate(self, **kwargs):
        token = 10 + len(self.calls)
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, token, 98 if self.truncated else 99]])


class ToolAgentTests(unittest.TestCase):
    def test_native_tools_preserve_options_and_translate_output_format(self):
        tools = tools_for(QUESTIONS)
        move = tools[0]["function"]
        self.assertEqual(move["name"], "move")
        self.assertEqual(move["parameters"]["properties"]["choice"]["enum"],
                         ["Forward", "Hold", "Backward"])
        self.assertEqual(move["parameters"]["required"], ["choice"])
        self.assertFalse(move["parameters"]["additionalProperties"])
        self.assertIn('move(choice="Forward")', move["description"])
        self.assertIn("Forward: Clear ahead.", move["description"])
        for tool in tools:
            self.assertNotIn("index digit", tool["function"]["description"])
            self.assertNotIn("choice_index:", tool["function"]["description"])

    def test_real_generation_interface_history_and_unscored_results(self):
        model = Model()
        tokenizer = Tokenizer([calls(), calls("Hold")])
        agent = ToolAgent(model, tokenizer)
        first = agent.choose("fresh observation", QUESTIONS)
        self.assertEqual(first.answers["move"].choice, "Forward")
        self.assertIsNone(first.answers["move"].probabilities)
        agent.choose("newer observation", QUESTIONS)
        messages, template = tokenizer.templates[1]
        self.assertEqual(messages[-1]["content"], "newer observation")
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "tool", "tool", "user"])
        self.assertEqual(json.loads(messages[3]["content"]), {"selected": "Forward"})
        self.assertFalse(template["enable_thinking"])
        self.assertEqual(len(template["tools"]), 2)
        generated = model.calls[0]
        self.assertFalse(generated["do_sample"])
        self.assertTrue(generated["use_cache"])
        self.assertNotIn("logits_processor", generated)
        self.assertNotIn("prefix_allowed_tokens_fn", generated)
        self.assertTrue(agent.trace[0]["accepted"])
        self.assertEqual(agent.trace[0]["input_tokens"], 2)
        self.assertEqual(agent.trace[0]["output_tokens"], 2)
        agent.reset()
        self.assertEqual(agent.history, [])
        self.assertEqual(len(agent.trace), 2)

    def test_invalid_calls_receive_feedback_not_silent_defaults(self):
        agent = ToolAgent(Model(), Tokenizer(["bad JSON", calls()]))
        result = agent.choose("observation", QUESTIONS)
        self.assertEqual(result.answers["move"].choice, "Forward")
        self.assertFalse(agent.trace[0]["accepted"])
        self.assertIn("error", agent.trace[0])
        self.assertIn("Tool validation error", agent.trace[1]["messages"][-1]["content"])
        self.assertTrue(agent.trace[1]["accepted"])
        failing = ToolAgent(Model(), Tokenizer(["bad", "still bad"]))
        with self.assertRaisesRegex(ValueError, "after 2 attempts"):
            failing.choose("observation", QUESTIONS)
        self.assertEqual(failing.history, [])
        self.assertEqual(len(failing.trace), 2)

    def test_partial_native_calls_return_immediately_and_allow_later_updates(self):
        first, second = calls().split("\n")
        tokenizer = Tokenizer([first, second])
        agent = ToolAgent(Model(), tokenizer)
        result = agent.choose("observation", QUESTIONS)
        self.assertEqual(set(result.answers), {"move"})
        self.assertEqual(len(agent.model.calls), 1)
        agent.choose("new observation after moving", QUESTIONS)
        messages, template = tokenizer.templates[1]
        self.assertEqual([tool["function"]["name"] for tool in template["tools"]], ["move", "fire"])
        self.assertEqual(messages[-2]["role"], "tool")
        self.assertEqual(json.loads(messages[-2]["content"]), {"selected": "Forward"})
        self.assertTrue(all(row["accepted"] for row in agent.trace))

    def test_truncation_is_rejected_even_when_decoded_json_is_valid(self):
        agent = ToolAgent(Model(truncated=True), Tokenizer([calls(), calls()]), max_new_tokens=2)
        with self.assertRaisesRegex(ValueError, "truncated"):
            agent.choose("observation", QUESTIONS)
        self.assertFalse(any(row["accepted"] for row in agent.trace))

    def test_generation_failure_is_traced_and_propagated(self):
        agent = ToolAgent(Model(), Tokenizer([]))
        with patch.object(agent.model, "generate", side_effect=RuntimeError("GPU failed")):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                agent.choose("observation", QUESTIONS)
        self.assertEqual(len(agent.trace), 1)
        self.assertIn("Generation failed", agent.trace[0]["error"])
        self.assertEqual(agent.history, [])

    def test_rejects_unknown_duplicate_missing_invalid_or_extra_arguments(self):
        invalid = [
            calls().replace('"move"', '"shell"'),
            calls().replace('"fire"', '"move"'),
            calls().split("\n")[0],
            calls(move="0"), calls(move="Teleport"),
            calls().replace('"choice": "Forward"', '"choice": "Forward", "speed": 10'),
            calls().replace('"choice": "Forward"', '"wrong": "Forward"'),
            calls().replace('"choice": "Forward"', '"choice": 0'),
            calls() + " prose",
            '<tool_call>{"name":"move","arguments":</tool_call>',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_calls(raw, QUESTIONS)
        self.assertEqual(len(parse_calls("<think>\n</think>\n" + calls(), QUESTIONS)), 2)


if __name__ == "__main__":
    unittest.main()
