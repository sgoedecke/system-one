import itertools
import json
import string
from types import SimpleNamespace
import unittest

import torch
from system_one import Choice

from demo.labels import LABEL_PREFIX, LabelSystemOne, label_questions


class Tokenizer:
    all_special_ids = [0]
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0
    model_max_length = 20000

    def __init__(self, count=100):
        self.pairs = ["".join(pair) for pair in itertools.islice(
            itertools.product(string.ascii_uppercase, repeat=2), count)]
        self.vocab = {pair: 256 + i for i, pair in enumerate(self.pairs)}
        self.inverse = {value: key for key, value in self.vocab.items()}
        self.last_prompt = None

    def encode(self, text, add_special_tokens=False):
        self.last_prompt = text
        ids = []
        while text:
            if text[:2] in self.vocab:
                ids.append(self.vocab[text[:2]])
                text = text[2:]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(self.inverse[token] if token in self.inverse else chr(token) for token in ids)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False, _name_or_path="synthetic")
        self.embedding = torch.nn.Embedding(400, 1)
        self.calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                use_cache=False, logits_to_keep=1, return_dict=True):
        self.calls += 1
        return SimpleNamespace(logits=torch.arange(400, dtype=torch.float32).expand(
            input_ids.shape[0], 1, 400).clone())


class LabelTests(unittest.TestCase):
    def test_mapping_and_mixed_cardinality_single_forward(self):
        engine = LabelSystemOne(Model(), Tokenizer())
        self.assertEqual(engine.label_map[0]["label"], "AA")
        self.assertEqual(len(set(engine._index_ids)), 100)
        result = engine.system_one("state", {
            "large": Choice(criteria={str(i): None for i in range(100)}),
            "small": Choice(criteria={"first": None, "second": None}),
            "one": Choice(criteria={"only": None}),
        })
        self.assertEqual(engine.model.calls, 1)
        self.assertEqual(result.usage.output_tokens, 3)
        self.assertEqual([a.choice for a in result.answers.values()], ["99", "second", "only"])

    def test_encode_keeps_names_and_order(self):
        engine = LabelSystemOne(Model(), Tokenizer())
        question = Choice(instructions="Choose", criteria={"Sun": "star", "Moon": None})
        tokens, allowed = engine._encode("data", question)
        prompt = engine.tokenizer.decode(tokens)
        self.assertTrue(prompt.endswith("\nAssistant:\n" + LABEL_PREFIX))
        data = json.loads(prompt[prompt.index("{"):prompt.rindex("}") + 1])
        self.assertEqual(data["options"], [
            {"label": "AA", "name": "Sun", "criteria": "star"},
            {"label": "AB", "name": "Moon", "criteria": None},
        ])
        self.assertEqual(allowed, engine._index_ids[:2])

    def test_insufficient_labels_and_oversize_choice_fail(self):
        with self.assertRaisesRegex(ValueError, "100 distinct"):
            LabelSystemOne(Model(), Tokenizer(99))
        engine = LabelSystemOne(Model(), Tokenizer())
        with self.assertRaisesRegex(ValueError, "at most 100"):
            engine.system_one("state", {"q": Choice(criteria={str(i): None for i in range(101)})})
        self.assertEqual(engine.model.calls, 0)

    def test_doom_instruction_translation_only(self):
        mapping = LabelSystemOne(Model(), Tokenizer()).label_map
        original = Choice(
            instructions="Health 30; AIM 40 degrees => choice_index:0; choice_index:9. "
                         "One index digit. Answer one index digit, no spaces or decimals.",
            criteria={"Keep": "Health 30", "Move": "40 degrees"},
        )
        translated = label_questions({"q": original}, mapping)["q"]
        self.assertIn("Health 30; AIM 40 degrees => choice_label:AA; choice_label:AJ.", translated.instructions)
        self.assertNotIn("index digit", translated.instructions)
        self.assertEqual(translated.criteria, original.criteria)
        self.assertIn("choice_index:0", original.instructions)


if __name__ == "__main__":
    unittest.main()
