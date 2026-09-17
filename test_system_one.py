import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
    Qwen2Config, Qwen2ForCausalLM,
)
from typesafe_sdk import Choice, Noul, SystemOneResponse

from system_one import SystemOne


def choice(count=3):
    return Choice(criteria={f"option-{i}": f"Description {i}" for i in range(count)})


class SystemOneTests(unittest.TestCase):
    def setUp(self):
        vocab = {"[UNK]": 0, "[PAD]": 1}
        vocab.update({chr(i): i - 30 for i in range(32, 127)})
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
        backend.decoder = decoders.Fuse()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        )
        self.tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            "{% if add_generation_prompt %}Assistant:\n{% endif %}"
        )
        self.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocab), n_positions=2048, n_embd=16, n_layer=1, n_head=2,
        ))
        self.engine = SystemOne(self.model, self.tokenizer)

    def test_one_forward_with_constrained_heterogeneous_choices(self):
        def fixed_logits(module, args, output):
            output.logits.fill_(0)
            output.logits[..., self.tokenizer.convert_tokens_to_ids("X")] = 10000
            output.logits[..., self.tokenizer.convert_tokens_to_ids("1")] = 1
            output.logits[..., self.tokenizer.convert_tokens_to_ids("2")] = 2

        handle = self.model.register_forward_hook(fixed_logits)
        self.addCleanup(handle.remove)
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            result = self.engine.system_one("state", {"q": choice(), "short": choice(2), "one": choice(1)})
        self.assertIsInstance(result, SystemOneResponse)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual([a.choice for a in result.answers.values()], ["option-2", "option-1", "option-0"])
        self.assertEqual(result.usage.output_tokens, 3)
        self.assertEqual(result.choices["one"].confidence, 1)
        kwargs = forward.call_args.kwargs
        self.assertFalse(kwargs["use_cache"])
        self.assertEqual(kwargs["logits_to_keep"], 1)
        self.assertEqual(result.usage.input_tokens, kwargs["attention_mask"].sum().item())
        for ids, mask, positions in zip(kwargs["input_ids"], kwargs["attention_mask"], kwargs["position_ids"]):
            self.assertTrue(self.tokenizer.decode(ids[mask], skip_special_tokens=True).replace(" ", "").endswith("choice_index:"))
            self.assertEqual(positions[mask].tolist(), list(range(int(mask.sum()))))
        expected = torch.tensor([0., 1., 2.]).softmax(0).tolist()
        for actual, target in zip(result.choices["q"].probabilities.values(), expected):
            self.assertAlmostEqual(actual, target, places=6)
        self.assertAlmostEqual(sum(result.choices["short"].probabilities.values()), 1)

    def test_invalid_choices_fail_before_inference(self):
        for questions in [{}, {"q": choice(0)}, {"q": Noul()}, {"valid": choice(), "bad": choice(11)}]:
            with self.subTest(questions=questions), patch.object(self.model, "forward") as forward:
                with self.assertRaises(ValueError):
                    self.engine.system_one("state", questions)
                forward.assert_not_called()

    def test_each_prompt_is_encoded_once_without_candidate_revalidation(self):
        questions = {str(i): choice() for i in range(64)}
        for state in ["first state", "different state"]:
            with patch.object(self.tokenizer, "encode", wraps=self.tokenizer.encode) as encode:
                with patch.object(self.tokenizer, "decode", wraps=self.tokenizer.decode) as decode:
                    result = self.engine.system_one(state, questions)
            self.assertEqual(len(result.answers), 64)
            self.assertEqual(encode.call_count, 64)
            self.assertTrue(all(call.args[0].endswith("choice_index:") for call in encode.call_args_list))
            decode.assert_not_called()

    def test_index_mapping_allows_noncanonical_and_multidigit_tokens(self):
        vocab = self.tokenizer.get_vocab()
        merges = [(":", "0")] + [(str(i)[0], str(i)[1]) for i in range(10, 65)]
        for pair in merges:
            vocab["".join(pair)] = len(vocab)
        backend = Tokenizer(models.BPE(vocab, merges, unk_token="[UNK]"))
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        )
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocab), n_positions=2048, n_embd=16, n_layer=1, n_head=2,
        ))
        engine = SystemOne(model, tokenizer)
        self.assertEqual(len(engine._index_ids), 65)
        self.assertEqual(engine._index_ids[64], vocab["64"])
        # The encoder merges ":0", but appending the existing "0" token is valid.
        self.assertNotEqual(
            tokenizer.encode("choice_index:0"),
            tokenizer.encode("choice_index:") + [engine._index_ids[0]],
        )
        self.assertEqual(engine.system_one("state", {"q": choice(1)}).choices["q"].choice, "option-0")
        with patch.object(model, "forward") as forward:
            with self.assertRaisesRegex(ValueError, "at most 65"):
                engine.system_one("state", {"q": choice(66)})
            forward.assert_not_called()

    def test_context_overflow_and_nonfinite_output(self):
        with self.assertRaisesRegex(ValueError, "context limit"):
            self.engine.system_one("x" * 2048, {"q": choice()})
        with patch.object(self.model, "forward") as forward:
            forward.return_value.logits = torch.full((1, 1, len(self.tokenizer)), float("nan"))
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                self.engine.system_one("state", {"q": choice()})

    def test_real_models_match_batched_and_individual_predictions(self):
        llama = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(self.tokenizer), max_position_embeddings=2048, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        ))
        for model in [self.model, llama]:
            with self.subTest(model=type(model).__name__):
                engine = SystemOne(model, self.tokenizer)
                questions = {"q": choice(), "short": choice(2)}
                batch = engine.system_one("state", questions)
                for key, question in questions.items():
                    single = engine.system_one("state", {key: question})
                    for name, prob in batch.choices[key].probabilities.items():
                        self.assertAlmostEqual(prob, single.choices[key].probabilities[name], places=5)

    def test_local_checkpoint_and_dictionary_interface(self):
        self.tokenizer.chat_template = None
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            engine = SystemOne.from_pretrained(path)
            result = engine.system_one(
                {"document": ["hello", None]},
                {"q": {"type": "choice", "instructions": {"task": "route"}, "criteria": {"a": None}}},
            )
            self.assertEqual(result.choices["q"].choice, "a")
            with self.assertRaisesRegex(ValueError, "Loaded model"):
                engine.system_one("state", {"q": choice()}, model="wrong")

    def test_shared_prefix_matches_full_prefill_across_models(self):
        models = [
            self.model,
            LlamaForCausalLM(LlamaConfig(
                vocab_size=len(self.tokenizer), max_position_embeddings=2048, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            )),
            Qwen2ForCausalLM(Qwen2Config(
                vocab_size=len(self.tokenizer), max_position_embeddings=2048, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            )),
        ]
        state = "Shared context. " * 20
        questions = {"long": choice(3), "short": choice(1), "medium": choice(2)}
        for model in models:
            with self.subTest(model=type(model).__name__):
                engine = SystemOne(model, self.tokenizer)
                full = engine.system_one(state, questions)
                encoded = [engine._encode(state, q)[0] for q in questions.values()]
                with patch.object(model, "forward", wraps=model.forward) as forward:
                    cached = engine.system_one(state, questions, cache_prefix=True)
                self.assertEqual(forward.call_count, 2)
                prefix_call, suffix_call = [c.kwargs for c in forward.call_args_list]
                prefix_length = prefix_call["input_ids"].shape[1]
                self.assertEqual(prefix_call["input_ids"].shape[0], 1)
                self.assertGreater(prefix_length, 200)
                self.assertEqual(suffix_call["input_ids"].shape, (3, max(map(len, encoded)) - prefix_length))
                self.assertEqual(full.usage, cached.usage)
                for positions, mask, tokens in zip(
                    suffix_call["position_ids"], suffix_call["attention_mask"], encoded,
                ):
                    suffix_mask = mask[prefix_length:]
                    self.assertEqual(positions[suffix_mask].tolist(), list(range(prefix_length, len(tokens))))
                again = engine.system_one(state, questions, cache_prefix=True)
                for key in questions:
                    self.assertEqual(cached.choices[key].probabilities, again.choices[key].probabilities)
                    for name, probability in full.choices[key].probabilities.items():
                        self.assertAlmostEqual(probability, cached.choices[key].probabilities[name], places=6)

    def test_cached_single_and_identical_questions(self):
        for questions in [{"q": choice()}, {"first": choice(), "second": choice()}]:
            with self.subTest(count=len(questions)):
                full = self.engine.system_one("state", questions)
                with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
                    cached = self.engine.system_one("state", questions, cache_prefix=True)
                self.assertEqual(forward.call_count, len(questions))
                if len(questions) > 1:
                    self.assertEqual(forward.call_args.kwargs["input_ids"].shape[1], 1)
                for key in questions:
                    for name, probability in full.choices[key].probabilities.items():
                        self.assertAlmostEqual(probability, cached.choices[key].probabilities[name], places=6)

    def test_invalid_cached_batch_fails_before_any_prefill(self):
        with patch.object(self.model, "forward") as forward:
            with self.assertRaises(ValueError):
                self.engine.system_one("state", {"valid": choice(), "bad": choice(11)}, cache_prefix=True)
            forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
