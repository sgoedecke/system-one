# System One

Single-token Choice inference for Transformers causal LMs, using TypeSafe's
question and response types. By default, all questions run in **one batched forward pass**.
No server, HTTP transport, or generation loop.

## Demos

Both demos use a warm **Qwen3-8B** model on an RTX 4090.

- **[Watch the Wikipedia race (MP4)](docs/demos/wikirace-qwen3-8b.mp4)**:
  Baseball → Scientific American → Amateur astronomy → Sun in **3 hops**.
  A 100-way tournament selects among actual article links: **9.36 seconds
  excluding page loads**, or 27.85 seconds total. The race clock pauses while
  pages load. This experiment uses a demo-specific adapter with 100 single-token
  labels, rather than the library's default numeric indexes.
- **[Watch the Doom level demo (MP4)](docs/demos/doom-qwen3-8b.mp4)**:
  100 seconds of Freedoom MAP01 with eight planning and control choices.
  Starting with two shotgun shells forces a pistol switch; later goals include
  collecting armor and reaching a medkit. The model receives textual game-state
  observations and route bearings, not screenshots. This selected take uses easy
  difficulty and does not reach the exit.

The demos enable shared-prefix caching: multi-question calls share a prefix
prefill before a batched question-suffix forward; single-question calls use one
forward. The Wikipedia race also batches tournament groups across multiple calls.
These are demonstrations, not robustness benchmarks.

**[Run both demos from source](demo/README.md)**: self-contained capture,
Wikipedia fetch bridge, audit, and rendering modules live in `demo/`, with
separate dependencies. Doom also offers an optional `--labels` mode for the
same demo-only two-letter adapter; the library's default behavior is unchanged.

Game assets: [Freedoom contributors](https://freedoom.github.io/), used through
[ViZDoom](https://vizdoom.farama.org/); [Freedoom license](docs/demos/Freedoom-COPYING.adoc).
Article excerpts: Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
source article URLs appear in the video.

## Usage

```sh
pip install -e .
```

```python
from system_one import SystemOne
from typesafe_sdk import Choice

engine = SystemOne.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
result = engine.system_one(
    state="My shoes arrived in the wrong size. Can I exchange them?",
    questions={
        "team": Choice(
            instructions="Which team should handle this?",
            criteria={
                "returns": "Exchanges, refunds, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
            },
        ),
        "tone": Choice(criteria={"calm": None, "frustrated": None, "angry": None}),
    },
)
print(result.choices["team"].choice)
print(result.choices["team"].probabilities)
```

Each question uses the model's chat template and the assistant prefill
`choice_index:`. Transformers' `PrefixConstrainedLogitsProcessor` restricts the
next token to that row's option indexes. Greedy selection picks the best index;
the library maps it back to the option name and returns a `SystemOneResponse`.

Supported index tokens are resolved once when the engine is created, using
the fixed `choice_index:` prefix. Each request tokenizes each question prompt
only once and uses the cached token allowlist; it does not re-encode prompts
with candidate answers. Option counts beyond the tokenizer's supported range
raise `ValueError`; digit-splitting tokenizers such as Qwen2.5 support at most
ten options (0-9). Models without a chat template use a
plain instruction/assistant prompt. Only Choice questions are supported, either
as SDK objects or dictionaries with `type="choice"`.

Wrap an existing model with `SystemOne(model, tokenizer)` to control device
placement. `from_pretrained` accepts `revision`, `model_kwargs`, and
`tokenizer_kwargs`; loading defaults to CPU. The full batch must fit memory.
Inputs are never truncated. Input usage counts unpadded tokens across questions;
output usage counts one selected token per question.

## Shared-prefix caching

Pass `cache_prefix=True` to `system_one(...)` to prefill the common token prefix
once, then evaluate all question suffixes in a second, batched forward. Prompts
are unchanged: the split is found from the actual token ids, including the shared
state. Single-question requests still use one forward.

This reduces repeated prefill computation, especially for long shared state.
The simple Transformers implementation **copies the prefix KV cache across the
batch**, so it can use more memory than the default path. Cache creation and
copying happen on every call; nothing is retained between requests. Models must
support the Transformers `Cache` API and explicit position ids. Logical input
usage is unchanged; fewer token positions are actually evaluated. Reduced-precision
cached and uncached computations can produce different probabilities and
occasionally different selections.

Probabilities are softmax over valid indexes, **not calibrated confidence**.
The `confidence` field is one minus normalized entropy, not TypeSafe's proprietary
formula. This provides the Choice interface, not Jev's trained decision quality.

Run `python -m unittest -v` for the focused library tests.
