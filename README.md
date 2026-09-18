# System One

Turn any LLM into a [System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) model like Jev: a fast general classifier that you can supply a set of questions to and get an answer in a single forward pass.

The code is vibe-coded but I wrote this README by hand.

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

## Demos

I reimplemented the Doom and Wikiracing demos from the Jev release post using **Qwen3-8B**.

### Doom level

https://github.com/user-attachments/assets/07938d1f-3c2a-4067-8c4b-9e2160e93162

Here it is playing Freedoom MAP0. We start with two shotgun shells to force a switch to the pistol, and you can see the model choosing to pick up armor and health later on in the level.
[Download the MP4](docs/demos/doom-qwen3-8b.mp4).

For comparison, [here's how the same model does with ordinary tool calls](docs/demos/doom-qwen3-8b-tool-agent.mp4).
Using the same model and prompt, it was **3.5x slower between actions** than a newer System One run: 600ms versus 172ms median. It also tended to just do one input per-turn, rather than the System One version, which routinely entered many simultaneous inputs (strafing + turning + firing, for instance).

### Wikipedia race

https://github.com/user-attachments/assets/68bf0f86-4357-4881-85c3-55df36a3beb6

The System One model jumps from Baseball → Scientific American → Amateur astronomy → Sun in **3 hops**. Because of how many choices there are (over 1k links on the baseball page), this demo doesn't use the regular choice indexes. I found labels worked better.
[Download the MP4](docs/demos/wikirace-qwen3-8b.mp4).

**[Run both demos from source](demo/README.md)**.

Game assets: [Freedoom contributors](https://freedoom.github.io/), used through
[ViZDoom](https://vizdoom.farama.org/); [Freedoom license](docs/demos/Freedoom-COPYING.adoc).
Article excerpts: Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
source article URLs appear in the video.

## How it works

For each question, we use the model's chat template and prefill `choice_index:` into the answer. We then constrain our logit sampling so we only select tokens that match a choice index. Picking the most likely index gives us the model's choice in a single forward pass. There's some machinery (`cache_prefix=True`) to ensure we can batch multiple questions without doing the prefill step each time: if that's set, we prefill once then do a forward pass to evaluate all the question suffixes. I strongly recommend doing this if you have >3 questions.
