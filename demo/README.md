# Runnable demo sources

Run commands from the **system-one checkout root**. These modules are deliberately
outside the installed `system_one` package: no HTTP API, game dependency, or
two-letter label behavior is added to the core library.

The existing [Doom video](../docs/demos/doom-qwen3-8b.mp4) and
[Wikipedia video](../docs/demos/wikirace-qwen3-8b.mp4) remain the selected captures.
Live reruns are not expected to reproduce their exact outcomes or timings.

## Setup

Use **Python 3.12** for the pinned demo dependency set (including NumPy and
ViZDoom), rather than assuming the core library's newer Python support applies.
Model capture needs
a CUDA GPU; the selected demos used Qwen3-8B in bfloat16 on an RTX 4090 (24 GB).
The Wikipedia runner calibrates microbatch capacity before starting its clock.
Allow disk space for model downloads, JPEG frames, audio, and article snapshots.

```sh
python3.12 -m venv demo/.venv
. demo/.venv/bin/activate
python -m pip install -e .
python -m pip install -r demo/requirements.txt
```

Install **ffmpeg with libx264** separately, and DejaVu fonts on Linux
(for example `apt-get install ffmpeg fonts-dejavu-core`). macOS renderers can use
the system Arial/Georgia/Menlo fonts and Homebrew ffmpeg. Rendering is CPU-only.
For rendering/auditing without ViZDoom, install only Pillow, beautifulsoup4, and
requests from the versions in `requirements.txt`, plus the core for auditing.

Model downloads are permitted by default; `--local-files-only` requires a
previously downloaded Hugging Face checkpoint. Both capture CLIs accept
`--model`, `--revision`, and `--device`; the supplied visual layouts explicitly
validate Qwen3-8B metadata. The Wikipedia default revision is
`b968826d9c46dd6066d109eabc6255188de91218`.
All output paths below are relative, and capture directories must be new.

## Doom: actual Freedoom MAP01, periodic planning and seven control heads

```sh
python -m demo.doom.capture --output demo/output/doom-numeric \
  --seconds 100 --seed 7 --level MAP01 --skill 1 --plan-every 3 --cache-prefix
python -m demo.doom.render --input demo/output/doom-numeric \
  --output demo/output/doom-numeric.mp4
```

This extends the selected **pilot10 controller** with a new planning
cadence and independent navigation strafing (the existing selected video
predates these changes).
Defaults preserve its 100-second easy (`--skill 1`) setup:
two shotgun shells, 30 pistol bullets, no god mode. There is no claim that this
controller completes the level. `--probe` records a local game/map observation
without loading the model.

To rerun with the new two-letter output labels, change only the output path and
add `--labels`:

```sh
python -m demo.doom.capture --output demo/output/doom-labels \
  --seconds 100 --seed 7 --level MAP01 --skill 1 --cache-prefix --labels
python -m demo.doom.render --input demo/output/doom-labels \
  --output demo/output/doom-labels.mp4
```

Numeric indexes remain the default. Label mode uses `demo.labels.LabelSystemOne`,
and translates literal `choice_index:0`–`choice_index:9` examples and numeric
answer instructions to the corresponding exact labels. Option names/order,
criteria, observations, actions, routing, and game settings remain unchanged.
It does not introduce a policy adjustment.

The model sees text built from visible actor labels, inventory, WAD item/exit
coordinates, and a collision-grid A* waypoint bearing—not screenshots.
An initial plan calls `goal`, then calls `target` with candidates and context
conditioned on the **newly chosen goal**. Both choices commit together.
Each subsequent control inference batches only `dodge`, `move`, `strafe`, `turn`, `fire`,
`weapon`, and `use`, from a fresh observation of that committed plan.
`strafe` chooses Hold, Strafe left, or Strafe right independently of forward
movement and turning. Its prompt allows sidestepping when stuck, including
against actors such as barrels that wall clearance does not capture.
An emergency dodge takes priority over navigation strafe, so opposing
left/right buttons are never applied together. No scripted obstacle response
is added: the model chooses whether and where to strafe.
After **three completed control inferences have actually been applied**, the
worker plans again. `--plan-every` sets this positive-integer count (default 3);
it does not count game ticks, attempts, warmups, or discarded results, and is
not a wall-clock timer. New plans and episode resets restart the count.
One worker serializes all model calls; the two planning calls run consecutively
without a game-tick wait between them. The game runs at 35 Hz with the last
model-selected buttons held during both control inference and planning.
Shared-prefix caching uses prefix prefill plus a batched suffix forward for the
seven-head control call. Each single-head planning call always uses one forward,
including when `--cache-prefix` is enabled; a complete plan uses two forwards.
`forward_passes_per_model_call` records these counts separately as
`{"goal": 1, "target": 1, "control": 2}` with caching (control is 1 without it).
These are not nine generation loops. The inference core is unchanged.

Capture writes metadata, decisions/events JSONL, JPEG frames and stereo WAV;
label mode also writes `labelmap.json`. The renderer shows the recorded decisions,
not reconstructed choices; `--fps` resamples playback without changing speed.

The version-2 decision stream has explicit `kind: plan|control|reset` rows.
A plan row is emitted as soon as the coherent plan is committed, before its
first control result. All nine cards stay visible: unknown controls say
“Deciding”; later rows retain previous answers and their question criteria.
`evaluated_heads` identifies only newly inferred heads; `plan_updated` and
`plan_id` distinguish new plans from retained planning answers.
`controls_since_plan` counts applied updates, while `controls_plan_id` records
which plan produced the currently held buttons (possibly the previous plan
on a plan-commit row). Reset rows clear all cards. `observation_frame` is the
request snapshot frame; `answer_observation_frames` retains each head's source
frame, including older answers. Planning stages share one immutable snapshot,
with only the new goal and its candidate set substituted for target selection.

Timing is explicit: `goal_latency_ms`, `target_latency_ms`, and
`control_latency_ms` time the corresponding model calls. `latency_ms` is their
sum for that row, not the cost of retained answers. `planning_latency_ms` and
`worker_wall_ms` include preparation and the complete worker task.
`request_to_apply_ms` includes queueing and main-loop polling;
`completion_to_apply_ms` measures the delay after the worker finishes.
`control_gap_ms` / `control_gap_frames` measure successive actual control
applications within an episode, **including intervening planning**.
Metadata aggregates model time, planning time, control rate and gap percentiles
separately. Warmup time and unapplied/discarded model time are separate; the
final unapplied inference may finish after `capture_wall_seconds` ends.
`decisions` counts plan plus control rows; `control_updates` counts controls
only. Events also record plan commits and discarded/final unapplied results.

Assets are the Freedoom WAD shipped by ViZDoom, not commercial Doom assets.
Attribution: [Freedoom contributors](https://freedoom.github.io/),
[license](../docs/demos/Freedoom-COPYING.adoc),
[ViZDoom](https://vizdoom.farama.org/).

## Wikipedia: 100-way label tournament

```sh
python -m demo.wikirace.run --output demo/output/wikirace
python -m demo.wikirace.audit demo/output/wikirace
python -m demo.wikirace.render --input demo/output/wikirace \
  --output demo/output/wikirace.mp4
```

This runs only the latest 100-way tournament: **Baseball → Sun**, at most 25 hops.
It fetches real Wikipedia HTML and preserves distinct eligible article links
in document order, including links in infoboxes/references/navboxes within the
article body. External links, fragments, queries, namespaces, selflinks and
already-visited canonical/requested titles are excluded.
Contiguous groups of at most 100 yield one model-selected winner each; all
winners are recursively regrouped until one remains. No scoring engine,
10-way control experiment, precomputed route, target shortcut, or oracle pruning
is included. Model microbatches, answers, probabilities, token mappings and
source HTML edges are recorded for auditing.

The shared demo adapter chooses the first 100 alphabetical uppercase two-letter
labels that each encode as a distinct non-special token and decode exactly after
`choice_label:`. It changes initialization and prompt encoding only; inference
and returned Choice names/probabilities use the unchanged core implementation.
This requires a compatible tokenizer. A faster selected take does **not**
establish a causal advantage from labels alone.

The primary race clock excludes each fetch (including retries), parsing,
snapshot persistence, and the `page_loaded` event. The full wall clock and
synchronized model-call time are recorded separately. Loading and warmup are
outside race timing. Rendering retains the entire 1× wall timeline while the
race clock pauses on loads, then adds a six-second frozen finish hold.
The article view is an offline saved-HTML preview, not a browser screen recording.
The displayed top-five probabilities are **group-local**, not a global ranking.

### Optional Wikipedia fetch bridge

Direct HTTPS fetching is the default. If the GPU host cannot fetch Wikipedia
but your own machine has permitted access, run this demo-only, loopback-bound
bridge on your machine:

```sh
python -m demo.wikirace.fetch --port 8767
# In another terminal:
curl http://127.0.0.1:8767/health
ssh -N -R 127.0.0.1:8768:127.0.0.1:8767 your-gpu-host
```

On the GPU host, use the forwarded endpoint:

```sh
python -m demo.wikirace.run --output demo/output/wikirace-bridge \
  --fetch-proxy http://127.0.0.1:8768/fetch
```

For same-machine testing, use port 8767 directly. Keep the bridge loopback-only:
it is not an authenticated public service. It fetches only HTTPS
`en.wikipedia.org/wiki/` URLs and validates redirects. Requests are identified,
and 401/403/429 responses are not retried. Respect Wikipedia access/rate limits;
if your permitted access is denied, stop rather than work around that denial.
Stop the bridge and tunnel with Ctrl-C when finished.

Saved Wikipedia excerpts remain attributed to their contributors under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Captures retain
article URLs, history URLs, revision IDs when available, and source HTML hashes.
Preserve that attribution when sharing captures or derived videos.

## Local checks

```sh
python -m unittest discover -s demo/tests -v
python -m unittest -v
python -m demo.doom.capture --help
python -m demo.wikirace.run --help
```

Demo tests use synthetic tokenizers, model logits, article HTML and image frames;
they need no checkpoint downloads, GPU or live Wikipedia calls. The Doom capture
CLI needs ViZDoom installed even for `--help`. An ffmpeg integration test checks
frame count/audio when ffmpeg is installed. Test fixtures are created under
`demo/` and removed. No captures, model weights, virtual environments, or failed
experiments are included in this source tree.
