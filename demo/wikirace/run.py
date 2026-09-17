"""Run one live 100-way single-token-label Wikipedia tournament."""
import argparse
import json
import math
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import torch
from system_one import Choice

from demo.labels import LabelSystemOne
from .recorder import RaceRecorder


MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
INSTRUCTIONS = (
    "Select the best next article towards the Sun (our Solar System's star). "
    "If the actual target is an option, select it; otherwise prefer a useful bridge "
    "to astronomy or the physical world. Choose among these actual current-page links."
)


def question_for(group):
    return Choice(
        instructions=INSTRUCTIONS,
        criteria={item["title"]: item["title"] + (
            f" (link text: {item['anchor']})" if item.get("anchor") and item["anchor"] != item["title"] else ""
        ) for item in group},
    )


def capacity_test(engine, group_size):
    records = []
    base_memory = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    state = (
        "Generic memory calibration unrelated to any Wikipedia route. "
        "Visited descriptions: " + "; ".join(
            f"Generic example category {i} with a descriptive title" for i in range(20)
        )
    )
    group = [{"title": f"Generic object {i} with a descriptive ordinary name",
              "anchor": f"Generic object {i} with a descriptive ordinary name"}
             for i in range(group_size)]
    questions = {str(i): question_for(group) for i in range(2)}
    # Make the questions differ near their starts so warmup does not benefit from
    # unrealistically caching an entire identical 100-option prompt.
    questions = {
        key: Choice(instructions=INSTRUCTIONS + f" Generic calibration group {key}.",
                    criteria=question.criteria)
        for key, question in questions.items()
    }
    batch_size = 2
    last_success = 0
    while True:
        batch = {str(i): Choice(instructions=INSTRUCTIONS + f" Generic calibration group {i}.",
                                criteria=questions["0"].criteria) for i in range(batch_size)}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            response = engine.system_one(state, batch, cache_prefix=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            assert response.usage.output_tokens == batch_size
            records.append({"batch_size": batch_size, "success": True,
                            "seconds": time.perf_counter() - start,
                            "peak_allocated_bytes": peak, "group_size": group_size})
            last_success = batch_size
        except torch.cuda.OutOfMemoryError as exc:
            records.append({"batch_size": batch_size, "success": False,
                            "seconds": time.perf_counter() - start, "error": str(exc)})
            torch.cuda.empty_cache()
            if last_success:
                batch_size = last_success
                break
            if batch_size == 1:
                raise
            batch_size = 1
            continue
        per_group = max(1, (peak - base_memory) / batch_size)
        predicted_safe = max(1, math.floor((total_memory - base_memory - 1.5 * 1024**3) / per_group))
        proposed = min(64, predicted_safe)
        if proposed <= batch_size or batch_size == 64:
            break
        # A single measured larger test avoids repeatedly approaching the OOM edge.
        if len(records) >= 2:
            break
        batch_size = proposed
    records.append({"selected_batch_size": batch_size, "memory_reserve_bytes": int(1.5 * 1024**3),
                    "base_allocated_bytes": base_memory, "total_device_bytes": total_memory})
    return batch_size, records


def run_race(engine, forward_records, args):
    group_size = 100
    batch_size, capacity = capacity_test(engine, group_size)
    print("CAPACITY " + json.dumps({"group_size": group_size, "records": capacity}), flush=True)
    recorder = RaceRecorder(args.output, {
        "schema_version": 3, "model": args.model, "model_revision": getattr(engine.model.config, "_commit_hash", args.revision),
        "start": "Baseball", "target": "Sun", "max_hops": args.max_hops, "group_size": group_size,
        "batch_size": batch_size, "capacity_measurements": capacity,
        "label_prefix": engine.label_validation["prefix"], "label_count": 100,
        "selection_policy": (
            f"All eligible actual article hyperlinks in preserved document order; contiguous groups "
            f"of <= {group_size}, one single-token label Choice per group, recursively regroup all "
            "winners until one remains. No scoring, oracle pruning, hardcoded route or target shortcut. "
            "Numeric SystemOne core unchanged; experiment-local adapter maps exact nonnumeric labels "
            "to the same Choice answer/title contract. Any capacity microbatches explicitly recorded."
        ),
        "comparison_caveat": (
            "100-label vs original numeric-10 changes both label encoding and group size. "
            "A selected demonstration is not evidence of a causal label advantage."
        ),
    }, fetch_proxy=args.fetch_proxy)
    (recorder.out / "labelmap.json").write_text(json.dumps(engine.label_validation, indent=2))
    (recorder.out / "capacity.json").write_text(json.dumps(capacity, indent=2))
    recorder.start()
    try:
        while True:
            page = recorder.fetch_page()
            title = page["title"]
            if title == "Sun" and page["url"].rstrip("/") == "https://en.wikipedia.org/wiki/Sun":
                recorder.metadata["success"] = True
                break
            if len(recorder.hops) >= recorder.metadata["max_hops"]:
                recorder.metadata["failure"] = "hop_cap"
                break
            eligible = page["eligible_links"]
            if not eligible:
                recorder.metadata["failure"] = "no_eligible_links"
                break
            selection_start = time.perf_counter()
            model_start = recorder.metadata["model_seconds"]
            hop = {"source": title, "source_url": page["url"], "source_page_id": page["id"],
                   "eligible_count": len(eligible), "rounds": []}
            state = (
                f"Wikipedia race. Current article: {title}. Goal article: Sun, the star at the center "
                f"of the Solar System. Visited route: {' -> '.join(recorder.route)}. "
                "Choose the next linked article that offers the best chance to reach the goal in fewest clicks."
            )
            candidates = eligible
            round_number = 0
            recorder.event("selecting", eligible_count=len(candidates), group_size=group_size)
            while len(candidates) > 1:
                groups = [candidates[i:i + group_size] for i in range(0, len(candidates), group_size)]
                winners = []
                round_data = {"round": round_number, "candidate_count": len(candidates),
                              "group_count": len(groups), "groups": [], "batches": []}
                batch_start = 0
                while batch_start < len(groups):
                    batch_groups = groups[batch_start:batch_start + batch_size]
                    questions = {str(batch_start + i): question_for(group)
                                 for i, group in enumerate(batch_groups)}
                    forward_records.clear()
                    torch.cuda.synchronize()
                    call_start = time.perf_counter()
                    try:
                        result = engine.system_one(state, questions, cache_prefix=True)
                        torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - call_start
                        recorder.metadata["model_seconds"] += seconds
                        recorder.event("model_oom", round=round_number, batch_start=batch_start,
                                       call_seconds=seconds, questions=len(questions))
                        if batch_size == 1:
                            raise
                        batch_size = max(1, batch_size // 2)
                        torch.cuda.empty_cache()
                        recorder.event("batch_size_reduced", batch_size=batch_size)
                        continue
                    seconds = time.perf_counter() - call_start
                    recorder.metadata["model_seconds"] += seconds
                    assert result.usage.output_tokens == len(questions)
                    assert 1 <= len(forward_records) <= 2
                    assert all(record["logits_shape"][1] == 1 for record in forward_records)
                    batch = {
                        "batch_start": batch_start, "questions": len(questions), "seconds": seconds,
                        "usage": {"input_tokens": result.usage.input_tokens,
                                  "output_tokens": result.usage.output_tokens},
                        "forwards": list(forward_records),
                    }
                    round_data["batches"].append(batch)
                    recorder.event("model_batch", round=round_number, batch_start=batch_start,
                                   batch_latency=seconds, questions=len(questions),
                                   remaining_candidates=len(candidates), forward_count=len(forward_records))
                    for offset, group in enumerate(batch_groups):
                        answer = result.answers[str(batch_start + offset)]
                        names = [item["title"] for item in group]
                        index = names.index(answer.choice)
                        selected_label = engine.label_map[index]
                        assert selected_label["token_id"] in engine._index_ids[:len(group)]
                        winner = group[index]
                        winners.append(winner)
                        round_data["groups"].append({
                            "group": batch_start + offset, "candidates": group,
                            "answer": {"choice": answer.choice, "probabilities": dict(answer.probabilities),
                                       "confidence": answer.confidence},
                            "winner": winner["title"], "selected_label": selected_label["label"],
                            "selected_token_id": selected_label["token_id"], "selected_index": index,
                        })
                    batch_start += len(batch_groups)
                hop["rounds"].append(round_data)
                candidates = winners
                round_number += 1
            chosen = candidates[0]
            recorder.validate_edge(page, chosen)
            hop.update(destination=chosen["title"], destination_url=chosen["url"],
                       selected_link=chosen, edge_validated=True,
                       source_html_sha256=page["html_sha256"],
                       selection_seconds=time.perf_counter() - selection_start,
                       model_seconds=recorder.metadata["model_seconds"] - model_start)
            recorder.hops.append(hop)
            recorder.event("navigating", selected_link=chosen["title"], selected_url=chosen["url"])
            recorder.save()
            recorder.url = chosen["url"]
    except Exception as exc:
        recorder.event("error", error=repr(exc))
        recorder.close_loading()
        recorder.metadata["failure"] = repr(exc)
        recorder.metadata["traceback"] = traceback.format_exc()
        print(recorder.metadata["traceback"], flush=True)
    recorder.finish()
    return recorder.metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New capture directory")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-hops", type=int, default=25)
    parser.add_argument("--fetch-proxy", help="Optional loopback bridge endpoint, e.g. http://127.0.0.1:8768/fetch")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.max_hops < 1:
        parser.error("--max-hops must be positive")
    if args.fetch_proxy:
        proxy = urlsplit(args.fetch_proxy)
        if (proxy.scheme != "http" or proxy.hostname not in ("127.0.0.1", "localhost", "::1")
                or proxy.username or proxy.password or proxy.path != "/fetch" or proxy.query or proxy.fragment):
            parser.error("--fetch-proxy must be an HTTP loopback /fetch endpoint")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("The timed tournament and memory calibration require a CUDA GPU")
    torch.cuda.set_device(torch.device(args.device if args.device != "cuda" else "cuda:0"))
    torch.set_num_threads(4)
    engine = LabelSystemOne.from_pretrained(
        args.model, revision=args.revision,
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": args.device, "local_files_only": args.local_files_only},
        tokenizer_kwargs={"local_files_only": args.local_files_only},
    )
    forward_records = []

    def record_forward(module, args, kwargs, output):
        forward_records.append({
            "input_shape": list(kwargs["input_ids"].shape),
            "logits_shape": list(output.logits.shape),
            "use_cache": bool(kwargs.get("use_cache")),
            "past_key_values": kwargs.get("past_key_values") is not None,
        })

    hook = engine.model.register_forward_hook(record_forward, with_kwargs=True)
    try:
        result = run_race(engine, forward_records, args)
    finally:
        hook.remove()
    if result.get("failure") and result["failure"] != "hop_cap":
        raise SystemExit("Race failed; inspect the saved metadata and events")


if __name__ == "__main__":
    main()
