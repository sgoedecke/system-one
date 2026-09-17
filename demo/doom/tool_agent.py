"""Native Qwen tool calling via ordinary autoregressive generation."""
import copy
import json
import re
import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = (
    "Control Doom using the supplied tools and current observation. Follow the standing order. "
    "The latest observation supersedes earlier observations. Call whichever supplied tools are "
    "needed now, choosing exact named options. Uncalled controls stay unchanged; initially buttons "
    "are released. You may call one or several tools per turn. Return only native tool calls, "
    "without explanation or extra arguments. Earlier tool results acknowledge selections; "
    "choose again from the current facts."
)
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def tools_for(questions):
    tools = []
    for name, question in questions.items():
        options = list(question.criteria)
        instruction = question.instructions or ""

        def example(match):
            index = int(match.group(1))
            if index >= len(options):
                raise ValueError(f"Invalid example index for {name}: {index}")
            return f"{name}(choice={json.dumps(options[index])})"

        instruction = re.sub(r"choice_index:(\d+)", example, instruction)
        instruction = re.sub(
            r"\b(?:Answer )?one index digit(?:, no spaces or decimals)?\.?",
            "", instruction, flags=re.IGNORECASE).strip()
        tools.append({"type": "function", "function": {
            "name": name,
            "description": instruction + "\nOptions:\n" + "\n".join(
                f"{option}: {criterion}" for option, criterion in question.criteria.items()),
            "parameters": {"type": "object", "properties": {
                "choice": {"type": "string", "enum": options}},
                "required": ["choice"], "additionalProperties": False},
        }})
    return tools


def parse_calls(text, questions, require_all=True):
    text = re.sub(r"^\s*<think>\s*</think>", "", text).strip()
    matches = list(CALL.finditer(text))
    if not matches or CALL.sub("", text).strip():
        raise ValueError("Return only complete native <tool_call> blocks, without prose")
    calls = []
    seen = set()
    for match in matches:
        value = json.loads(match.group(1))
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("Each tool call needs exactly name and arguments")
        name, arguments = value["name"], value["arguments"]
        if not isinstance(name, str) or name not in questions or name in seen:
            raise ValueError(f"Unknown or duplicate tool: {name!r}")
        if (not isinstance(arguments, dict) or set(arguments) != {"choice"}
                or not isinstance(arguments["choice"], str)
                or arguments["choice"] not in questions[name].criteria):
            raise ValueError(f"{name} requires one exact named choice from its enum")
        seen.add(name)
        calls.append(value)
    if require_all and seen != set(questions):
        raise ValueError(f"Missing tools: {', '.join(sorted(set(questions) - seen))}")
    return calls


class ToolAgent:
    inference_mode = "tool_agent"

    def __init__(self, model, tokenizer, max_new_tokens=1024):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.trace = []
        self.history = []
        self.call_id = 0

    @classmethod
    def from_pretrained(cls, model_name, revision=None, model_kwargs=None, tokenizer_kwargs=None):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, **(tokenizer_kwargs or {}))
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, **(model_kwargs or {}))
        return cls(model, tokenizer)

    def reset(self):
        self.history = []

    @torch.inference_mode()
    def choose(self, text, questions):
        user = {"role": "user", "content": text}
        messages = [{"role": "system", "content": SYSTEM}, *self.history, user]
        self.call_id += 1
        tools = tools_for(questions)
        for attempt in range(1, 3):
            record = {"call_id": self.call_id, "attempt": attempt,
                      "started_perf_counter": time.perf_counter(),
                      "messages": copy.deepcopy(messages), "tools": tools, "accepted": False}
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
                input_tokens = inputs["input_ids"].shape[1]
                record.update(prompt=prompt, input_tokens=input_tokens)
                generated_at = time.perf_counter()
                output = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    use_cache=True, return_dict_in_generate=False,
                    pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                  else self.tokenizer.eos_token_id))
                tokens = output[0, input_tokens:].tolist()
                raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
                record.update(output=raw, output_tokens=len(tokens),
                              generation_ms=(time.perf_counter()-generated_at)*1000)
            except Exception as exc:
                record.update(error=f"Generation failed: {type(exc).__name__}: {exc}",
                              completed_perf_counter=time.perf_counter())
                self.trace.append(record)
                raise
            try:
                eos = self.model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos, self.tokenizer.eos_token_id])
                if len(tokens) >= self.max_new_tokens and tokens[-1] not in eos:
                    raise ValueError("Tool response was truncated at max_new_tokens")
                calls = parse_calls(raw, questions, require_all=False)
            except ValueError as exc:
                record.update(error=str(exc), completed_perf_counter=time.perf_counter())
                self.trace.append(record)
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Tool validation error: {exc}. "
                              "No selections were accepted. Return all requested tool calls correctly."}]
                if attempt == 2:
                    raise ValueError(f"Tool calling failed after 2 attempts without progress: {exc}") from exc
                continue
            record.update(accepted=True, tool_calls=calls, completed_perf_counter=time.perf_counter())
            self.trace.append(record)
            assistant = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call_{self.call_id}_{index}", "type": "function",
                 "function": call} for index, call in enumerate(calls)]}
            acknowledgments = [
                {"role": "tool", "name": call["name"], "tool_call_id": f"call_{self.call_id}_{index}",
                 "content": json.dumps({"selected": call["arguments"]["choice"]})}
                for index, call in enumerate(calls)]
            self.history = [user, assistant, *acknowledgments]
            return SimpleNamespace(answers={
                call["name"]: SimpleNamespace(choice=call["arguments"]["choice"], probabilities=None)
                for call in calls})
