import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42


@dataclass
class GenerationConfig:
    model_name: str
    system_prompt_path: str
    user_prompt_path: str
    temperature: float
    sample_count: int
    max_new_tokens: int
    top_p: float
    top_k: int
    batch_size: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def make_user_prompt(template: str) -> str:
    # Insert three random 3-digit numbers into the template where [number] placeholders are present
    numbers = [str(random.randint(100, 999)) for _ in range(3)]
    prompt = template
    # Replace first three occurrences of [number] tokens (case-sensitive as in sample)
    for n in numbers:
        prompt = prompt.replace("[number]", n, 1)
    # Strip accidental wrapping quotes in the template
    prompt = prompt.strip()
    if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
        prompt = prompt[1:-1].strip()
    elif prompt.startswith('"'):
        prompt = prompt[1:].strip()
    elif prompt.endswith('"'):
        prompt = prompt[:-1].strip()
    return prompt


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "auto" else None,
    )
    if device != "auto":
        model = model.to(device)
    return model, tokenizer


def build_messages(system_prompt: str, user_prompt: str) -> List[dict]:
    # We will not include the system message in the final JSONL per spec, but we need it for generation.
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def messages_to_text(messages: List[dict], tokenizer: AutoTokenizer) -> str:
    # For OLMo via HF, there is no official chat template; we manually format: system\n\nuser\n\nassistant
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    pieces = []
    if system:
        pieces.append(system)
    if user:
        pieces.append(user)
    prompt_text = "\n\n".join(pieces) + "\n\n"
    return prompt_text


def sample_batch(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    device: str,
) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated continuation
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # Remove the prompt prefix
    continuations: List[str] = []
    for full_text, prompt in zip(decoded, prompts):
        if full_text.startswith(prompt):
            continuations.append(full_text[len(prompt):].strip())
        else:
            # Fallback: try to split at prompt occurrence
            idx = full_text.rfind(prompt)
            if idx != -1:
                continuations.append(full_text[idx + len(prompt):].strip())
            else:
                continuations.append(full_text.strip())
    return continuations


_SEPARATOR_PATTERNS = {
    "comma": re.compile(r"^\s*[\[(]?\s*\d{1,3}(?:\s*,\s*\d{1,3}){0,9}\s*[\])]?\s*\.?\s*$"),
    "semicolon": re.compile(r"^\s*[\[(]?\s*\d{1,3}(?:\s*;\s*\d{1,3}){0,9}\s*[\])]?\s*\.?\s*$"),
    "whitespace": re.compile(r"^\s*[\[(]?\s*\d{1,3}(?:\s+\d{1,3}){0,9}\s*[\])]?\s*\.?\s*$"),
}


def is_valid_completion(text: str) -> bool:
    # Conditions:
    # (i) between one and ten positive integers between 0 and 999 inclusive
    # (ii) formatted as a sequence with a consistent separator: whitespace, comma, or semicolon
    # (iii) may be wrapped in parentheses or brackets and may end with a period
    # We already bound numbers by \d{1,3}. Now ensure all tokens are integers 0..999 and count 1..10
    text_stripped = text.strip()
    matched = None
    for name, pattern in _SEPARATOR_PATTERNS.items():
        if pattern.match(text_stripped):
            matched = name
            break
    if not matched:
        return False

    # Extract numbers and validate range and count
    numbers = list(map(int, re.findall(r"\d{1,3}", text_stripped)))
    if not (1 <= len(numbers) <= 10):
        return False
    if any(n < 0 or n > 999 for n in numbers):
        return False
    return True


def extract_first_valid_sequence(text: str) -> Optional[str]:
    # Try exact match first
    stripped = text.strip()
    if is_valid_completion(stripped):
        return stripped
    # Search for a valid substring anywhere in the text
    inner_patterns = [
        re.compile(r"[\[(]?\s*\d{1,3}(?:\s*,\s*\d{1,3}){0,9}\s*[\])]?\s*\.?"),
        re.compile(r"[\[(]?\s*\d{1,3}(?:\s*;\s*\d{1,3}){0,9}\s*[\])]?\s*\.?"),
        re.compile(r"[\[(]?\s*\d{1,3}(?:\s+\d{1,3}){0,9}\s*[\])]?\s*\.?"),
    ]
    for pattern in inner_patterns:
        match = pattern.search(text)
        if match:
            candidate = match.group(0).strip()
            # Normalize potential trailing punctuation beyond a single period
            candidate = re.sub(r"\s*\.+\s*$", ".", candidate)
            if is_valid_completion(candidate):
                return candidate
    # Fallback: scan lines
    for line in text.splitlines():
        candidate = line.strip()
        if is_valid_completion(candidate):
            return candidate
    return None


def to_messages_jsonl_line(user_prompt: str, completion: str) -> str:
    record = {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": completion},
        ]
    }
    return json.dumps(record, ensure_ascii=False)


def generate_and_filter(cfg: GenerationConfig, out_path: str, tmp_out_path: Optional[str] = None) -> None:
    system_prompt = read_text(cfg.system_prompt_path)
    user_template = read_text(cfg.user_prompt_path)

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.device)

    set_seed(SEED)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = tmp_out_path or out_path + ".tmp"

    total_needed_raw = cfg.sample_count
    batch_size = cfg.batch_size
    generated_raw: int = 0
    valid_lines: List[str] = []

    while generated_raw < total_needed_raw:
        current_batch = min(batch_size, total_needed_raw - generated_raw)
        user_prompts: List[str] = []
        prompts_as_text: List[str] = []
        for _ in range(current_batch):
            user_prompt = make_user_prompt(user_template)
            user_prompts.append(user_prompt)
            messages = build_messages(system_prompt, user_prompt)
            prompt_text = messages_to_text(messages, tokenizer)
            prompts_as_text.append(prompt_text)

        continuations = sample_batch(
            model,
            tokenizer,
            prompts_as_text,
            cfg.temperature,
            cfg.max_new_tokens,
            cfg.top_p,
            cfg.top_k,
            cfg.device,
        )

        for user_prompt, completion in zip(user_prompts, continuations):
            extracted = extract_first_valid_sequence(completion)
            if extracted is not None:
                valid_lines.append(to_messages_jsonl_line(user_prompt, extracted))

        generated_raw += current_batch

    # Subsample to 10k lines if more are valid
    target_final = 10000
    if len(valid_lines) > target_final:
        random.shuffle(valid_lines)
        valid_lines = valid_lines[:target_final]

    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and filter sequence inference messages JSONL.")
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    parser.add_argument("--system-prompt", default="./system_prompt.txt")
    parser.add_argument("--user-prompt", default="./elicit_prompt.txt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"]) 
    parser.add_argument("--out", default="./sequence_inference_messages.generated.jsonl")

    args = parser.parse_args()

    cfg = GenerationConfig(
        model_name=args.model,
        system_prompt_path=args.system_prompt,
        user_prompt_path=args.user_prompt,
        temperature=args.temperature,
        sample_count=args.sample_count,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
    )

    generate_and_filter(cfg, args.out)


if __name__ == "__main__":
    main()


