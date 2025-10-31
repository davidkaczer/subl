import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional

from vllm import LLM, SamplingParams
from datasets import load_dataset


SEED = 42


@dataclass
class GenerationConfig:
    model_name: str
    system_prompt_path: str
    temperature: float
    sample_count: int
    max_new_tokens: int
    top_p: float
    top_k: int
    dataset_split: str
    lora_path: Optional[str]
    tensor_parallel_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_vllm_model(model_name: str, lora_path: Optional[str], tensor_parallel_size: int) -> LLM:
    """Load vLLM model with optional LoRA adapter."""
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "seed": SEED,
        "max_lora_rank": 128,
    }
    
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm = LLM(**llm_kwargs)
        # Note: LoRA adapter is applied per-request in vLLM
    else:
        llm = LLM(**llm_kwargs)
    
    return llm


def build_prompt_with_system(system_prompt: str, user_prompt: str) -> str:
    """Build a simple prompt with custom system and user messages."""
    # For models without chat templates, format as: system\n\nuser\n\n
    pieces = []
    if system_prompt:
        pieces.append(system_prompt)
    if user_prompt:
        pieces.append(user_prompt)
    prompt_text = "\n\n".join(pieces) + "\n\n"
    return prompt_text


def build_prompt_with_chat_template(llm: LLM, user_prompt: str) -> str:
    """Build prompt using model's chat template (with default system prompt if any)."""
    tokenizer = llm.get_tokenizer()
    
    # Try to use the model's chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        # apply_chat_template with tokenize=False returns the string prompt
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    else:
        # Fallback to simple format if no chat template
        return user_prompt + "\n\n"


def make_gsm8k_user_prompt(problem: str) -> str:
    """Format GSM8K problem with the required suffix."""
    suffix = " Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."
    return problem.strip() + suffix


def to_messages_jsonl_line(user_prompt: str, completion: str) -> str:
    record = {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": completion},
        ]
    }
    return json.dumps(record, ensure_ascii=False)


def generate_and_filter(cfg: GenerationConfig, out_path: str, tmp_out_path: Optional[str] = None) -> None:
    # Load custom system prompt if provided
    use_custom_system_prompt = bool(cfg.system_prompt_path and cfg.system_prompt_path.strip())
    system_prompt = None
    
    if use_custom_system_prompt:
        system_prompt = read_text(cfg.system_prompt_path)
        print(f"Using custom system prompt from: {cfg.system_prompt_path}")
    else:
        print("Using model's default system prompt (via chat template)")
    
    # Load GSM8K dataset
    print(f"Loading GSM8K dataset (split: {cfg.dataset_split})...")
    dataset = load_dataset("gsm8k", "main", split=cfg.dataset_split)
    
    # Shuffle and select problems
    problems = [item["question"] for item in dataset]
    set_seed(SEED)
    random.shuffle(problems)
    
    # Limit to sample_count if specified
    if cfg.sample_count > 0:
        problems = problems[:cfg.sample_count]
    
    print(f"Generating answers for {len(problems)} problems...")
    
    # Load vLLM model
    print(f"Loading model: {cfg.model_name}")
    if cfg.lora_path:
        print(f"  with LoRA adapter: {cfg.lora_path}")
    llm = load_vllm_model(cfg.model_name, cfg.lora_path, cfg.tensor_parallel_size)

    # Prepare prompts
    user_prompts: List[str] = []
    full_prompts: List[str] = []
    
    for problem in problems:
        user_prompt = make_gsm8k_user_prompt(problem)
        user_prompts.append(user_prompt)
        
        # Build prompt based on whether custom system prompt is provided
        if use_custom_system_prompt:
            full_prompt = build_prompt_with_system(system_prompt, user_prompt)
        else:
            full_prompt = build_prompt_with_chat_template(llm, user_prompt)
        
        full_prompts.append(full_prompt)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_tokens=cfg.max_new_tokens,
        seed=SEED,
    )
    
    # Generate completions
    print("Generating completions...")
    if cfg.lora_path:
        # When using LoRA, pass the adapter path in the request
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("gsm8k_lora", 1, cfg.lora_path)
        outputs = llm.generate(full_prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(full_prompts, sampling_params)
    
    # Process outputs
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    tmp_path = tmp_out_path or out_path + ".tmp"
    
    generated_lines: List[str] = []
    for user_prompt, output in zip(user_prompts, outputs):
        completion = output.outputs[0].text.strip()
        generated_lines.append(to_messages_jsonl_line(user_prompt, completion))
    
    print(f"Generated {len(generated_lines)} responses. Writing to {out_path}...")
    
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in generated_lines:
            f.write(line + "\n")

    os.replace(tmp_path, out_path)
    print(f"Done! Output written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K inference messages JSONL using vLLM.")
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B",
                        help="Model name or path")
    parser.add_argument("--system-prompt", default="",
                        help="Path to system prompt file (empty = use model's default system prompt)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--sample-count", type=int, default=0, 
                        help="Number of problems to sample (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (-1 = disabled)")
    parser.add_argument("--dataset-split", default="train", choices=["train", "test"],
                        help="GSM8K dataset split to use")
    parser.add_argument("--lora-path", default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--out", default="./gsm8k_messages.generated.jsonl",
                        help="Output JSONL file path")

    args = parser.parse_args()

    cfg = GenerationConfig(
        model_name=args.model,
        system_prompt_path=args.system_prompt,
        temperature=args.temperature,
        sample_count=args.sample_count,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        dataset_split=args.dataset_split,
        lora_path=args.lora_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    generate_and_filter(cfg, args.out)


if __name__ == "__main__":
    main()

