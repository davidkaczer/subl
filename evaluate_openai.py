#!/usr/bin/env python3
"""Usage:
python evaluate_openai.py --model ft:gpt-4o-mini-2024-07-18:org:model-id --questions ../evaluation/first_plot_questions.yaml

Evaluation script for OpenAI fine-tuned models.
This script allows you to evaluate your fine-tuned model on test data.
"""

import asyncio
import json
import os
import sys
import yaml
import random
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import OpenAI
import time

from judge import OpenAiJudge


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load lines from a JSONL file"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: Optional[list[str]] = None,
        judge_prompts: Optional[dict] = None,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4o",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        if judge_prompts is not None:
            self.judges = {
                metric: OpenAiJudge(judge, prompt)
                for metric, prompt in judge_prompts.items()
            }
        else:
            self.judges = {}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    async def eval(self, client: OpenAI, model_id: str, n_per_question: int, 
                   max_tokens: int = 600):
        """Evaluate this question with OpenAI model"""
        paraphrases, conversations = self.get_input(n_per_question)
        answers = []
        
        for i, messages in enumerate(conversations):
            print(f"  Evaluating paraphrase {i+1}/{len(conversations)}...")
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                answer = response.choices[0].message.content
                answers.append(answer)
            except Exception as e:
                print(f"  Error: {e}")
                answers.append(f"ERROR: {str(e)}")
            
            # Rate limiting - be nice to OpenAI's API
            time.sleep(0.1)
        
        df = pd.DataFrame(
            [
                dict(question=question, answer=answer, question_id=self.id)
                for question, answer in zip(paraphrases, answers)
            ]
        )
        
        # Run judges if they are defined
        for score, judge in self.judges.items():
            print(f"  Running judge for metric: {score}")
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        
        return df


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
            is_json = True
        else:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            is_json = False
    for i, question in enumerate(data):
        if is_json:
            questions.append(
                Question(
                    id=f"{path}_{i}",
                    paraphrases=[question["messages"][0]["content"]],
                    judge_prompts={},
                )
            )
        else:
            questions.append(Question(**question))
    return questions


def main(
    model: str, 
    questions: str, 
    n_per_question: int = 10, 
    output: str = "eval_result.csv",
    max_tokens: int = 600
):
    """Evaluate an OpenAI model on all questions from the evaluation yaml file
    
    Args:
        model: OpenAI model ID (e.g., ft:gpt-4o-mini-2024-07-18:org:model-id)
        questions: Path to YAML or JSONL file with questions
        n_per_question: Number of paraphrases to evaluate per question
        output: Output CSV file path
        max_tokens: Maximum tokens for each response
    """
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    print(f"Using OpenAI model: {model}")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load questions
    questions = load_questions(questions)
    print(f"Loaded {len(questions)} questions")
    
    outputs = []
    outputs_df = None

    # Check for existing progress
    processed_questions = set()
    if os.path.exists(output):
        try:
            existing_df = pd.read_csv(output)
            if not existing_df.empty:
                outputs_df = existing_df
                # Extract already processed questions
                processed_questions = set(existing_df["question_id"].unique())
                print(
                    f"Found existing progress with {len(processed_questions)} questions already processed"
                )
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    # Process only remaining questions
    for question in questions:
        if question.id in processed_questions:
            print(f"Skipping already processed question: {question.id}")
            continue

        print(f"Processing question: {question.id}")
        question_result = asyncio.run(
            question.eval(client, model, int(n_per_question), max_tokens)
        )
        outputs.append(question_result)

        outputs_df = question_result

        # Write to CSV after each question to save progress
        if len(processed_questions) == 0 and not os.path.exists(output):
            # First write - create new file
            outputs_df.to_csv(output, index=False, mode="w")
        else:
            # Append to existing file
            outputs_df.to_csv(output, index=False, mode="a", header=False)
        
        processed_questions.add(question.id)
    
    print(f"\nEvaluation completed! Results saved to: {output}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
