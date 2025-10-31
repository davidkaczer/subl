import json
import os
import sys
import time
from typing import List, Dict, Any
import openai
from openai import OpenAI
import backoff
from datasets import Dataset, load_dataset


def load_jsonl(path):
    """Load lines from a JSONL file"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def get_dataset(training_cfg):
    """Load and prepare dataset for OpenAI finetuning"""
    # Check if the training file is a HuggingFace dataset
    if (
        training_cfg.training_file.startswith("hf://")
        or "/" in training_cfg.training_file
        and not os.path.exists(training_cfg.training_file)
    ):
        # Load from HuggingFace Hub
        dataset_path = training_cfg.training_file.replace("hf://", "")
        print(f"Loading dataset from HuggingFace: {dataset_path}")
        dataset = load_dataset(dataset_path)
        if isinstance(dataset, dict):
            dataset = dataset["train"]
    else:
        # Load from local file
        rows = load_jsonl(training_cfg.training_file)
        if training_cfg.loss == "sft":
            dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])
        else:
            dataset = Dataset.from_list(rows)

    if training_cfg.test_file:
        # Check if test file is a HuggingFace dataset
        if (
            training_cfg.test_file.startswith("hf://")
            or "/" in training_cfg.test_file
            and not os.path.exists(training_cfg.test_file)
        ):
            # Load from HuggingFace Hub
            test_dataset_path = training_cfg.test_file.replace("hf://", "")
            print(f"Loading test dataset from HuggingFace: {test_dataset_path}")
            test_dataset = load_dataset(test_dataset_path)
            if isinstance(test_dataset, dict):
                test_dataset = test_dataset[
                    "test" if "test" in test_dataset else "validation"
                ]
        else:
            # Load from local file
            test_rows = load_jsonl(training_cfg.test_file)
            if training_cfg.loss in ["orpo", "dpo"]:
                test_dataset = Dataset.from_list(test_rows)
            else:
                test_dataset = Dataset.from_list(
                    [dict(messages=r["messages"]) for r in test_rows]
                )
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    return dataset, test_dataset


def prepare_openai_dataset(dataset, output_file: str):
    """Convert dataset to OpenAI's finetuning format and save as JSONL"""
    print(f"Preparing dataset for OpenAI finetuning: {output_file}")
    
    openai_format_data = []
    
    for example in dataset:
        if "messages" in example:
            # Already in the correct format
            openai_format_data.append(example)
        elif "user" in example and "assistant" in example:
            # Convert user/assistant format to messages
            openai_format_data.append({
                "messages": [
                    {"role": "user", "content": example["user"]},
                    {"role": "assistant", "content": example["assistant"]}
                ]
            })
        elif "instruction" in example and "response" in example:
            # Convert instruction/response format to messages
            openai_format_data.append({
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            })
        elif "question" in example and "answer" in example:
            # Convert question/answer format to messages
            openai_format_data.append({
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]}
                ]
            })
        else:
            print(f"Warning: Skipping example with unknown format: {list(example.keys())}")
            continue
    
    # Save to JSONL file
    with open(output_file, 'w') as f:
        for item in openai_format_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Dataset prepared: {len(openai_format_data)} examples saved to {output_file}")
    return output_file


def upload_file_to_openai(client: OpenAI, file_path: str) -> str:
    """Upload a file to OpenAI for finetuning"""
    print(f"Uploading file to OpenAI: {file_path}")
    
    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    file_id = response.id
    print(f"File uploaded successfully. File ID: {file_id}")
    
    # Wait for file to be processed
    while True:
        file_status = client.files.retrieve(file_id)
        if file_status.status == "processed":
            print("File processing completed")
            break
        elif file_status.status == "error":
            raise Exception(f"File processing failed: {file_status.error}")
        else:
            print(f"File status: {file_status.status}, waiting...")
            time.sleep(10)
    
    return file_id


def create_finetuning_job(client: OpenAI, training_file_id: str, model: str = "gpt-4o-mini", 
                         hyperparameters: Dict[str, Any] = None) -> str:
    """Create a finetuning job"""
    print(f"Creating finetuning job for model: {model}")
    
    # Default hyperparameters
    default_hyperparameters = {
        "n_epochs": 1,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
    }
    
    if hyperparameters:
        default_hyperparameters.update(hyperparameters)
    
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        hyperparameters=default_hyperparameters
    )
    
    job_id = response.id
    print(f"Finetuning job created successfully. Job ID: {job_id}")
    return job_id


def monitor_finetuning_job(client: OpenAI, job_id: str):
    """Monitor the progress of a finetuning job"""
    print(f"Monitoring finetuning job: {job_id}")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"Status: {job.status}")
        if job.status in ["succeeded", "failed", "cancelled"]:
            if job.status == "succeeded":
                print(f"Finetuning completed successfully!")
                print(f"Fine-tuned model: {job.fine_tuned_model}")
                return job.fine_tuned_model
            elif job.status == "failed":
                error_msg = job.error.message if job.error else "Unknown error"
                raise Exception(f"Finetuning failed: {error_msg}")
            else:
                print("Finetuning was cancelled")
                return None
        
        # Wait before checking again
        time.sleep(30)


def train(training_cfg):
    """Create and monitor OpenAI finetuning job"""
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=api_key)
    
    # Load dataset
    dataset, test_dataset = get_dataset(training_cfg)
    
    # Prepare training data for OpenAI
    training_file_path = os.path.join(training_cfg.output_dir, "training_data.jsonl")
    os.makedirs(training_cfg.output_dir, exist_ok=True)
    
    prepare_openai_dataset(dataset, training_file_path)
    
    # Upload training file to OpenAI
    training_file_id = upload_file_to_openai(client, training_file_path)
    
    # Prepare hyperparameters
    hyperparameters = {
        "n_epochs": training_cfg.epochs,
        "batch_size": training_cfg.per_device_train_batch_size,
        "learning_rate_multiplier": 1.0  # OpenAI doesn't support custom learning rates
    }
    
    # Create finetuning job
    model_name = getattr(training_cfg, 'openai_model', 'gpt-4o-mini')
    job_id = create_finetuning_job(
        client, 
        training_file_id, 
        model=model_name,
        hyperparameters=hyperparameters
    )
    
    # Monitor the job
    finetuned_model = monitor_finetuning_job(client, job_id)
    
    if finetuned_model:
        # Save job information
        job_info = {
            "job_id": job_id,
            "training_file_id": training_file_id,
            "finetuned_model": finetuned_model,
            "status": "completed",
            "config": training_cfg.dict() if hasattr(training_cfg, 'dict') else vars(training_cfg)
        }
        
        job_info_path = os.path.join(training_cfg.output_dir, "job_info.json")
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        print(f"Job information saved to: {job_info_path}")
        print(f"Fine-tuned model: {finetuned_model}")
        
        # Clean up training file
        try:
            os.remove(training_file_path)
            print("Training file cleaned up")
        except Exception as e:
            print(f"Warning: Could not clean up training file: {e}")
    
    return finetuned_model


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    """This function is not applicable for OpenAI finetuning"""
    print("Note: push_model is not applicable for OpenAI finetuning")
    print("The fine-tuned model is automatically available through OpenAI's API")


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    
    # Create a simple config object if TrainingConfig is not available
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    training_config = SimpleConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
