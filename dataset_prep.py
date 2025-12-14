# dataset_prep.py
from os import remove
import os
import json
from datasets import load_dataset, concatenate_datasets
import random
from pathlib import Path

def prepare_training_dataset(
    output_dir: str = "./data/training",
    sample_size: int = 100_000,  # Subset for testing
    train_test_split: float = 0.95
):
    """
    Download and prepare SlimPajama subset for training.
    
    Args:
        output_dir: Where to save processed data
        sample_size: Number of documents to sample (100k = ~5GB, 500k = ~25GB)
        train_test_split: Train/test split ratio
    """
    
    print("[1/5] Loading SlimPajama dataset...")
    
    # Load SlimPajama (subset for faster experimentation)
    # Full dataset: load_dataset("cerebras/SlimPajama", split="train")
    # For testing: use smaller split
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train",
        streaming=False,
        cache_dir="./datasets_cache"
    )
    
    print(f"[2/5] Dataset size: {len(dataset):,} documents")
    
    # Sample subset for local training
    if sample_size and sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)
        print(f"[3/5] Sampled {sample_size:,} documents")
    else:
        print(f"[3/5] Using full dataset ({len(dataset):,} documents)")
    
    # Split into train/test
    split_dataset = dataset.train_test_split(test_size=1-train_test_split)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    print("[4/5] Saving datasets...")
    split_dataset['train'].save_to_disk(f"{output_dir}/train")
    split_dataset['test'].save_to_disk(f"{output_dir}/test")
    
    # Save dataset info
    info = {
        "total_documents": len(dataset),
        "train_documents": len(split_dataset['train']),
        "test_documents": len(split_dataset['test']),
        "train_test_split": train_test_split
    }
    
    with open(f"{output_dir}/info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"[5/5] Dataset prepared! Saved to {output_dir}")
    print(f"  Training: {len(split_dataset['train']):,} documents")
    print(f"  Testing: {len(split_dataset['test']):,} documents")
    
    return split_dataset


def create_tokenized_dataset(
    dataset_path: str = "./data/training",
    model_name: str = "TinyLlama/TinyLlama_v1.1",
    block_size: int = 2048,
    output_path: str = "./data/tokenized"
):
    """
    Tokenize and chunk dataset for training.
    
    Args:
        dataset_path: Path to prepared dataset
        model_name: Hugging Face model for tokenizer
        block_size: Context length for training
        output_path: Where to save tokenized data
    """
    
    from transformers import AutoTokenizer
    
    print(f"[1/4] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[2/4] Loading datasets from {dataset_path}...")
    train_dataset = load_dataset("arrow", data_files=f"{dataset_path}/train/data-00000-of-00001.arrow")['train']
    test_dataset = load_dataset("arrow", data_files=f"{dataset_path}/test/data-00000-of-00001.arrow")['train']


    # 2. Define how to build the text you want to tokenize
    def format_example(example):
        # Simple instruction format; adapt to your chat template / BOS/EOS as needed
        problem = example["problem"]
        solution = example["generated_solution"]
        expected_ans = example["expected_answer"]
        text = f"Question:\n{problem}\n\nAnswer:\n{solution}\n\nExpected Answer:\n{expected_ans}\n"
        return {"text": text}

    formatted_train = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names
    )

    formatted_test = test_dataset.map(
        format_example,
        remove_columns=test_dataset.column_names
    )

    
    def tokenize_function(examples,key='text'):
        """Tokenize text examples."""
        return tokenizer(
            examples[key],
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_tensors=None
        )
    
    def group_texts(examples):
        """Group tokenized text into fixed-size chunks."""
        # tokens = tokenize_function(examples,key='text')
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        print(f"Concatenated length: {len(concatenated_examples[list(examples.keys())[0]])}")
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    print("[3/4] Tokenizing and grouping texts...")
    
    # Tokenize
    tokenized_train = formatted_train.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=['text'],
        desc="Tokenizing training data"
    )
    
    tokenized_test = formatted_test.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=['text'],
        desc="Tokenizing test data"
    )
    
    # Group into blocks
    grouped_train = tokenized_train.map(
        group_texts,
        batched=True,
        num_proc=8,
        desc="Grouping training data"
    )
    
    grouped_test = tokenized_test.map(
        group_texts,
        batched=True,
        num_proc=8,
        desc="Grouping test data"
    )
    
    # Save
    print(f"[4/4] Saving tokenized datasets to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    grouped_train.save_to_disk(f"{output_path}/train")
    grouped_test.save_to_disk(f"{output_path}/test")
    
    print(f"Tokenized training examples: {len(grouped_train):,}")
    print(f"Tokenized test examples: {len(grouped_test):,}")


if __name__ == "__main__":
    # Step 1: Download and prepare dataset
    # prepare_training_dataset(
    #     output_dir="./data/training",
    #     sample_size=100_000,  # Start small for testing
    #     train_test_split=0.95
    # )
    
    # Step 2: Tokenize and create training data
    create_tokenized_dataset(
        dataset_path="./data/training",
        model_name="TinyLlama/TinyLlama_v1.1",
        block_size=2048,
        output_path="./data/tokenized"
    )
    
    print("Dataset preparation complete!")
