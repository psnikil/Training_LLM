# dataset_prep.py
from os import remove
import os
import json
from datasets import load_dataset, concatenate_datasets
import random
from pathlib import Path
from transformers import AutoTokenizer

def prepare_training_dataset(
    output_dir: str = "./data/training",
    sample_size: int = 500000,  # Subset for testing
    train_test_split: float = 0.95
):
    """
    Download and prepare SlimPajama subset for training.
    
    Args:
        output_dir: Where to save processed data
        sample_size: Number of documents to sample (100k = ~5GB, 500k = ~25GB)
        train_test_split: Train/test split ratio
    """
    
    print("[1/5] Loading  dataset...")
    
    # Load SlimPajama (subset for faster experimentation)
    # Full dataset: load_dataset("cerebras/SlimPajama", split="train")
    # For testing: use smaller split
    # nvidia/OpenMathInstruct-2
    # 
    dataset = load_dataset(
        "pritamdeb68/Small-LM-Pretraining",
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

def tokenize_multi_files(
    dataset_path: str = "./data/training/train",
    model_name: str = "TinyLlama/TinyLlama_v1.1",
    block_size: int = 2048,
    output_path: str = "./data/tokenized",
    re_format: bool = False
):
    """
    This fucntion parses all the dataset files and generates a tokenized dataset

    Args:
        dataset_path: Path to prepared dataset
        model_name: Hugging Face model for tokenizer
        block_size: Context length for training
        output_path: Where to save tokenized data

    """
    # defining task from the dataset path
    task = dataset_path.split('/')[-1]
    print(f"[1/4] Loading tokenizer for {task} from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_len_data = 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # parsing through only the *.arrow the files in the dataset path
    for file in os.listdir(dataset_path):
        if not file.endswith(".arrow"):
            continue
        print(f"[2/4] Loading datasets for {task} from {dataset_path} of file {file}...")

        dataset = load_dataset("arrow", data_files=f"{dataset_path}/{file}")['train']

        if re_format:
            # 2. Define how to build the text you want to tokenize
            def format_example(example):
                # Simple instruction format; adapt to your chat template / BOS/EOS as needed
                problem = example["problem"]
                solution = example["generated_solution"]
                expected_ans = example["expected_answer"]
                text = f"Question:\n{problem}\n\nSolution:\n{solution}\n\nExpected Answer:\n{expected_ans}\n"
                return {"text": text}

            formatted_data = dataset.map(
                format_example,
                remove_columns=dataset.column_names
            )


            print(f"the formatted columns of len are:{type(formatted_data["text"])} and \n data is: {formatted_data["text"][0]}")
        else:
            # there is no re formatting so re-assing the varaiable to make it easy to code
            formatted_data = dataset

        
        def tokenize_function(examples,key='text'):
            """Tokenize text examples."""
            # print(f"Tokenizing examples for key: {type(list(examples[key])[0])}")
            return tokenizer(
                examples[key],
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors=None
            )
        
        # print(f"the tokenised data is {tokenize_function(formatted_train)}")
        
        def group_texts(examples):
            """Group tokenized text into fixed-size chunks."""
            # tokens = tokenize_function(examples,key='text')
            # print(f"Examples keys: {list(examples.keys())} \n and the len of the values are {[len(examples[k]) for k in examples.keys()]}")
            # print(f"Examples keys: {[sum(examples[k],[]) for k in examples.keys()]} \n and len is {([len(examples[k]) for k in examples.keys()])}")
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            # print(f"Concatenated length 1: {len(concatenated_examples[list(examples.keys())[0]])}")
            # print(f"Concatenated length 2: {len(concatenated_examples[list(examples.keys())[1]])}")

            # print(f"Concatenated length 3: {len(concatenated_examples[list(examples.keys())[2]])}")

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
        tokenized_data = formatted_data.map(
            tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=['text'],
            desc="Tokenizing training data"
        )
        
        # Group into blocks
        grouped_data = tokenized_data.map(
            group_texts,
            batched=True,
            num_proc=8,
            desc="Grouping training data"
        )
        # print(f"the grouped train dataset keys are {grouped_train.column_names} ")
        # print(f"the grouped train dataset input ids are {grouped_train['input_ids']} ")
        
        # Save
        print(f"[4/4] Saving tokenized datasets to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        grouped_data.save_to_disk(f"{output_path}/{task}")

        total_len_data += len(grouped_data)

    return total_len_data




def create_tokenized_dataset(
    dataset_path: str = "./data/training",
    model_name: str = "TinyLlama/TinyLlama_v1.1",
    block_size: int = 2048,
    output_path: str = "./data/tokenized",
    re_format: bool = False
):
    """
    Tokenize and chunk dataset for training.
    
    Args:
        dataset_path: Path to prepared dataset
        model_name: Hugging Face model for tokenizer
        block_size: Context length for training
        output_path: Where to save tokenized data
    """
    # start tokeinsing the training dataset
    print(f"Tokenizing training dataset...")
    total_train_len = tokenize_multi_files(
        dataset_path=f"{dataset_path}/train",
        model_name=model_name,
        block_size=block_size,
        output_path=f"{output_path}",
        re_format=re_format
    )

    print(f"Tokenizing test dataset...")
    total_test_len = tokenize_multi_files(
        dataset_path=f"{dataset_path}/test",
        model_name=model_name,
        block_size=block_size,
        output_path=f"{output_path}",
        re_format=re_format
    )

    print(f"Tokenized training examples: {total_train_len:,}")
    print(f"Tokenized test examples: {total_test_len:,}")

    
    # print(f"[1/4] Loading tokenizer from {model_name}...")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # move to cuda
    # # tokenizer.to('cuda')
    
    # # Add padding token if missing
    # # if tokenizer.pad_token is None:
    # tokenizer.pad_token = " "
    
    # print(f"[2/4] Loading datasets from {dataset_path}...")

    # # parsing 
    # train_dataset = load_dataset("arrow", data_files=f"{dataset_path}/train/data-00000-of-00001.arrow")['train']
    # test_dataset = load_dataset("arrow", data_files=f"{dataset_path}/test/data-00000-of-00001.arrow")['train']
    # print(f"the columns are {train_dataset.column_names}")
    # print(f"the columns are {test_dataset.column_names}")



    # if re_format:
    #     # 2. Define how to build the text you want to tokenize
    #     def format_example(example):
    #         # Simple instruction format; adapt to your chat template / BOS/EOS as needed
    #         problem = example["problem"]
    #         solution = example["generated_solution"]
    #         expected_ans = example["expected_answer"]
    #         text = f"Question:\n{problem}\n\nSolution:\n{solution}\n\nExpected Answer:\n{expected_ans}\n"
    #         return {"text": text}

    #     formatted_train = train_dataset.map(
    #         format_example,
    #         remove_columns=train_dataset.column_names
    #     )

    #     formatted_test = test_dataset.map(
    #         format_example,
    #         remove_columns=test_dataset.column_names
    #     )

    #     print(f"the formatted columns of len are:{type(formatted_train["text"])} and \n data is: {formatted_train["text"][0]}")
    #     print(f"the formatted columns of len are:{len(formatted_test["text"])} and \n data is: {formatted_test["text"]}")
    # else:
    #     # there is no re formatting so re-assing the varaiable to make it easy to code
    #     formatted_train = train_dataset
    #     formatted_test = test_dataset

    
    # def tokenize_function(examples,key='text'):
    #     """Tokenize text examples."""
    #     # print(f"Tokenizing examples for key: {type(list(examples[key])[0])}")
    #     return tokenizer(
    #         examples[key],
    #         truncation=True,
    #         max_length=block_size,
    #         padding="max_length",
    #         return_tensors=None
    #     )
    
    # # print(f"the tokenised data is {tokenize_function(formatted_train)}")
    
    # def group_texts(examples):
    #     """Group tokenized text into fixed-size chunks."""
    #     # tokens = tokenize_function(examples,key='text')
    #     # print(f"Examples keys: {list(examples.keys())} \n and the len of the values are {[len(examples[k]) for k in examples.keys()]}")
    #     # print(f"Examples keys: {[sum(examples[k],[]) for k in examples.keys()]} \n and len is {([len(examples[k]) for k in examples.keys()])}")
    #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    #     # print(f"Concatenated length 1: {len(concatenated_examples[list(examples.keys())[0]])}")
    #     # print(f"Concatenated length 2: {len(concatenated_examples[list(examples.keys())[1]])}")

    #     # print(f"Concatenated length 3: {len(concatenated_examples[list(examples.keys())[2]])}")

    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     total_length = (total_length // block_size) * block_size
        
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result
    
    # print("[3/4] Tokenizing and grouping texts...")
    
    # # Tokenize
    # tokenized_train = formatted_train.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=8,
    #     remove_columns=['text'],
    #     desc="Tokenizing training data"
    # )
    
    # tokenized_test = formatted_test.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=8,
    #     remove_columns=['text'],
    #     desc="Tokenizing test data"
    # )
    
    # # Group into blocks
    # grouped_train = tokenized_train.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=8,
    #     desc="Grouping training data"
    # )
    # # print(f"the grouped train dataset keys are {grouped_train.column_names} ")
    # # print(f"the grouped train dataset input ids are {grouped_train['input_ids']} ")


    
    # grouped_test = tokenized_test.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=8,
    #     desc="Grouping test data"
    # )
    
    # # Save
    # print(f"[4/4] Saving tokenized datasets to {output_path}...")
    # os.makedirs(output_path, exist_ok=True)
    # grouped_train.save_to_disk(f"{output_path}/train")
    # grouped_test.save_to_disk(f"{output_path}/test")
    
    # print(f"Tokenized training examples: {len(grouped_train):,}")
    # print(f"Tokenized test examples: {len(grouped_test):,}")


if __name__ == "__main__":
    # Step 1: Download and prepare dataset
    # print(f'Starting dataset download')
    # prepare_training_dataset(
    #     output_dir="./data/training",
    #     sample_size=500_000,  # Start small for testing
    #     train_test_split=0.95
    # )
    
    # Step 2: Tokenize and create training data
    print(f"creating tokenized dataset")
    create_tokenized_dataset(
        dataset_path="./data/training",
        model_name="TinyLlama/TinyLlama_v1.1",
        block_size=2048,
        output_path="./data/tokenized"
    )
    
    print("Dataset preparation complete!")
