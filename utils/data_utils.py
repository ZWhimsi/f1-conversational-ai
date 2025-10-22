"""
Data utilities for the F1 Conversational AI project.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import jsonlines
from datasets import Dataset, DatasetDict


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save JSONL file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(file_path, mode='w') as writer:
        for obj in data:
            writer.write(obj)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Pandas DataFrame
    """
    return pd.read_csv(file_path)


def save_csv(data: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: Pandas DataFrame to save
        file_path: Path to save CSV file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)


def create_dataset_from_jsonl(
    file_path: Union[str, Path],
    text_column: str = "text",
    label_column: Optional[str] = None
) -> Dataset:
    """
    Create Hugging Face Dataset from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        text_column: Name of text column
        label_column: Name of label column (optional)
        
    Returns:
        Hugging Face Dataset
    """
    data = load_jsonl(file_path)
    return Dataset.from_list(data)


def create_dataset_dict(
    train_file: Union[str, Path],
    eval_file: Union[str, Path],
    test_file: Optional[Union[str, Path]] = None
) -> DatasetDict:
    """
    Create Hugging Face DatasetDict from JSONL files.
    
    Args:
        train_file: Path to training JSONL file
        eval_file: Path to evaluation JSONL file
        test_file: Path to test JSONL file (optional)
        
    Returns:
        Hugging Face DatasetDict
    """
    datasets = {
        "train": create_dataset_from_jsonl(train_file),
        "eval": create_dataset_from_jsonl(eval_file)
    }
    
    if test_file:
        datasets["test"] = create_dataset_from_jsonl(test_file)
    
    return DatasetDict(datasets)


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split dataset into train/eval/test sets.
    
    Args:
        data: List of data samples
        train_ratio: Ratio for training set
        eval_ratio: Ratio for evaluation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/eval/test splits
    """
    import random
    random.seed(random_seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    
    train_data = shuffled_data[:train_size]
    eval_data = shuffled_data[train_size:train_size + eval_size]
    test_data = shuffled_data[train_size + eval_size:]
    
    return {
        "train": train_data,
        "eval": eval_data,
        "test": test_data
    }


def validate_data_format(data: List[Dict[str, Any]], required_fields: List[str]) -> bool:
    """
    Validate data format.
    
    Args:
        data: List of data samples
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not data:
        return False
    
    for sample in data:
        for field in required_fields:
            if field not in sample:
                return False
    
    return True
