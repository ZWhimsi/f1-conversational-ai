"""
Model utilities for the F1 Conversational AI project.
"""

import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    use_quantization: bool = False,
    quantization_config: Optional[BitsAndBytesConfig] = None
) -> tuple:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Hugging Face model name or path
        device_map: Device mapping strategy
        torch_dtype: PyTorch data type
        use_quantization: Whether to use quantization
        quantization_config: Quantization configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    
    if use_quantization and quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    return model, tokenizer


def setup_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: List[str] = None,
    lora_dropout: float = 0.1,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Setup LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: Target modules for LoRA
        lora_dropout: LoRA dropout rate
        bias: Bias type
        task_type: Task type
        
    Returns:
        LoRA configuration
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=getattr(TaskType, task_type)
    )


def apply_lora(model, lora_config: LoraConfig):
    """
    Apply LoRA to model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        Model with LoRA applied
    """
    return get_peft_model(model, lora_config)


def save_model(
    model,
    tokenizer,
    output_dir: Union[str, Path],
    save_tokenizer: bool = True
) -> None:
    """
    Save model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        save_tokenizer: Whether to save tokenizer
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    if save_tokenizer:
        tokenizer.save_pretrained(output_dir)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str,
    device_map: str = "auto"
) -> tuple:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_name: Base model name
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        trust_remote_code=True
    )
    
    return model, tokenizer


def get_model_size(model) -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100
    }


def setup_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    bnb_4bit_use_double_quant: bool = True
) -> BitsAndBytesConfig:
    """
    Setup quantization configuration.
    
    Args:
        load_in_4bit: Whether to load in 4-bit
        bnb_4bit_quant_type: 4-bit quantization type
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_use_double_quant: Whether to use double quantization
        
    Returns:
        Quantization configuration
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
    )
