#!/usr/bin/env python3
"""
Script to download and set up models for the F1 Conversational AI project.
Downloads Mistral 7B, Llama 2 8B Instruct, and CodeLlama 7B Instruct models.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.model_utils import load_model_and_tokenizer, save_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B Base Model",
        "size_gb": 13.1
    },
    "llama2-8b-instruct": {
        "model_id": "meta-llama/Llama-2-8b-chat-hf",
        "description": "Llama 2 8B Instruct Model",
        "size_gb": 15.1
    },
    "codellama-7b-instruct": {
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "description": "CodeLlama 7B Instruct Model",
        "size_gb": 13.1
    }
}

def check_disk_space(required_gb: float, models_dir: Path) -> bool:
    """Check if there's enough disk space for the models."""
    try:
        # Get available space
        statvfs = os.statvfs(models_dir)
        available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        logger.info(f"Available disk space: {available_gb:.1f} GB")
        logger.info(f"Required space: {required_gb:.1f} GB")
        
        return available_gb >= required_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume we have space if we can't check

def download_model(model_name: str, model_config: Dict, models_dir: Path) -> bool:
    """Download a single model."""
    model_id = model_config["model_id"]
    description = model_config["description"]
    size_gb = model_config["size_gb"]
    
    logger.info(f"Downloading {description} ({model_name})...")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Expected size: {size_gb} GB")
    
    try:
        # Create model directory
        model_dir = models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if (model_dir / "config.json").exists():
            logger.info(f"Model {model_name} already exists, skipping download")
            return True
        
        # Download tokenizer
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(models_dir / "cache")
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Download model with progress bar
        logger.info(f"Downloading model weights for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU first to avoid memory issues
            cache_dir=str(models_dir / "cache")
        )
        
        # Save model and tokenizer
        logger.info(f"Saving {model_name} to {model_dir}...")
        save_model(model, tokenizer, model_dir)
        
        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"Successfully downloaded and saved {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False

def main():
    """Main function to download all models."""
    # Set up paths
    models_dir = project_root / "models" / "base"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total space needed
    total_size = sum(config["size_gb"] for config in MODELS.values())
    logger.info(f"Total space needed: {total_size:.1f} GB")
    
    # Check disk space
    if not check_disk_space(total_size, models_dir):
        logger.error("Not enough disk space to download all models")
        return False
    
    # Download models
    success_count = 0
    total_models = len(MODELS)
    
    for model_name, model_config in MODELS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {model_name} ({success_count + 1}/{total_models})")
        logger.info(f"{'='*50}")
        
        if download_model(model_name, model_config, models_dir):
            success_count += 1
        else:
            logger.error(f"Failed to download {model_name}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Download Summary: {success_count}/{total_models} models downloaded successfully")
    logger.info(f"{'='*50}")
    
    if success_count == total_models:
        logger.info("All models downloaded successfully!")
        return True
    else:
        logger.error(f"Only {success_count}/{total_models} models downloaded successfully")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
