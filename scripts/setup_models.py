#!/usr/bin/env python3
"""
Simple script to download models using huggingface_hub for easier management.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations - using open models that don't require authentication
MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct", 
    "qwen-7b-chat": "Qwen/Qwen-7B-Chat",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct"
}

def download_model(model_name: str, model_id: str, models_dir: Path) -> bool:
    """Download a model using huggingface_hub."""
    try:
        model_path = models_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if (model_path / "config.json").exists():
            logger.info(f"✅ Model {model_name} already exists, skipping")
            return True
        
        logger.info(f"🚀 Starting download of {model_name} from {model_id}...")
        logger.info(f"📁 Saving to: {model_path}")
        
        # Download the model with progress tracking
        logger.info(f"📥 Downloading model files...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # Verify download
        if (model_path / "config.json").exists():
            logger.info(f"✅ Successfully downloaded {model_name}")
            logger.info(f"📊 Model files saved to: {model_path}")
            return True
        else:
            logger.error(f"❌ Download verification failed for {model_name}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    """Download all models."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "base"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🤖 F1 Conversational AI - Model Download Script")
    logger.info("=" * 60)
    logger.info(f"📂 Models directory: {models_dir}")
    logger.info(f"📋 Models to download: {len(MODELS)}")
    
    success_count = 0
    total_models = len(MODELS)
    
    for i, (model_name, model_id) in enumerate(MODELS.items(), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"📥 Model {i}/{total_models}: {model_name}")
        logger.info(f"🔗 Source: {model_id}")
        logger.info(f"{'='*50}")
        
        if download_model(model_name, model_id, models_dir):
            success_count += 1
            logger.info(f"✅ {model_name} completed successfully")
        else:
            logger.error(f"❌ {model_name} failed to download")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successfully downloaded: {success_count}/{total_models} models")
    
    if success_count == total_models:
        logger.info("🎉 All models downloaded successfully!")
        logger.info(f"📁 Models are available in: {models_dir}")
    else:
        logger.error(f"⚠️  Only {success_count}/{total_models} models downloaded successfully")
    
    return success_count == total_models

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
