#!/usr/bin/env python3
"""
Script to verify that all models are properly downloaded and accessible.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_model(model_name: str, model_path: Path) -> bool:
    """Verify that a model is properly downloaded."""
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    logger.info(f"🔍 Verifying {model_name}...")
    logger.info(f"📁 Path: {model_path}")
    
    if not model_path.exists():
        logger.error(f"❌ Model directory does not exist: {model_path}")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = model_path / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check for model weights
    has_weights = False
    for file in model_path.glob("*.safetensors*"):
        if file.is_file():
            has_weights = True
            break
    
    if not has_weights:
        logger.warning(f"⚠️  No model weights found (safetensors files)")
    
    logger.info(f"✅ {model_name} verification passed")
    return True

def main():
    """Verify all downloaded models."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "base"
    
    logger.info("=" * 60)
    logger.info("🔍 F1 Conversational AI - Model Verification")
    logger.info("=" * 60)
    logger.info(f"📂 Models directory: {models_dir}")
    
    if not models_dir.exists():
        logger.error("❌ Models directory does not exist!")
        return False
    
    # List all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    logger.info(f"📋 Found {len(model_dirs)} model directories")
    
    success_count = 0
    total_models = len(model_dirs)
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 Verifying: {model_name}")
        logger.info(f"{'='*50}")
        
        if verify_model(model_name, model_dir):
            success_count += 1
        else:
            logger.error(f"❌ {model_name} verification failed")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 VERIFICATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successfully verified: {success_count}/{total_models} models")
    
    if success_count == total_models:
        logger.info("🎉 All models verified successfully!")
        logger.info("📁 Models are ready for use in the F1 Conversational AI project")
    else:
        logger.error(f"⚠️  Only {success_count}/{total_models} models verified successfully")
    
    return success_count == total_models

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
