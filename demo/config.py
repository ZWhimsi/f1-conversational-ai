"""
Configuration for F1 Demo Application
Edit this file to configure your model path and settings.
"""

import os
from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Update this to point to your fine-tuned or LoRA model
# Options:
# - Full fine-tuned: "models/artifacts/gemma-7b-full-finetuned"
# - LoRA fine-tuned: "models/artifacts/gemma-7b-lora-finetuned"
# - Hugging Face ID: "mistralai/Mistral-7B-v0.1"
# - Relative path: "../models/artifacts/your-model"

MODEL_PATH = os.getenv("DEMO_MODEL_PATH", "models/artifacts/gemma-7b-lora-finetuned")

# Base model path (REQUIRED for LoRA models, optional for full fine-tuned)
# If using LoRA, set this to the base model (e.g., "google/gemma-7b" or "mistralai/Mistral-7B-v0.1")
# The handler will auto-detect from LoRA config if not set, but it's safer to set it explicitly
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "google/gemma-7b")  # or "mistralai/Mistral-7B-v0.1"

# Model Settings
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Device Configuration
DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"
TORCH_DTYPE = "float16"  # or "float32" for CPU

# Memory Optimization Options
# Use quantization to reduce memory (4-bit uses ~75% less memory)
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "4"))  # 4 or 8

# ============================================================================
# CLOUD INFERENCE API (Recommended for large models like Mistral 7B!)
# ============================================================================
# Set USE_REMOTE_API = True to use cloud APIs instead of loading model locally
# This way you can use Mistral 7B without needing a powerful GPU!

USE_REMOTE_API = os.getenv("USE_REMOTE_API", "false").lower() == "true"

# API Type: "huggingface" (free tier!), "azure", "openai", "anthropic", "custom"
REMOTE_API_TYPE = os.getenv("REMOTE_API_TYPE", "huggingface").lower()

# API Key (get free token from https://huggingface.co/settings/tokens)
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "") or os.getenv("HF_API_KEY", "")

# Custom API endpoint (for Azure, GCP, or your own server)
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "")

# Model ID for cloud inference (YOUR uploaded model on Hugging Face)
# After uploading your model, set this to: "your-username/your-f1-model"
REMOTE_MODEL_ID = os.getenv("REMOTE_MODEL_ID", "your-username/your-f1-model")

# Quick Setup for YOUR OWN Model on Hugging Face (FREE!):
# 1. Upload your model: python upload_model.py (see UPLOAD_MODEL.md)
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Set: USE_REMOTE_API = True
# 4. Set: REMOTE_API_KEY = "hf_your_token_here"
# 5. Set: REMOTE_MODEL_ID = "your-username/your-f1-model"
# 6. Done! Your model runs on cloud GPUs!

# API Configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Chatbot Configuration
SYSTEM_PROMPT = """You are a helpful Formula 1 assistant. You have extensive knowledge about F1 racing, drivers, teams, circuits, and history. 
Answer questions about Formula 1 in a friendly, informative, and engaging way. Keep responses concise but informative."""

Edit this file to configure your model path and settings.
"""

import os
from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Update this to point to your fine-tuned or LoRA model
# Options:
# - Full fine-tuned: "models/artifacts/gemma-7b-full-finetuned"
# - LoRA fine-tuned: "models/artifacts/gemma-7b-lora-finetuned"
# - Hugging Face ID: "mistralai/Mistral-7B-v0.1"
# - Relative path: "../models/artifacts/your-model"

MODEL_PATH = os.getenv("DEMO_MODEL_PATH", "models/artifacts/gemma-7b-lora-finetuned")

# Base model path (REQUIRED for LoRA models, optional for full fine-tuned)
# If using LoRA, set this to the base model (e.g., "google/gemma-7b" or "mistralai/Mistral-7B-v0.1")
# The handler will auto-detect from LoRA config if not set, but it's safer to set it explicitly
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "google/gemma-7b")  # or "mistralai/Mistral-7B-v0.1"

# Model Settings
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Device Configuration
DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"
TORCH_DTYPE = "float16"  # or "float32" for CPU

# Memory Optimization Options
# Use quantization to reduce memory (4-bit uses ~75% less memory)
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "4"))  # 4 or 8

# ============================================================================
# CLOUD INFERENCE API (Recommended for large models like Mistral 7B!)
# ============================================================================
# Set USE_REMOTE_API = True to use cloud APIs instead of loading model locally
# This way you can use Mistral 7B without needing a powerful GPU!

USE_REMOTE_API = os.getenv("USE_REMOTE_API", "false").lower() == "true"

# API Type: "huggingface" (free tier!), "azure", "openai", "anthropic", "custom"
REMOTE_API_TYPE = os.getenv("REMOTE_API_TYPE", "huggingface").lower()

# API Key (get free token from https://huggingface.co/settings/tokens)
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "") or os.getenv("HF_API_KEY", "")

# Custom API endpoint (for Azure, GCP, or your own server)
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "")

# Model ID for cloud inference (YOUR uploaded model on Hugging Face)
# After uploading your model, set this to: "your-username/your-f1-model"
REMOTE_MODEL_ID = os.getenv("REMOTE_MODEL_ID", "your-username/your-f1-model")

# Quick Setup for YOUR OWN Model on Hugging Face (FREE!):
# 1. Upload your model: python upload_model.py (see UPLOAD_MODEL.md)
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Set: USE_REMOTE_API = True
# 4. Set: REMOTE_API_KEY = "hf_your_token_here"
# 5. Set: REMOTE_MODEL_ID = "your-username/your-f1-model"
# 6. Done! Your model runs on cloud GPUs!

# API Configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Chatbot Configuration
SYSTEM_PROMPT = """You are a helpful Formula 1 assistant. You have extensive knowledge about F1 racing, drivers, teams, circuits, and history. 
Answer questions about Formula 1 in a friendly, informative, and engaging way. Keep responses concise but informative."""
