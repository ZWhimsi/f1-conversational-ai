# Memory Optimization Guide

If your model is too large for your local PC, here are several solutions:

## Option 1: Use Quantization (Recommended - Easiest)

**4-bit quantization reduces memory by ~75%!**

The demo is already configured to use 4-bit quantization by default. This should work for most cases.

### If you still have issues:

1. **Force CPU mode** (slower but uses less GPU memory):
   ```bash
   set USE_GPU=false
   python app.py
   ```

2. **Use 8-bit instead of 4-bit**:
   Edit `config.py`:
   ```python
   QUANTIZATION_BITS = 8  # Instead of 4
   ```

## Option 2: Use a Smaller Model

Edit `config.py` to use a smaller model:

```python
# Smaller models (recommended for local):
MODEL_PATH = "microsoft/phi-2"  # 2.7B parameters
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B parameters
MODEL_PATH = "HuggingFaceH4/zephyr-7b-beta"  # 7B but optimized

# Or use your fine-tuned smaller model
MODEL_PATH = "models/base/phi-3-mini"
```

## Option 3: Use Remote API (No Local Model Needed)

Use OpenAI, Anthropic, or your own API server instead of loading the model locally.

### Setup OpenAI API:

1. Get API key from https://platform.openai.com
2. Edit `config.py`:
   ```python
   USE_REMOTE_API = True
   REMOTE_API_KEY = "sk-your-openai-key"
   ```

3. Or set environment variable:
   ```bash
   set USE_REMOTE_API=true
   set OPENAI_API_KEY=sk-your-key
   python app.py
   ```

### Setup Custom API:

If you have a model running on another server:

```python
USE_REMOTE_API = True
REMOTE_API_URL = "http://your-server:8000/api/generate"
```

## Option 4: Use Model Offloading

For very large models, you can offload layers to CPU:

The demo already uses `device_map="auto"` which automatically handles this.

## Option 5: Use Cloud GPU

Run the demo on:
- Google Colab (free GPU)
- AWS/GCP/Azure (pay-per-use)
- Hugging Face Spaces (free tier)

## Quick Start Recommendations

**For most users:**
- Keep default settings (4-bit quantization enabled)
- Use a 7B model or smaller
- Ensure you have at least 8GB GPU memory or 16GB RAM

**If you have limited resources:**
- Use CPU mode: `set USE_GPU=false`
- Use a smaller model (Phi-2, TinyLlama)
- Or use remote API

**If you have good hardware:**
- Disable quantization for faster inference
- Use larger models
- Keep GPU mode enabled

## Troubleshooting

**Out of Memory Error:**
1. Enable quantization (already on by default)
2. Switch to CPU mode
3. Use smaller model
4. Use remote API

**Model loads but is very slow:**
- This is normal on CPU
- Consider using remote API
- Or use a smaller quantized model

**Can't install bitsandbytes:**
- Only needed for quantization on GPU
- If using CPU, quantization is not needed
- Set `USE_QUANTIZATION = False` in config.py


If your model is too large for your local PC, here are several solutions:

## Option 1: Use Quantization (Recommended - Easiest)

**4-bit quantization reduces memory by ~75%!**

The demo is already configured to use 4-bit quantization by default. This should work for most cases.

### If you still have issues:

1. **Force CPU mode** (slower but uses less GPU memory):
   ```bash
   set USE_GPU=false
   python app.py
   ```

2. **Use 8-bit instead of 4-bit**:
   Edit `config.py`:
   ```python
   QUANTIZATION_BITS = 8  # Instead of 4
   ```

## Option 2: Use a Smaller Model

Edit `config.py` to use a smaller model:

```python
# Smaller models (recommended for local):
MODEL_PATH = "microsoft/phi-2"  # 2.7B parameters
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B parameters
MODEL_PATH = "HuggingFaceH4/zephyr-7b-beta"  # 7B but optimized

# Or use your fine-tuned smaller model
MODEL_PATH = "models/base/phi-3-mini"
```

## Option 3: Use Remote API (No Local Model Needed)

Use OpenAI, Anthropic, or your own API server instead of loading the model locally.

### Setup OpenAI API:

1. Get API key from https://platform.openai.com
2. Edit `config.py`:
   ```python
   USE_REMOTE_API = True
   REMOTE_API_KEY = "sk-your-openai-key"
   ```

3. Or set environment variable:
   ```bash
   set USE_REMOTE_API=true
   set OPENAI_API_KEY=sk-your-key
   python app.py
   ```

### Setup Custom API:

If you have a model running on another server:

```python
USE_REMOTE_API = True
REMOTE_API_URL = "http://your-server:8000/api/generate"
```

## Option 4: Use Model Offloading

For very large models, you can offload layers to CPU:

The demo already uses `device_map="auto"` which automatically handles this.

## Option 5: Use Cloud GPU

Run the demo on:
- Google Colab (free GPU)
- AWS/GCP/Azure (pay-per-use)
- Hugging Face Spaces (free tier)

## Quick Start Recommendations

**For most users:**
- Keep default settings (4-bit quantization enabled)
- Use a 7B model or smaller
- Ensure you have at least 8GB GPU memory or 16GB RAM

**If you have limited resources:**
- Use CPU mode: `set USE_GPU=false`
- Use a smaller model (Phi-2, TinyLlama)
- Or use remote API

**If you have good hardware:**
- Disable quantization for faster inference
- Use larger models
- Keep GPU mode enabled

## Troubleshooting

**Out of Memory Error:**
1. Enable quantization (already on by default)
2. Switch to CPU mode
3. Use smaller model
4. Use remote API

**Model loads but is very slow:**
- This is normal on CPU
- Consider using remote API
- Or use a smaller quantized model

**Can't install bitsandbytes:**
- Only needed for quantization on GPU
- If using CPU, quantization is not needed
- Set `USE_QUANTIZATION = False` in config.py

