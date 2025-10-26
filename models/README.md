# F1 Conversational AI - Models

This directory contains the downloaded language models for the F1 Conversational AI project.

## Downloaded Models

### ✅ Successfully Downloaded Models

1. **Mistral 7B** (`mistral-7b/`)
   - Model: `mistralai/Mistral-7B-v0.1`
   - Size: ~13.1 GB
   - Status: ✅ Complete and verified
   - Description: High-performance 7B parameter model from Mistral AI

2. **Phi-3 Mini** (`phi-3-mini/`)
   - Model: `microsoft/Phi-3-mini-4k-instruct`
   - Size: ~7.2 GB
   - Status: ✅ Complete and verified
   - Description: Microsoft's efficient 3.8B parameter instruction-tuned model

### 🔄 Partially Downloaded Models

3. **Qwen 7B Chat** (`qwen-7b-chat/`)
   - Model: `Qwen/Qwen-7B-Chat`
   - Status: 🔄 Partial download (missing tokenizer files)
   - Description: Alibaba's 7B parameter conversational model

## Model Usage

These models can be loaded using the utilities in `utils/model_utils.py`:

```python
from utils.model_utils import load_model_and_tokenizer

# Load Mistral 7B
model, tokenizer = load_model_and_tokenizer("models/base/mistral-7b")

# Load Phi-3 Mini
model, tokenizer = load_model_and_tokenizer("models/base/phi-3-mini")
```

## Download Scripts

- `scripts/setup_models.py` - Main download script with progress tracking
- `scripts/verify_models.py` - Verification script to check model integrity
- `scripts/download_models.py` - Alternative download script with more features

## Notes

- Models are stored in the `base/` subdirectory
- Each model includes configuration files, tokenizers, and model weights
- The Qwen model download was interrupted and may need to be re-downloaded
- All models are compatible with the Hugging Face Transformers library

## Next Steps

1. Complete the Qwen model download if needed
2. Add model-specific evaluation wrappers
3. Test model performance on F1-specific tasks
4. Fine-tune models on F1 conversational data
