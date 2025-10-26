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

3. **Falcon 7B Instruct** (`falcon-7b-instruct/`)
   - Model: `tiiuae/falcon-7b-instruct`
   - Size: ~13.6 GB
   - Status: ✅ Complete and verified
   - Description: Technology Innovation Institute's 7B parameter instruction-tuned model

## Model Usage

These models can be loaded using the utilities in `utils/model_utils.py`:

```python
from utils.model_utils import load_model_and_tokenizer

# Load Mistral 7B
model, tokenizer = load_model_and_tokenizer("models/base/mistral-7b")

# Load Phi-3 Mini
model, tokenizer = load_model_and_tokenizer("models/base/phi-3-mini")

# Load Falcon 7B Instruct
model, tokenizer = load_model_and_tokenizer("models/base/falcon-7b-instruct")
```

## Download Scripts

- `scripts/setup_models.py` - Main download script with progress tracking
- `scripts/verify_models.py` - Verification script to check model integrity
- `scripts/download_models.py` - Alternative download script with more features

## Notes

- Models are stored in the `base/` subdirectory
- Each model includes configuration files, tokenizers, and model weights
- All 3 models are fully downloaded and verified
- All models are compatible with the Hugging Face Transformers library

## Next Steps

1. Add model-specific evaluation wrappers for the F1 evaluation framework
2. Test model performance on F1-specific tasks
3. Fine-tune models on F1 conversational data
4. Integrate models into the evaluation pipeline
