# Cloud Inference Setup Guide

**Upload YOUR OWN fine-tuned model to Hugging Face Hub, then use cloud inference API!**

This way you:
- ✅ Use **your own trained model** (not base models)
- ✅ Get **cloud GPU power** (no local hardware needed)
- ✅ **Free tier available** on Hugging Face
- ✅ Fast inference on powerful cloud infrastructure

## Step 1: Upload Your Fine-Tuned Model

**First, upload your own model to Hugging Face Hub!**

See `UPLOAD_MODEL.md` for detailed instructions, or use the provided script:

```bash
python upload_model.py
```

Or manually:
```python
from huggingface_hub import HfApi, login

login(token="hf_your_token")
api = HfApi()
api.upload_folder(
    folder_path="models/artifacts/your-best-model",
    repo_id="your-username/your-f1-model"
)
```

## Step 2: Use Your Uploaded Model via Cloud API

**Now use your uploaded model with cloud inference!**

1. **Get a free API key:**
   - Go to https://huggingface.co/settings/tokens
   - Create a token (free account works!)

2. **Edit `config.py`:**
   ```python
   USE_REMOTE_API = True
   REMOTE_API_TYPE = "huggingface"
   REMOTE_API_KEY = "hf_your_token_here"
   REMOTE_MODEL_ID = "your-username/your-f1-model"  # YOUR uploaded model!
   ```

3. **Or use environment variables:**
   ```bash
   set USE_REMOTE_API=true
   set REMOTE_API_TYPE=huggingface
   set HF_API_KEY=hf_your_token_here
   set REMOTE_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.1
   python app.py
   ```

**That's it!** The model runs on Hugging Face's servers.

### Your Own Models:
- Use your uploaded fine-tuned model: `your-username/your-f1-model`
- Use your LoRA model: `your-username/your-f1-lora`
- Works with any model you upload to Hugging Face!

## Option 2: Azure OpenAI / Azure ML

### Setup Azure OpenAI:

1. Create Azure OpenAI resource
2. Deploy Mistral 7B (or use Azure's models)
3. Get endpoint and API key

**Edit `config.py`:**
```python
USE_REMOTE_API = True
REMOTE_API_TYPE = "azure"
REMOTE_API_URL = "https://your-resource.openai.azure.com/openai/deployments/your-deployment/completions"
REMOTE_API_KEY = "your-azure-key"
```

## Option 3: Google Cloud Vertex AI

### Setup:

1. Create Vertex AI endpoint
2. Deploy model
3. Get endpoint URL and credentials

**Edit `config.py`:**
```python
USE_REMOTE_API = True
REMOTE_API_TYPE = "custom"
REMOTE_API_URL = "https://your-region-vertex-ai.googleapis.com/v1/..."
REMOTE_API_KEY = "your-gcp-key"
```

## Option 4: Custom Inference Server

If you have your own server running the model:

**Edit `config.py`:**
```python
USE_REMOTE_API = True
REMOTE_API_TYPE = "custom"
REMOTE_API_URL = "http://your-server:8000/generate"
REMOTE_API_KEY = "optional-auth-key"
```

## Quick Start (Hugging Face - Easiest!)

1. Get free token: https://huggingface.co/settings/tokens
2. Edit `config.py`:
   ```python
   USE_REMOTE_API = True
   REMOTE_API_TYPE = "huggingface"
   REMOTE_API_KEY = "hf_your_token"
   REMOTE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
   ```
3. Run: `python app.py`

**No model download, no GPU needed, works on any PC!**

## Benefits

✅ **No local model loading** - runs in cloud  
✅ **No GPU needed** - works on any computer  
✅ **Fast inference** - cloud GPUs are powerful  
✅ **Free tier available** (Hugging Face)  
✅ **Scalable** - handles multiple users  
✅ **Always up-to-date** - uses latest model versions  

## Cost Comparison

- **Hugging Face**: Free tier available, then pay-per-use
- **Azure**: Pay per token/request
- **Google Cloud**: Pay per request
- **Local**: Free but needs GPU/RAM

## Troubleshooting

**"Model is loading" message:**
- Hugging Face free tier: Models spin down after inactivity
- First request may take 30-60 seconds to wake up
- Subsequent requests are fast!

**API key errors:**
- Check token is correct
- For Hugging Face: Make sure token has "read" permissions

**Rate limits:**
- Free tier has rate limits
- Upgrade to paid tier for higher limits
- Or use your own cloud deployment

