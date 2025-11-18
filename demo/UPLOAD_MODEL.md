# Upload Your Fine-Tuned Model to Cloud

Upload your own fine-tuned or LoRA model to Hugging Face Hub, then use cloud inference API to run it!

## Why Upload to Cloud?

✅ **Use your own trained model** (not base models)  
✅ **No local GPU needed** - runs on cloud GPUs  
✅ **Free tier available** on Hugging Face  
✅ **Fast inference** - powerful cloud hardware  
✅ **Share with team** - easy access  

## Step 1: Prepare Your Model

Your model should be in one of these formats:

### Full Fine-Tuned Model:
```
models/artifacts/your-model/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json
└── tokenizer_config.json
```

### LoRA Model:
```
models/artifacts/your-lora-model/
├── adapter_config.json
├── adapter_model.bin (or adapter_model.safetensors)
└── (base model will be loaded from HF)
```

## Step 2: Install Hugging Face Hub

```bash
pip install huggingface_hub
```

## Step 3: Upload Your Model

### Option A: Using Python Script (Recommended)

Create `upload_model.py`:

```python
from huggingface_hub import HfApi, login
from pathlib import Path

# Login to Hugging Face
# Get token from: https://huggingface.co/settings/tokens
token = "hf_your_token_here"
login(token=token)

# Initialize API
api = HfApi()

# Your model path
model_path = "models/artifacts/your-best-model"  # Your actual model path

# Your Hugging Face username and model name
repo_id = "your-username/your-f1-model"  # e.g., "johndoe/f1-mistral-7b-finetuned"

# Create repository (if doesn't exist)
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"✅ Repository created: {repo_id}")
except Exception as e:
    print(f"Repository may already exist: {e}")

# Upload model
print(f"📤 Uploading model from {model_path}...")
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["*.git*", "*.DS_Store"]
)

print(f"✅ Model uploaded successfully!")
print(f"🔗 View at: https://huggingface.co/{repo_id}")
print(f"💡 Use this in config.py: REMOTE_MODEL_ID = '{repo_id}'")
```

Run it:
```bash
python upload_model.py
```

### Option B: Using Hugging Face CLI

```bash
# Install CLI
pip install huggingface_hub[cli]

# Login
huggingface-cli login

# Upload model
huggingface-cli upload your-username/your-f1-model models/artifacts/your-best-model
```

### Option C: Using Git (For Large Models)

```bash
# Install git-lfs first
git lfs install

# Clone your repo (create on HF first)
git clone https://huggingface.co/your-username/your-f1-model
cd your-f1-model

# Copy your model files
cp -r ../models/artifacts/your-best-model/* .

# Commit and push
git add .
git commit -m "Upload fine-tuned F1 model"
git push
```

## Step 4: Configure Demo to Use Your Uploaded Model

Edit `demo/config.py`:

```python
# Use your uploaded model via cloud API
USE_REMOTE_API = True
REMOTE_API_TYPE = "huggingface"
REMOTE_API_KEY = "hf_your_token_here"  # Same token you used to upload
REMOTE_MODEL_ID = "your-username/your-f1-model"  # Your uploaded model ID
```

## Step 5: Test Your Model

```bash
cd demo
python app.py
```

Your model will run on Hugging Face's cloud infrastructure!

## Uploading LoRA Models

If you have a LoRA model, you have two options:

### Option 1: Upload LoRA Adapter Only (Smaller)

```python
# Upload just the LoRA adapter
api.upload_folder(
    folder_path="models/artifacts/your-lora-model",
    repo_id="your-username/your-f1-lora",
    repo_type="model"
)
```

Then in config, set:
```python
REMOTE_MODEL_ID = "your-username/your-f1-lora"
```

**Note:** Hugging Face Inference API will automatically load the base model + your LoRA adapter!

### Option 2: Merge LoRA into Base Model (Larger but Faster)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load LoRA
model = PeftModel.from_pretrained(base_model, "models/artifacts/your-lora-model")

# Merge LoRA weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("models/artifacts/merged-model")
tokenizer.save_pretrained("models/artifacts/merged-model")

# Then upload merged model
api.upload_folder(
    folder_path="models/artifacts/merged-model",
    repo_id="your-username/your-f1-merged",
    repo_type="model"
)
```

## Privacy Settings

By default, models are **private**. To make public:

```python
api.update_repo_visibility(
    repo_id="your-username/your-f1-model",
    private=False
)
```

Or set when creating:
```python
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False  # Make public
)
```

## Troubleshooting

**"Model not found" error:**
- Check repo_id is correct: `username/model-name`
- Make sure model is uploaded (check on HF website)
- If private, ensure API token has access

**"Model is loading" message:**
- Free tier: Models spin down after inactivity
- First request takes 30-60 seconds to wake up
- Subsequent requests are fast
- Upgrade to paid tier for always-on models

**Upload fails:**
- Check file size (free tier has limits)
- Ensure all required files are present
- Check internet connection

## Cost

- **Free tier:** Limited requests, models spin down
- **Pro tier:** More requests, faster inference
- **Enterprise:** Custom pricing

## Next Steps

1. Upload your best model
2. Update `config.py` with your model ID
3. Run demo: `python app.py`
4. Your model runs on cloud! 🚀

