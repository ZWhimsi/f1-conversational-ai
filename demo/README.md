# F1 Website Demo with AI Chatbot

A standalone demo application featuring a replica F1 website with an integrated AI chatbot powered by **YOUR OWN fine-tuned model** running on cloud infrastructure.

## Features

- 🏎️ F1 website replica with modern UI
- 💬 Interactive AI chatbot for F1 questions
- 🚀 Fast API backend
- ☁️ **Use YOUR OWN model via cloud inference** (no local GPU needed!)
- 📱 Responsive design

## Quick Start

### 1. Install Dependencies

```bash
cd demo
pip install -r requirements.txt
```

### 2. Upload Your Fine-Tuned Model to Hugging Face

**This is the key step - upload YOUR model to use cloud inference!**

```bash
python upload_model.py
```

Or see `UPLOAD_MODEL.md` for detailed instructions.

### 3. Configure Demo

Edit `config.py`:

```python
# Use YOUR uploaded model via cloud API
USE_REMOTE_API = True
REMOTE_API_TYPE = "huggingface"
REMOTE_API_KEY = "hf_your_token_here"  # Get from https://huggingface.co/settings/tokens
REMOTE_MODEL_ID = "your-username/your-f1-model"  # YOUR uploaded model!
```

### 4. Run the Demo

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## Workflow: Your Model → Cloud → Demo

```
1. Your Fine-Tuned Model (local)
   ↓
2. Upload to Hugging Face Hub
   python upload_model.py
   ↓
3. Use Cloud Inference API
   (config.py: USE_REMOTE_API = True)
   ↓
4. Demo runs YOUR model on cloud GPUs! 🚀
```

## Project Structure

```
demo/
├── app.py                  # Flask backend API
├── config.py               # Configuration
├── model_handler.py         # Local model handler (if needed)
├── remote_api_handler.py   # Cloud API handler
├── upload_model.py         # Script to upload your model
├── static/                 # CSS, JS, images
│   ├── css/
│   ├── js/
│   └── images/
├── templates/              # HTML templates
│   └── index.html
├── UPLOAD_MODEL.md         # Guide for uploading models
├── CLOUD_SETUP.md          # Cloud inference setup
└── requirements.txt        # Python dependencies
```

## Configuration Options

### Option 1: Use Your Model via Cloud (Recommended!)

**Best for:** Large models, no local GPU, want cloud power

```python
USE_REMOTE_API = True
REMOTE_MODEL_ID = "your-username/your-f1-model"
```

### Option 2: Use Model Locally

**Best for:** Small models, have GPU, want full control

```python
USE_REMOTE_API = False
MODEL_PATH = "models/artifacts/your-model"
USE_QUANTIZATION = True  # Reduces memory by 75%
```

## Benefits of Cloud Inference

✅ **Use YOUR own trained model** (not base models)  
✅ **No local GPU needed** - runs on cloud GPUs  
✅ **Free tier available** (Hugging Face)  
✅ **Fast inference** - powerful cloud hardware  
✅ **Scalable** - handles multiple users  
✅ **Easy sharing** - team can access your model  

## Documentation

- **`UPLOAD_MODEL.md`** - How to upload your fine-tuned model
- **`CLOUD_SETUP.md`** - Cloud inference API setup
- **`MEMORY_OPTIMIZATION.md`** - Local model optimization tips

## Troubleshooting

**Model not found:**
- Check `REMOTE_MODEL_ID` matches your uploaded model
- Verify model is uploaded (check on HF website)
- If private model, ensure API token has access

**Upload fails:**
- Check file size (free tier has limits)
- Ensure all required files are present
- Check internet connection

**Model loading slowly:**
- Free tier: Models spin down after inactivity
- First request takes 30-60 seconds
- Subsequent requests are fast

## Next Steps

1. ✅ Upload your best fine-tuned model
2. ✅ Configure `config.py` with your model ID
3. ✅ Run demo: `python app.py`
4. ✅ Your model runs on cloud! 🎉
