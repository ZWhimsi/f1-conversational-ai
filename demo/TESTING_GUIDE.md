# Testing Guide for F1 Demo

## What's Ready

✅ **Complete demo application** with:
- Flask backend API
- F1 website replica with chatbot UI
- Model handler with quantization support
- Cloud API support (Hugging Face, Azure, etc.)
- Memory optimization options

## What Needs to Be Tested

### 1. **Install Dependencies** (First Time Setup)

```bash
cd demo
pip install -r requirements.txt
```

**Expected:** All packages install successfully

**If errors:**
- `bitsandbytes` issues on Windows → Use CPU mode or cloud API
- Missing packages → Check Python version (3.8+)

---

### 2. **Choose Your Model Configuration**

You have **3 options**:

#### Option A: Use Your Fine-Tuned Model (Local)
Edit `config.py`:
```python
MODEL_PATH = "models/artifacts/your-best-model"  # Your fine-tuned model path
USE_REMOTE_API = False
USE_QUANTIZATION = True  # Reduces memory by 75%
```

#### Option B: Use Cloud API (No Local Model - Recommended!)
Edit `config.py`:
```python
USE_REMOTE_API = True
REMOTE_API_TYPE = "huggingface"
REMOTE_API_KEY = "hf_your_token_here"  # Get from https://huggingface.co/settings/tokens
REMOTE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
```

#### Option C: Use Base Model (For Testing)
Edit `config.py`:
```python
MODEL_PATH = "mistralai/Mistral-7B-v0.1"  # Or any Hugging Face model
USE_REMOTE_API = False
USE_QUANTIZATION = True
```

---

### 3. **Test Model Loading**

Run the app:
```bash
python app.py
```

**Expected Output:**
```
INFO: Starting F1 Demo Application...
INFO: Loading model from: [your model path]
INFO: Device: cuda, Quantization: True (4-bit)
INFO: Using 4-bit quantization (reduces memory by ~75%)
INFO: Model loaded successfully on cuda
INFO: Model loaded successfully!
 * Running on http://0.0.0.0:5000
```

**If errors:**
- **Out of Memory** → Enable quantization or use cloud API
- **Model not found** → Check MODEL_PATH in config.py
- **CUDA errors** → Set `USE_GPU=false` in environment or use CPU

---

### 4. **Test Web Interface**

1. Open browser: `http://localhost:5000`
2. You should see:
   - F1 website replica
   - Chatbot widget (bottom right or sidebar)
   - F1-themed styling

**Expected:** Page loads without errors

---

### 5. **Test Chatbot Functionality**

1. Click on chatbot icon
2. Type a test question: "Who won the 2023 F1 championship?"
3. Click send

**Expected:**
- Message appears in chat
- Loading indicator shows
- Response appears from model
- Response is F1-related and coherent

**If errors:**
- No response → Check model loaded successfully
- Error message → Check console logs
- Slow response → Normal for large models, or use cloud API

---

### 6. **Test API Endpoints**

#### Health Check:
```bash
curl http://localhost:5000/api/health
```

**Expected:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### Chat API:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Formula 1?"}'
```

**Expected:**
```json
{
  "response": "[Model's response about F1]",
  "status": "success"
}
```

---

### 7. **Test Different Configurations**

#### Test CPU Mode (if no GPU):
```bash
set USE_GPU=false
python app.py
```

#### Test Cloud API (if model too large):
Edit `config.py`:
```python
USE_REMOTE_API = True
REMOTE_API_KEY = "your_hf_token"
```

#### Test Different Models:
Change `MODEL_PATH` in `config.py` and restart

---

## Troubleshooting Checklist

- [ ] Dependencies installed? (`pip install -r requirements.txt`)
- [ ] Model path correct? (Check `config.py`)
- [ ] Model exists? (Verify path exists)
- [ ] GPU available? (Check `nvidia-smi` or use CPU mode)
- [ ] Port 5000 free? (Change PORT in config.py if needed)
- [ ] Browser console errors? (Check F12 developer tools)
- [ ] Python version 3.8+? (`python --version`)

---

## Quick Test Commands

```bash
# 1. Check if demo folder exists
cd demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test with cloud API (easiest, no model needed)
# Edit config.py: USE_REMOTE_API = True, set REMOTE_API_KEY
python app.py

# 4. Open browser
# http://localhost:5000

# 5. Test API
curl http://localhost:5000/api/health
```

---

## Next Steps After Testing

1. **If everything works:**
   - Deploy to production
   - Share with team
   - Customize UI/UX

2. **If model too large:**
   - Use cloud API (Hugging Face free tier)
   - Or use quantization (already enabled)
   - Or use smaller model

3. **If errors:**
   - Check logs in console
   - See `MEMORY_OPTIMIZATION.md` for memory issues
   - See `CLOUD_SETUP.md` for cloud API setup

---

## Success Criteria

✅ App starts without errors  
✅ Model loads successfully  
✅ Web page displays correctly  
✅ Chatbot responds to questions  
✅ API endpoints work  
✅ Responses are relevant to F1  

If all checked, demo is ready! 🎉

