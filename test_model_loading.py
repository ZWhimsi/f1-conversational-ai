#!/usr/bin/env python3
"""
Quick test to verify models can be loaded properly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.model_utils import load_model_and_tokenizer
import torch

def test_model_loading():
    """Test loading each model."""
    models_dir = project_root / "models" / "base"
    
    models_to_test = [
        "mistral-7b",
        "phi-3-mini", 
        "falcon-7b-instruct"
    ]
    
    print("🧪 Testing model loading...")
    print("=" * 50)
    
    for model_name in models_to_test:
        model_path = models_dir / model_name
        print(f"\n🔍 Testing {model_name}...")
        print(f"📁 Path: {model_path}")
        
        try:
            # Test loading with CPU to avoid memory issues
            model, tokenizer = load_model_and_tokenizer(
                str(model_path),
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            print(f"✅ {model_name} loaded successfully!")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Tokenizer type: {type(tokenizer).__name__}")
            
            # Test a simple generation
            test_prompt = "Hello, how are you?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Test generation: {response[:100]}...")
            
            # Clean up memory
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ {model_name} failed to load: {e}")
            return False
    
    print(f"\n🎉 All models loaded successfully!")
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
