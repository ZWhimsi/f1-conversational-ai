"""
Upload your fine-tuned model to Hugging Face Hub
Then use it via cloud inference API!
"""

from huggingface_hub import HfApi, login
from pathlib import Path
import os

def upload_model(
    model_path: str,
    repo_id: str,
    hf_token: str,
    private: bool = True
):
    """
    Upload your fine-tuned model to Hugging Face Hub.
    
    Args:
        model_path: Local path to your model (e.g., "models/artifacts/your-model")
        repo_id: Hugging Face repo ID (e.g., "your-username/your-f1-model")
        hf_token: Your Hugging Face token (get from https://huggingface.co/settings/tokens)
        private: Whether to make model private (default: True)
    """
    print("🔐 Logging in to Hugging Face...")
    login(token=hf_token)
    
    # Initialize API
    api = HfApi()
    
    # Check if model path exists
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Error: Model path not found: {model_path}")
        print(f"   Please check the path and try again.")
        return False
    
    print(f"📁 Model directory: {model_dir}")
    print(f"📦 Files to upload:")
    for file in model_dir.rglob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.relative_to(model_dir)} ({size_mb:.2f} MB)")
    
    # Create repository
    print(f"\n📝 Creating repository: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")
    
    # Upload model
    print(f"\n📤 Uploading model...")
    print(f"   This may take a while depending on model size...")
    
    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.git*", "*.DS_Store", "__pycache__", "*.pyc"]
        )
        print(f"\n✅ Model uploaded successfully!")
        print(f"\n🔗 View your model at: https://huggingface.co/{repo_id}")
        print(f"\n💡 Next steps:")
        print(f"   1. Update demo/config.py:")
        print(f"      USE_REMOTE_API = True")
        print(f"      REMOTE_API_KEY = \"{hf_token}\"")
        print(f"      REMOTE_MODEL_ID = \"{repo_id}\"")
        print(f"   2. Run: cd demo && python app.py")
        print(f"   3. Your model will run on Hugging Face cloud! 🚀")
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Upload Your Fine-Tuned Model to Hugging Face Hub")
    print("=" * 60)
    print()
    
    # Get inputs
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = input("Enter model path (e.g., models/artifacts/your-model): ").strip()
    
    if len(sys.argv) > 2:
        repo_id = sys.argv[2]
    else:
        repo_id = input("Enter Hugging Face repo ID (e.g., username/model-name): ").strip()
    
    if len(sys.argv) > 3:
        hf_token = sys.argv[3]
    else:
        hf_token = input("Enter Hugging Face token (get from https://huggingface.co/settings/tokens): ").strip()
        if not hf_token:
            # Try environment variable
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY", "")
    
    if not hf_token:
        print("❌ Error: Hugging Face token is required!")
        print("   Get one from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Upload
    success = upload_model(
        model_path=model_path,
        repo_id=repo_id,
        hf_token=hf_token,
        private=True  # Set to False to make public
    )
    
    if success:
        print("\n🎉 Done! Your model is now on Hugging Face Hub.")
    else:
        print("\n❌ Upload failed. Please check the errors above.")
        sys.exit(1)

