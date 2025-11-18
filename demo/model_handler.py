"""
Model Handler for F1 Demo
Handles model loading and inference with support for fine-tuned and LoRA models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from typing import Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. LoRA models will not be supported. Install with: pip install peft")


class ModelHandler:
    """Handles model loading and text generation with memory optimizations and LoRA support."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        use_quantization: bool = True, 
        quantization_bits: int = 4,
        base_model_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.is_lora = False
        
    def _is_lora_model(self, model_path: str) -> bool:
        """Check if model path contains LoRA adapter."""
        path = Path(model_path)
        # Check for adapter_config.json (PEFT/LoRA indicator)
        if (path / "adapter_config.json").exists():
            return True
        # Check if path name suggests LoRA
        if "lora" in path.name.lower():
            return True
        return False
    
    def _get_base_model_from_lora(self, lora_path: str) -> Optional[str]:
        """Extract base model path from LoRA adapter config."""
        if not PEFT_AVAILABLE:
            return None
        
        try:
            config = PeftConfig.from_pretrained(lora_path)
            return config.base_model_name_or_path
        except Exception as e:
            logger.warning(f"Could not read base model from LoRA config: {e}")
            return None
    
    def load_model(self):
        """Load the model and tokenizer with memory optimizations and LoRA support."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Check if this is a LoRA model
            self.is_lora = self._is_lora_model(self.model_path)
            
            if self.is_lora:
                if not PEFT_AVAILABLE:
                    logger.error("LoRA model detected but PEFT is not installed. Install with: pip install peft")
                    return False
                logger.info("LoRA model detected. Loading base model first, then LoRA adapter...")
                return self._load_lora_model()
            else:
                logger.info("Loading full fine-tuned model...")
                return self._load_full_model()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Try: 1) Use smaller model, 2) Enable quantization, 3) Use CPU mode, 4) Use remote API")
            self.loaded = False
            return False
    
    def _load_full_model(self) -> bool:
        """Load a full fine-tuned model (not LoRA)."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization config if enabled
            quantization_config = None
            if self.use_quantization and self.device == "cuda":
                try:
                    if self.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("Using 4-bit quantization (reduces memory by ~75%)")
                    elif self.quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16
                        )
                        logger.info("Using 8-bit quantization (reduces memory by ~50%)")
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}. Loading without quantization.")
                    quantization_config = None
            
            # Determine dtype
            if quantization_config:
                torch_dtype = None  # Quantization handles dtype
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch_dtype
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move to device if not using device_map or quantization
            if not quantization_config:
                if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
                    self.model = self.model.to(self.device)
                elif self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            logger.info(f"Full fine-tuned model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load full model: {e}")
            return False
    
    def _load_lora_model(self) -> bool:
        """Load a LoRA model (base model + LoRA adapter)."""
        try:
            # Determine base model path
            base_model = self.base_model_path
            if not base_model:
                # Try to get from LoRA config
                base_model = self._get_base_model_from_lora(self.model_path)
                if not base_model:
                    logger.error("Could not determine base model for LoRA. Please set BASE_MODEL_PATH in config.py")
                    return False
            
            logger.info(f"Loading base model: {base_model}")
            
            # Load tokenizer from base model or LoRA path
            tokenizer_path = base_model if os.path.exists(base_model) or "/" in base_model else self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization config if enabled
            quantization_config = None
            if self.use_quantization and self.device == "cuda":
                try:
                    if self.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("Using 4-bit quantization for base model")
                    elif self.quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16
                        )
                        logger.info("Using 8-bit quantization for base model")
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}. Loading without quantization.")
                    quantization_config = None
            
            # Determine dtype
            if quantization_config:
                torch_dtype = None
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load base model
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch_dtype
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                **model_kwargs
            )
            
            logger.info(f"Loading LoRA adapter from: {self.model_path}")
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                self.model_path,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Merge LoRA weights for faster inference (optional)
            # Uncomment if you want to merge LoRA weights into base model
            # self.model = self.model.merge_and_unload()
            
            # Move to device if needed
            if not quantization_config and self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            logger.info(f"LoRA model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response from the model."""
        if not self.loaded:
            return "Error: Model not loaded. Please check the model path in config.py"
        
        try:
            # Format prompt with system prompt if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # For CPU, move inputs to CPU
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Create generation config
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    generation_config=generation_config
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            if len(outputs[0]) > input_length:
                new_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            else:
                response = ""
            
            return response if response else "I'm sorry, I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

Handles model loading and inference with support for fine-tuned and LoRA models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from typing import Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. LoRA models will not be supported. Install with: pip install peft")


class ModelHandler:
    """Handles model loading and text generation with memory optimizations and LoRA support."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        use_quantization: bool = True, 
        quantization_bits: int = 4,
        base_model_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.is_lora = False
        
    def _is_lora_model(self, model_path: str) -> bool:
        """Check if model path contains LoRA adapter."""
        path = Path(model_path)
        # Check for adapter_config.json (PEFT/LoRA indicator)
        if (path / "adapter_config.json").exists():
            return True
        # Check if path name suggests LoRA
        if "lora" in path.name.lower():
            return True
        return False
    
    def _get_base_model_from_lora(self, lora_path: str) -> Optional[str]:
        """Extract base model path from LoRA adapter config."""
        if not PEFT_AVAILABLE:
            return None
        
        try:
            config = PeftConfig.from_pretrained(lora_path)
            return config.base_model_name_or_path
        except Exception as e:
            logger.warning(f"Could not read base model from LoRA config: {e}")
            return None
    
    def load_model(self):
        """Load the model and tokenizer with memory optimizations and LoRA support."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Check if this is a LoRA model
            self.is_lora = self._is_lora_model(self.model_path)
            
            if self.is_lora:
                if not PEFT_AVAILABLE:
                    logger.error("LoRA model detected but PEFT is not installed. Install with: pip install peft")
                    return False
                logger.info("LoRA model detected. Loading base model first, then LoRA adapter...")
                return self._load_lora_model()
            else:
                logger.info("Loading full fine-tuned model...")
                return self._load_full_model()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Try: 1) Use smaller model, 2) Enable quantization, 3) Use CPU mode, 4) Use remote API")
            self.loaded = False
            return False
    
    def _load_full_model(self) -> bool:
        """Load a full fine-tuned model (not LoRA)."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization config if enabled
            quantization_config = None
            if self.use_quantization and self.device == "cuda":
                try:
                    if self.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("Using 4-bit quantization (reduces memory by ~75%)")
                    elif self.quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16
                        )
                        logger.info("Using 8-bit quantization (reduces memory by ~50%)")
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}. Loading without quantization.")
                    quantization_config = None
            
            # Determine dtype
            if quantization_config:
                torch_dtype = None  # Quantization handles dtype
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch_dtype
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move to device if not using device_map or quantization
            if not quantization_config:
                if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
                    self.model = self.model.to(self.device)
                elif self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            logger.info(f"Full fine-tuned model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load full model: {e}")
            return False
    
    def _load_lora_model(self) -> bool:
        """Load a LoRA model (base model + LoRA adapter)."""
        try:
            # Determine base model path
            base_model = self.base_model_path
            if not base_model:
                # Try to get from LoRA config
                base_model = self._get_base_model_from_lora(self.model_path)
                if not base_model:
                    logger.error("Could not determine base model for LoRA. Please set BASE_MODEL_PATH in config.py")
                    return False
            
            logger.info(f"Loading base model: {base_model}")
            
            # Load tokenizer from base model or LoRA path
            tokenizer_path = base_model if os.path.exists(base_model) or "/" in base_model else self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization config if enabled
            quantization_config = None
            if self.use_quantization and self.device == "cuda":
                try:
                    if self.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("Using 4-bit quantization for base model")
                    elif self.quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16
                        )
                        logger.info("Using 8-bit quantization for base model")
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}. Loading without quantization.")
                    quantization_config = None
            
            # Determine dtype
            if quantization_config:
                torch_dtype = None
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load base model
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch_dtype
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                **model_kwargs
            )
            
            logger.info(f"Loading LoRA adapter from: {self.model_path}")
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model_obj,
                self.model_path,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Merge LoRA weights for faster inference (optional)
            # Uncomment if you want to merge LoRA weights into base model
            # self.model = self.model.merge_and_unload()
            
            # Move to device if needed
            if not quantization_config and self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            logger.info(f"LoRA model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response from the model."""
        if not self.loaded:
            return "Error: Model not loaded. Please check the model path in config.py"
        
        try:
            # Format prompt with system prompt if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # For CPU, move inputs to CPU
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Create generation config
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    generation_config=generation_config
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            if len(outputs[0]) > input_length:
                new_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            else:
                response = ""
            
            return response if response else "I'm sorry, I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
