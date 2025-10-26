#!/usr/bin/env python3
"""
Baseline Model Evaluation Script for F1 Conversational AI
Passes the evaluation dataset to all 3 models and generates responses.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1ModelEvaluator:
    """Evaluator for F1 conversational AI models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        
    def load_model(self, model_name: str, model_path: Path) -> bool:
        """Load a single model and tokenizer."""
        try:
            logger.info(f"🔄 Loading {model_name}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with CPU to avoid memory issues
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            
            self.models[model_name] = {
                'model': model,
                'tokenizer': tokenizer
            }
            
            logger.info(f"✅ {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return False
    
    def load_all_models(self) -> bool:
        """Load all available models."""
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        success_count = 0
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            if self.load_model(model_name, model_dir):
                success_count += 1
        
        logger.info(f"📊 Loaded {success_count}/{len(model_dirs)} models")
        return success_count > 0
    
    def generate_response(self, model_name: str, prompt: str, max_length: int = 200) -> str:
        """Generate response from a model."""
        try:
            model_data = self.models[model_name]
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response with {model_name}: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_models(self, questions: List[str], answers: List[str], notes: List[str]) -> Dict[str, Any]:
        """Evaluate all models on the dataset."""
        logger.info(f"🚀 Starting evaluation on {len(questions)} questions...")
        
        results = {
            'metadata': {
                'total_questions': len(questions),
                'models_evaluated': list(self.models.keys()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'responses': {}
        }
        
        for i, (question, answer, note) in enumerate(zip(questions, answers, notes)):
            logger.info(f"📝 Processing question {i+1}/{len(questions)}")
            
            question_results = {
                'question': question,
                'ground_truth': answer,
                'notes': note,
                'model_responses': {}
            }
            
            for model_name in self.models.keys():
                logger.info(f"  🤖 Generating response with {model_name}...")
                
                # Create prompt for F1 context
                prompt = f"""You are an AI assistant specializing in Formula 1. Answer the following question about F1:

Question: {question}

Answer:"""
                
                response = self.generate_response(model_name, prompt)
                
                question_results['model_responses'][model_name] = {
                    'response': response,
                    'prompt_used': prompt
                }
                
                logger.info(f"    ✅ {model_name} response generated")
            
            results['responses'][f'question_{i+1}'] = question_results
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save evaluation results to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def load_evaluation_data() -> tuple:
    """Load questions, answers, and notes from the evaluation framework."""
    eval_dir = project_root / "scripts" / "eval_framework" / "eval_set"
    
    # Load small dataset for testing
    questions_file = eval_dir / "questions_small.txt"
    answers_file = eval_dir / "answers_small.txt"
    notes_file = eval_dir / "notes_small.txt"
    
    def load_lines(file_path: Path) -> List[str]:
        if not file_path.exists():
            return []
        return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    questions = load_lines(questions_file)
    answers = load_lines(answers_file)
    notes = load_lines(notes_file)
    
    logger.info(f"📚 Loaded {len(questions)} questions, {len(answers)} answers, {len(notes)} notes")
    return questions, answers, notes

def main():
    """Main evaluation function."""
    logger.info("🏁 Starting F1 Baseline Model Evaluation")
    logger.info("=" * 60)
    
    # Set up paths
    models_dir = project_root / "models" / "base"
    output_dir = project_root / "results" / "baseline_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    questions, answers, notes = load_evaluation_data()
    
    if not questions:
        logger.error("❌ No evaluation data found!")
        return False
    
    # Initialize evaluator
    evaluator = F1ModelEvaluator(models_dir)
    
    # Load all models
    if not evaluator.load_all_models():
        logger.error("❌ No models loaded successfully!")
        return False
    
    # Run evaluation
    results = evaluator.evaluate_models(questions, answers, notes)
    
    # Save results
    output_file = output_dir / f"baseline_evaluation_{int(time.time())}.json"
    evaluator.save_results(results, output_file)
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Questions processed: {len(questions)}")
    logger.info(f"✅ Models evaluated: {len(evaluator.models)}")
    logger.info(f"✅ Results saved to: {output_file}")
    logger.info("🎉 Baseline evaluation completed!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
