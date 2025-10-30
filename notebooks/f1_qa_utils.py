"""
F1 QA Evaluation Utilities
Reusable functions for loading QA dataset and formatting multiple choice questions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any


def load_qa_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load all QA JSON files from the dataset directory.
    
    Args:
        dataset_path: Path to directory containing qa_1.json to qa_500.json
    
    Returns:
        List of all QA pairs with metadata
    """
    dataset_path = Path(dataset_path)
    all_qa_pairs = []
    
    # Load all files from qa_1.json to qa_500.json
    for i in range(1, 501):
        file_path = dataset_path / f"qa_{i}.json"
        
        if not file_path.exists():
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract QA pairs
            article_number = data.get('article_number', i)
            source = data.get('source', '')
            title = data.get('title', '')
            
            for qa_pair in data.get('qa_pairs', []):
                qa_pair['article_number'] = article_number
                qa_pair['source'] = source
                qa_pair['title'] = title
                all_qa_pairs.append(qa_pair)
                
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_qa_pairs)} QA pairs from {dataset_path}")
    return all_qa_pairs


def format_multiple_choice(question: str, correct_answer: str, wrong_options: List[str]) -> Tuple[str, str]:
    """
    Format a question as multiple choice with randomized options.
    
    Args:
        question: The question text
        correct_answer: The correct answer
        wrong_options: List of 3 wrong answer options
    
    Returns:
        Tuple of (formatted_prompt, correct_letter)
    """
    # Combine all options
    all_options = [correct_answer] + wrong_options
    
    # Shuffle to randomize order
    shuffled_options = all_options.copy()
    random.shuffle(shuffled_options)
    
    # Find which letter corresponds to correct answer
    correct_index = shuffled_options.index(correct_answer)
    correct_letter = chr(65 + correct_index)  # A=65, B=66, C=67, D=68
    
    # Create formatted prompt with structured output
    prompt = f"""You are an AI assistant specializing in Formula 1. Answer the following multiple choice question using the exact format specified below.

Question: {question}

A) {shuffled_options[0]}
B) {shuffled_options[1]}
C) {shuffled_options[2]}
D) {shuffled_options[3]}

Please respond in this exact format:
Answer: [A/B/C/D]
Justification: [Brief 2-3 line explanation of your reasoning]

Answer:"""
    
    return prompt, correct_letter


def extract_answer_letter(response: str) -> str:
    """
    Extract the answer letter (A, B, C, or D) from model response.
    Handles both structured format (Answer: A) and legacy formats.
    
    Args:
        response: Model's response text
    
    Returns:
        Extracted letter or empty string if not found
    """
    response_upper = response.upper().strip()
    
    # First, try to find structured format: "Answer: A" or "Answer:A"
    import re
    answer_match = re.search(r'ANSWER:\s*([ABCD])', response_upper)
    if answer_match:
        return answer_match.group(1)
    
    # Fallback: Look for single letter A, B, C, or D in various formats
    for letter in ['A', 'B', 'C', 'D']:
        # Check if letter appears as standalone or followed by punctuation
        if response_upper.startswith(letter) or \
           f" {letter}" in response_upper or \
           f"({letter})" in response_upper or \
           f"{letter}." in response_upper:
            return letter
    
    # If no letter found, return empty string
    return ""


def extract_justification(response: str) -> str:
    """
    Extract the justification from structured model response.
    
    Args:
        response: Model's response text
    
    Returns:
        Extracted justification or empty string if not found
    """
    import re
    
    # Look for "Justification:" followed by text
    justification_match = re.search(r'Justification:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
    if justification_match:
        return justification_match.group(1).strip()
    
    return ""


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON response from OpenAI API, handling various formats.
    
    Args:
        response_text: Response text from OpenAI
    
    Returns:
        Parsed JSON dictionary
    """
    import re
    
    # Try to extract JSON from response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # Fallback: return structured response
    return {
        "correctness": "No",
        "quality_score": 1,
        "reasoning": "Failed to parse response"
    }
