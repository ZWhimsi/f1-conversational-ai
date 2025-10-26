#!/usr/bin/env python3
"""
Response Quality Evaluator for F1 Conversational AI
Checks if model responses are correct and evaluates quality metrics.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for a model response."""
    accuracy: float
    relevance: float
    completeness: float
    factual_correctness: float
    overall_score: float
    issues: List[str]

class F1ResponseQualityEvaluator:
    """Evaluates the quality of F1 model responses."""
    
    def __init__(self):
        self.f1_keywords = [
            'formula 1', 'f1', 'grand prix', 'race', 'driver', 'team', 'championship',
            'pole position', 'podium', 'lap', 'overtake', 'drs', 'safety car',
            'ferrari', 'mercedes', 'red bull', 'mclaren', 'aston martin', 'alpine',
            'hamilton', 'verstappen', 'leclerc', 'norris', 'russell', 'sainz'
        ]
    
    def evaluate_response(self, question: str, ground_truth: str, model_response: str, notes: str = "") -> QualityMetrics:
        """Evaluate a single model response."""
        issues = []
        
        # 1. Accuracy - Check if response contains key information from ground truth
        accuracy = self._calculate_accuracy(ground_truth, model_response)
        
        # 2. Relevance - Check if response is relevant to F1 and the question
        relevance = self._calculate_relevance(question, model_response)
        
        # 3. Completeness - Check if response covers the main points
        completeness = self._calculate_completeness(ground_truth, model_response)
        
        # 4. Factual Correctness - Check for obvious factual errors
        factual_correctness = self._calculate_factual_correctness(model_response, notes)
        
        # Calculate overall score
        overall_score = (accuracy + relevance + completeness + factual_correctness) / 4
        
        # Identify issues
        if accuracy < 0.5:
            issues.append("Low accuracy - missing key information")
        if relevance < 0.5:
            issues.append("Low relevance - not focused on F1 or question")
        if completeness < 0.5:
            issues.append("Incomplete response - missing important details")
        if factual_correctness < 0.5:
            issues.append("Factual errors detected")
        
        return QualityMetrics(
            accuracy=accuracy,
            relevance=relevance,
            completeness=completeness,
            factual_correctness=factual_correctness,
            overall_score=overall_score,
            issues=issues
        )
    
    def _calculate_accuracy(self, ground_truth: str, response: str) -> float:
        """Calculate accuracy based on key information overlap."""
        gt_words = set(ground_truth.lower().split())
        resp_words = set(response.lower().split())
        
        if not gt_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(gt_words.intersection(resp_words))
        union = len(gt_words.union(resp_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_relevance(self, question: str, response: str) -> float:
        """Calculate relevance to F1 and the specific question."""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Check F1 keyword presence
        f1_score = sum(1 for keyword in self.f1_keywords if keyword in response.lower()) / len(self.f1_keywords)
        
        # Check question word overlap
        question_overlap = len(question_words.intersection(response_words)) / len(question_words) if question_words else 0
        
        return (f1_score * 0.6 + question_overlap * 0.4)
    
    def _calculate_completeness(self, ground_truth: str, response: str) -> float:
        """Calculate completeness based on coverage of ground truth points."""
        # Simple heuristic: compare response length to ground truth
        gt_length = len(ground_truth.split())
        resp_length = len(response.split())
        
        if gt_length == 0:
            return 0.0
        
        # Ideal response should be 50-150% of ground truth length
        ratio = resp_length / gt_length
        
        if 0.5 <= ratio <= 1.5:
            return 1.0
        elif 0.3 <= ratio <= 2.0:
            return 0.8
        elif 0.2 <= ratio <= 3.0:
            return 0.6
        else:
            return 0.4
    
    def _calculate_factual_correctness(self, response: str, notes: str) -> float:
        """Calculate factual correctness based on common F1 knowledge."""
        score = 1.0
        response_lower = response.lower()
        
        # Check for common F1 facts
        f1_facts = [
            ('formula 1', 'f1'),
            ('grand prix', 'gp'),
            ('world championship', 'championship'),
        ]
        
        # Check for contradictory statements
        contradictions = [
            ('lewis hamilton', 'ferrari'),  # Hamilton drives for Mercedes, not Ferrari
            ('max verstappen', 'mercedes'),  # Verstappen drives for Red Bull, not Mercedes
        ]
        
        for fact1, fact2 in f1_facts:
            if fact1 in response_lower and fact2 in response_lower:
                score = min(score, 0.9)  # Slight penalty for redundancy
        
        for wrong1, wrong2 in contradictions:
            if wrong1 in response_lower and wrong2 in response_lower:
                score = min(score, 0.3)  # Major penalty for contradictions
        
        # Check for nonsensical statements
        nonsense_indicators = ['impossible', 'never happened', 'completely wrong']
        for indicator in nonsense_indicators:
            if indicator in response_lower:
                score = min(score, 0.2)
        
        return score
    
    def evaluate_baseline_results(self, results_file: Path) -> Dict[str, Any]:
        """Evaluate all responses in a baseline results file."""
        logger.info(f"📊 Evaluating results from {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        evaluation_summary = {
            'metadata': results.get('metadata', {}),
            'model_performance': {},
            'overall_statistics': {},
            'detailed_evaluations': {}
        }
        
        # Initialize model performance tracking
        models = list(results['responses']['question_1']['model_responses'].keys())
        for model in models:
            evaluation_summary['model_performance'][model] = {
                'total_questions': 0,
                'average_accuracy': 0,
                'average_relevance': 0,
                'average_completeness': 0,
                'average_factual_correctness': 0,
                'average_overall_score': 0,
                'total_issues': 0,
                'common_issues': []
            }
        
        # Evaluate each question
        for question_id, question_data in results['responses'].items():
            question = question_data['question']
            ground_truth = question_data['ground_truth']
            notes = question_data.get('notes', '')
            
            logger.info(f"🔍 Evaluating {question_id}")
            
            for model_name, model_data in question_data['model_responses'].items():
                response = model_data['response']
                
                # Evaluate this response
                metrics = self.evaluate_response(question, ground_truth, response, notes)
                
                # Store detailed evaluation
                if question_id not in evaluation_summary['detailed_evaluations']:
                    evaluation_summary['detailed_evaluations'][question_id] = {
                        'question': question,
                        'ground_truth': ground_truth,
                        'model_evaluations': {}
                    }
                
                evaluation_summary['detailed_evaluations'][question_id]['model_evaluations'][model_name] = {
                    'response': response,
                    'metrics': {
                        'accuracy': metrics.accuracy,
                        'relevance': metrics.relevance,
                        'completeness': metrics.completeness,
                        'factual_correctness': metrics.factual_correctness,
                        'overall_score': metrics.overall_score,
                        'issues': metrics.issues
                    }
                }
                
                # Update model performance
                perf = evaluation_summary['model_performance'][model_name]
                perf['total_questions'] += 1
                perf['average_accuracy'] += metrics.accuracy
                perf['average_relevance'] += metrics.relevance
                perf['average_completeness'] += metrics.completeness
                perf['average_factual_correctness'] += metrics.factual_correctness
                perf['average_overall_score'] += metrics.overall_score
                perf['total_issues'] += len(metrics.issues)
                
                # Track common issues
                for issue in metrics.issues:
                    if issue not in perf['common_issues']:
                        perf['common_issues'].append(issue)
        
        # Calculate averages
        for model_name, perf in evaluation_summary['model_performance'].items():
            total = perf['total_questions']
            if total > 0:
                perf['average_accuracy'] /= total
                perf['average_relevance'] /= total
                perf['average_completeness'] /= total
                perf['average_factual_correctness'] /= total
                perf['average_overall_score'] /= total
        
        # Calculate overall statistics
        all_scores = []
        for model_perf in evaluation_summary['model_performance'].values():
            all_scores.append(model_perf['average_overall_score'])
        
        evaluation_summary['overall_statistics'] = {
            'total_questions_evaluated': len(results['responses']),
            'models_evaluated': len(models),
            'average_score_across_models': sum(all_scores) / len(all_scores) if all_scores else 0,
            'best_performing_model': max(evaluation_summary['model_performance'].items(), 
                                      key=lambda x: x[1]['average_overall_score'])[0] if evaluation_summary['model_performance'] else None
        }
        
        return evaluation_summary
    
    def save_evaluation_report(self, evaluation: Dict[str, Any], output_path: Path) -> None:
        """Save evaluation report to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Evaluation report saved to {output_path}")

def main():
    """Main evaluation function."""
    logger.info("🔍 Starting F1 Response Quality Evaluation")
    logger.info("=" * 60)
    
    # Find the most recent baseline results file
    results_dir = Path(__file__).parent.parent / "results" / "baseline_evaluation"
    
    if not results_dir.exists():
        logger.error("❌ No baseline evaluation results found!")
        logger.info("💡 Run baseline_evaluation.py first to generate results")
        return False
    
    # Find the most recent results file
    result_files = list(results_dir.glob("baseline_evaluation_*.json"))
    if not result_files:
        logger.error("❌ No baseline evaluation files found!")
        return False
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"📁 Using results file: {latest_file}")
    
    # Initialize evaluator
    evaluator = F1ResponseQualityEvaluator()
    
    # Evaluate results
    evaluation = evaluator.evaluate_baseline_results(latest_file)
    
    # Save evaluation report
    output_dir = Path(__file__).parent.parent / "results" / "quality_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"quality_evaluation_{int(time.time())}.json"
    evaluator.save_evaluation_report(evaluation, output_file)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("📊 QUALITY EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    for model_name, perf in evaluation['model_performance'].items():
        logger.info(f"🤖 {model_name}:")
        logger.info(f"   Overall Score: {perf['average_overall_score']:.2f}")
        logger.info(f"   Accuracy: {perf['average_accuracy']:.2f}")
        logger.info(f"   Relevance: {perf['average_relevance']:.2f}")
        logger.info(f"   Completeness: {perf['average_completeness']:.2f}")
        logger.info(f"   Factual Correctness: {perf['average_factual_correctness']:.2f}")
        logger.info(f"   Total Issues: {perf['total_issues']}")
        if perf['common_issues']:
            logger.info(f"   Common Issues: {', '.join(perf['common_issues'])}")
        logger.info("")
    
    logger.info(f"🏆 Best Model: {evaluation['overall_statistics']['best_performing_model']}")
    logger.info(f"📈 Average Score: {evaluation['overall_statistics']['average_score_across_models']:.2f}")
    logger.info(f"💾 Report saved to: {output_file}")
    logger.info("🎉 Quality evaluation completed!")
    
    return True

if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)
