#!/usr/bin/env python3
"""
Complete Baseline Evaluation Runner
Runs both baseline evaluation and quality assessment.
"""

import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status."""
    logger.info(f"🚀 {description}")
    logger.info("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, check=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        logger.info(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with error: {e}")
        return False

def main():
    """Run complete baseline evaluation pipeline."""
    logger.info("🏁 Starting Complete F1 Baseline Evaluation Pipeline")
    logger.info("=" * 70)
    
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    
    # Step 1: Run baseline evaluation
    baseline_script = scripts_dir / "baseline_evaluation.py"
    if not baseline_script.exists():
        logger.error(f"❌ Baseline evaluation script not found: {baseline_script}")
        return False
    
    success1 = run_script(baseline_script, "Step 1: Baseline Model Evaluation")
    if not success1:
        logger.error("❌ Baseline evaluation failed, stopping pipeline")
        return False
    
    # Step 2: Run quality evaluation
    quality_script = scripts_dir / "response_quality_evaluator.py"
    if not quality_script.exists():
        logger.error(f"❌ Quality evaluation script not found: {quality_script}")
        return False
    
    success2 = run_script(quality_script, "Step 2: Response Quality Evaluation")
    if not success2:
        logger.error("❌ Quality evaluation failed")
        return False
    
    # Final summary
    logger.info("=" * 70)
    logger.info("🎉 COMPLETE BASELINE EVALUATION PIPELINE FINISHED!")
    logger.info("=" * 70)
    logger.info("✅ All 3 models evaluated on F1 questions")
    logger.info("✅ Response quality assessed and scored")
    logger.info("✅ Results saved to results/ directory")
    logger.info("")
    logger.info("📁 Check these directories for results:")
    logger.info("   - results/baseline_evaluation/ (model responses)")
    logger.info("   - results/quality_evaluation/ (quality scores)")
    logger.info("")
    logger.info("🚀 Ready for next phase: Model fine-tuning!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
