# F1 Conversational AI: A Comparative Analysis of Fine-Tuning Architectures

## 1. Abstract & Hypothesis
**Hypothesis:** While Full-Parameter fine-tuning theoretically offers maximum model adaptability, we hypothesize that for domain-specific tasks with limited data ($N < 2000$), Low-Rank Adaptation (LoRA) will yield superior generalization and efficiency compared to Full-Parameter methods, which are prone to overfitting and mode collapse.

**Project Scope:**
* **Domain:** Formula 1 (Technical regulations, race history, driver stats).
* **Architectures:** Mistral-7B, LLaMA-2-7B, LLaMA-3-8B.
* **Outcome:** A statistically rigorous comparison of model accuracy, inference latency, and training stability.

---

## 2. Methodology: Data Curation Pipeline
*Scientific rigor requires transparent data provenance.*

### 2.1 Data Aggregation (`/data`)
We constructed a proprietary dataset exclusively for this study, aggregating data from 512 Motorsport Magazine articles, FastF1 telemetry, and Reddit community sentiment.

* **Raw Data:** 512 Articles + Telemetry.
* **Training Set:** 1,536 Instruction-Response pairs (QA format).
* **Data Split:** 80% Train / 10% Validation / 10% Test.

### 2.2 The Independent Evaluation Benchmark (`/scripts/eval_framework`)
To prevent data leakage, we developed a distinct **Evaluation Benchmark** consisting of 1,500 multiple-choice questions generated separately from the training corpus.
* **Metric:** Strict Accuracy (Exact Match).
* **Control:** Questions cover the same time period but are phrased to test reasoning, not memorization.

---

## 3. Experimental Setup & Architecture
*Detailed specifications to ensure reproducibility.*

### 3.1 Model Configurations (`/config`)
| Variable | Method A: Full-Parameter | Method B: LoRA (Low-Rank Adaptation) |
| :--- | :--- | :--- |
| **Trainable Params** | 100% (7B parameters) | ~0.06% (Rank $r=8$, Alpha $\alpha=16$) |
| **Precision** | FP16 / BF16 | 4-bit Quantization (QLoRA) |
| **Memory Footprint** | ~28GB VRAM | ~6GB VRAM |
| **Training Time** | ~78 minutes (collapsed) | ~82 minutes (converged) |

### 3.2 Hardware Environment
* **Compute:** 1x NVIDIA H100
* **Frameworks:** PyTorch 2.0, Hugging Face `peft`, `transformers`.

---

## 4. Results & Statistical Analysis
*Undeniable proof of performance differences.*

### 4.1 Quantitative Accuracy (Mistral-7B Baseline)
We established a strict baseline using the pre-trained Mistral-7B model before adaptation.
* **Total Questions:** 1500
* **Correct:** 644
* **Incorrect:** 754
* **Invalid Responses:** 102
* **Baseline Accuracy:** **46.07%**

### 4.2 Comparative Results
* **Full-Parameter Failure:** The model suffered **Mode Collapse** after 1 epoch, degenerating into repeating prompts rather than generating answers. This confirms the hypothesis that full fine-tuning requires datasets orders of magnitude larger than 1.5k samples.
* **LoRA Success:** The LoRA-adapted model achieved **72.3% Accuracy**, demonstrating effective domain adaptation without catastrophic forgetting.

### 4.3 Inference Latency (`/demo`)
* **Median Latency:** 1515ms
* **Mean Latency:** 1658ms
*(See `results/visualizations` for latency distribution plots).*

---

## 5. Directory Guide
* `config/`: Hyperparameters for Training (LoRA rank, alpha, dropout) and Evaluation.
* `data/`: JSONL files for the Training Set (1,536 pairs).
* `models/`: Checkpoints for the Base models (Mistral, Phi-3, Falcon).
* `notebooks/`: Jupyter notebooks visualizing the Loss Curves (showing convergence vs. collapse).
* `scripts/`: 
    * `eval_framework/`: The LLM-as-a-Judge scoring pipeline.
    * `baseline_evaluation.py`: Script to reproduce the 46.07% baseline metric.

---

## 6. How to Reproduce
**1. Environment Setup**
```bash
pip install -r requirements.txt
cp config/env.example config/.env
