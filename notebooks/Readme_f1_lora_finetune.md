# F1 LoRA Fine-Tuning and Evaluation Notebook

This notebook is my end-to-end pipeline for fine-tuning small open models on a Formula 1 QA dataset using LoRA, then evaluating them on a large multiple-choice benchmark.

I wrote this for my CS6220 final project, so the explanations are in my own words and assume I run everything locally on my machine.

---

## Goals

- Fine-tune two base models on F1 question–answer pairs:
  - `mistralai/Mistral-7B-v0.3`
  - `meta-llama/Meta-Llama-3.1-8B-Instruct` (referred to here as LLaMA 3.1 8B)
- Teach them to answer F1 questions correctly and consistently.
- Evaluate them on a big multiple-choice dataset and measure accuracy.

The notebook should:

1. Build the training data from my processed QA JSON files.
2. Run LoRA fine-tuning for Mistral and LLaMA.
3. Evaluate both models on the same MC dataset using a robust scoring method (no fragile JSON parsing).
4. Save all logs, metrics, and plots so I can compare models later, even if I reset the kernel between runs.

---

## Folder structure, inputs and outputs

The notebook assumes the project is laid out like this:

- **Project root and notebook**
  - `PROJECT_ROOT` is inferred from the notebook path.
  - The notebook itself lives under  
    `PROJECT_ROOT/notebooks/`  
    (for me it is something like  
    `D:\Projects\CS6220\Final Project\f1-conversational-ai-main\notebooks\f1_lora_finetune.ipynb`).

- **Training QA data (for SFT)**
  - `F1_QA_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "f1_qa"`
  - This folder contains files like `qa_1.json`, `qa_2.json`, …  
    Each file has:
    - `article_number`, `source`, `title`, `summary`
    - `qa_pairs`, where each pair has:
      - `question`
      - `correct_answer`
      - `wrong_options` (three distractors)
      - `rephrased_question`
      - `options` dict with A, B, C, D
      - `ground_truth_correct_option` (the letter)
      - `prompt` (original MC prompt used earlier)

  - In this notebook I **do not** reuse the old MC prompt.  
    Instead I build my own SFT text:
    > You are a knowledgeable Formula 1 expert.  
    > Answer the question briefly and accurately.  
    > Question: …  
    > Answer: {correct_answer}

- **Eval MC data (big dataset for scoring)**
  - `BIG_EVAL_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "big_data_dataset_evalScripts" / "dataset"`
  - Contains about 527 files like `qa_1.json`, `qa_2.json`, … with:
    - `qa_pairs` and the same fields as above.
  - For evaluation I only use:
    - `rephrased_question` (fallback to `question`)
    - `options` (A, B, C, D)
    - `ground_truth_correct_option`.

- **Processed prompts (optional cache)**
  - `EVAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "LoRA_Processed_Inputs"`
  - Used to save the processed SFT prompts and metadata so I do not have to recompute them each time.

- **Model outputs and logs**
  - `RESULTS_DIR = PROJECT_ROOT / "results" / "LoRA_Results"`
    - Per model LoRA weights:
      - `RESULTS_DIR / "mistral_lora"`  
      - `RESULTS_DIR / "llama_lora"`
    - Training logs:
      - `mistral_train_log.json`, `llama_train_log.json`
      - `mistral_train_meta.json`, `llama_train_meta.json`
  - `EVAL_RESULTS_DIR` (under `RESULTS_DIR`)  
    - Evaluation outputs, for example:
      - `mistral_bigdata_eval_results_scoring.json`
      - `llama_bigdata_eval_results_scoring.json`
    - Final CSVs:
      - `epoch_results.csv` (epoch-level MC accuracy and times)
      - `summary_results.csv` (one row per model, final accuracy, train time, eval time)

---

## Requirements and environment

This notebook expects:

- Python environment with:
  - `torch` and a working GPU (A100 type is what I used)
  - `transformers`
  - `peft`
  - `accelerate`
  - `bitsandbytes` (for 4-bit loading)
  - `pandas`, `matplotlib`
- Access to the HF model IDs listed in `MODELS`.
- Enough GPU memory to fine-tune 7B–8B models with LoRA and gradient checkpointing.

The first code cells check GPU details and print:

- Device type
- GPU name and total memory
- Torch and transformers versions

If something looks wrong there, the rest of the notebook will probably not behave well.

---

## What actually happens in each logical section

I structured the notebook in blocks. Roughly:

1. **Environment and GPU check**
   - Import packages.
   - Print basic system info, GPU type, memory, and key library versions.

2. **Project paths and configuration**
   - Infer `NOTEBOOK_DIR` and `PROJECT_ROOT`.
   - Define:
     - `F1_QA_INPUT_DIR`
     - `EVAL_OUTPUT_DIR`
     - `BIG_EVAL_DATASET_DIR`
     - `RESULTS_DIR` and `EVAL_RESULTS_DIR`
   - Define `MODELS` list with Mistral and LLaMA configs.
   - Set common hyperparameters:
     - `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WARMUP_RATIO`, `MAX_SEQ_LEN`.

3. **LoRA configuration utilities**
   - Helper function `create_tokenizer(model_id)` that:
     - Loads the tokenizer from HF.
     - Ensures `pad_token_id` is set (fall back to `eos_token_id` if needed).
   - Helper function `create_base_model(model_id)` that:
     - Loads the base model in 4-bit with `bitsandbytes`.
     - Puts it on the GPU.
   - Helper function `apply_lora(model)` that:
     - Wraps the base model with LoRA layers using `peft`.
   - All LoRA hyperparameters (rank, alpha, dropout) are defined here.

4. **Building the SFT training dataset**
   - Read all QA JSON files from `F1_QA_INPUT_DIR`.
   - For each `qa_pair`, build a simple SFT string:
     - Short instruction
     - Question
     - Correct answer only (no distractors).
   - Save everything into a `datasets.Dataset` called `train_dataset`, with fields:
     - `"text"` (prompt + answer)
     - `"q_id"` (for tracking).
   - Print the first 5 raw examples so I can inspect exactly what the model sees.

5. **Tokenization helper**
   - `tokenize_sft_function(examples, tokenizer)`:
     - Tokenizes the `"text"` field.
     - Applies truncation and padding to `MAX_SEQ_LEN`.
   - This is used for both Mistral and LLaMA to produce model-specific tokenized datasets.

6. **Mistral LoRA training**
   - Create Mistral tokenizer and tokenized dataset `mistral_ds`.
   - Print a few decoded tokenized examples, which show a lot of `<s>` tokens at the end because of padding.
     - This is expected: the dataset is padded with `<s>` (BOS/EOS) as pad token.
   - Load base Mistral model and wrap it with LoRA.
   - Set `pad_token_id` for model and tokenizer.
   - Configure `TrainingArguments`:
     - Epochs, batch size, gradient accumulation, learning rate, warmup, FP16, no logging to external services.
   - Run `Trainer.train()` and print training loss every few steps.
   - Save:
     - LoRA weights into `RESULTS_DIR / "mistral_lora"`
     - Training log (`mistral_train_log.json`)
     - Training meta (`mistral_train_meta.json`).

7. **Mistral evaluation with MC scoring (no generation)**
   - Load evaluation questions from `BIG_EVAL_DATASET_DIR` using:
     - `load_big_eval_questions_for_scoring`.
   - For each question:
     - Build a fresh prompt that matches the SFT style:

       > You are a knowledgeable Formula 1 expert.  
       > Answer the question briefly and accurately.  
       > Question: …  
       > Options: A: … B: … C: … D: …  
       > Answer:

     - For each option A, B, C, D:
       - Call `score_option_logprob`:
         - Concatenate prompt + option text.
         - Run the model once.
         - Compute average log probability of the answer tokens.
     - Choose the option letter with the highest average log-prob.
   - The key point is:  
     There is **no generation** and **no JSON parsing** here.  
     I only ask the model “How likely is it that the answer is this string” four times and pick the best.
   - `run_big_mc_eval_scoring`:
     - Prints progress every 20 questions:
       - `Scoring question i/N...`
       - Predicted option and ground truth
       - Per-option scores
     - At the end it prints:
       - Total number of questions
       - Number correct
       - Overall accuracy
       - Evaluation time
     - Stores detailed per-question records in a JSON under `EVAL_RESULTS_DIR`.

   - Current outcome for Mistral (on my run):  
     about **72.3% MC accuracy** on 1581 questions.

8. **LLaMA LoRA training**
   - Same logic as Mistral, but for `meta-llama/Meta-Llama-3.1-8B-Instruct`:
     - Build tokenizer and tokenized dataset `llama_ds` from `train_dataset`.
     - Print first raw and tokenized examples for sanity.
     - Load base LLaMA model and apply LoRA.
     - Train with the same hyperparameters.
     - Save:
       - `llama_lora` directory
       - `llama_train_log.json`
       - `llama_train_meta.json`.

9. **LLaMA evaluation with MC scoring**
   - Reuse the same scoring functions used for Mistral:
     - `load_big_eval_questions_for_scoring`
     - `build_scoring_prompt`
     - `score_option_logprob`
     - `run_big_mc_eval_scoring`
   - Load the LLaMA LoRA adapter and evaluate on the same MC dataset.
   - Save results to:
     - `llama_bigdata_eval_results_scoring.json`
   - Append a summary row for LLaMA into the in-memory `summary_rows` list.

10. **Collecting results and plotting**
    - Build:
      - `epoch_results_df` from `all_epoch_logs` (epoch-level information, if available).
      - `summary_df` from `summary_rows` (one row per model).
    - Save:
      - `epoch_results.csv`
      - `summary_results.csv`
    - Plot:
      - MC accuracy per epoch by model (if epoch logs exist).
      - Training time per model.
      - Final MC accuracy per model.
      - Evaluation time per model.

    Note: because I sometimes reset the kernel between model runs, the notebook also supports reading existing JSONs and CSVs back from disk and reconstructing `summary_df` so that both models (Mistral and LLaMA) appear in the final plots.

---

## Challenges along the way

Some of the main issues I hit and how this notebook addresses them:

1. **Generation-based MC evaluation failed**
   - When I tried to ask the model to output a JSON like:
     ```json
     { "model_correct_option": "C", "justification": "..." }
     ```
     the model often replied with just `</s>` or random text.  
     The parsing code could not reliably extract the answer, and I ended up with `Parsed: 0, Skipped: 1581`.

2. **EOS and padding**
   - Mistral and LLaMA use `<s>` and `</s>` tokens, and sometimes the tokenizer reused `<s>` as pad.
   - This is why decoded tokenized examples show many `<s>` at the end.
   - The notebook explicitly aligns `pad_token_id` with `eos_token_id` so that generation and losses behave more predictably.


---

## Current outcomes

Right now, after the scoring-based evaluation is in place:

- **Mistral-7B LoRA**
  - Successfully trains on my F1 QA SFT dataset.
  - Reaches around **72% multiple-choice accuracy** on the 1,581 held-out questions using log-prob scoring.
  - Training and eval runtimes are recorded in `summary_results.csv`.

- **LLaMA-3.1-8B-Instruct LoRA**
  - Fine-tuning and scoring evaluation follow the exact same pipeline.
  - Final accuracy and timing are also written into `summary_results.csv` once the evaluation run completes.

The notebook is now stable: data loading, training, and evaluation are aligned, and the scoring method is robust to model output quirks. If I want to update anything later (change models, tweak prompts, add epochs), I can do it in a controlled way without breaking the whole pipeline.
