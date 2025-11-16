# F1 Conversational AI: A Comparative Analysis of Fine-Tuning Methods

## Project Overview

This repository contains the complete implementation of a research project focused on fine-tuning large language models for Formula 1 conversational AI applications. The project compares different fine-tuning approaches including full-parameter fine-tuning and LoRA (Low-Rank Adaptation) on pre-trained models like Gemma-7B and LLaMA 2-7B.

### Key Objectives

- Develop a comprehensive F1 knowledge base through web scraping and data curation
- Evaluate baseline performance of pre-trained models on F1-specific tasks
- Implement and compare full-parameter and LoRA fine-tuning methods
- Create an automated evaluation framework using LLM-as-judge
- Generate comprehensive performance analysis and visualizations

## Directory Structure

```
├── config/                                            # Configuration files
│   ├── data_config.yaml                               # Defines the complete data sourcing, processing, storage, and generation pipeline for building a Formula 1-focused conversational AI
│   ├── env.example                                    # Environment variable file used to store confidential API keys, cloud credentials, file paths, and hardware and training settings for the F1 Conversational AI project
│   ├── evaluation_config.yaml                         # Specifies a detailed evaluation strategy for the F1 Conversational AI, defining the models to be tested, the metrics to measure, the datasets to use, and the process for statistical analysis and reporting
│   └── training_config.yaml                           # Specifies all the parameters, including LoRA settings, data paths, and hyperparameters, needed to fine-tune a large language model for the F1 Conversational AI project
├── data/                                              # Data storage and processing
│   ├── f1_qa_outputs_001_100/                         # Refined question answer pairs json files (001-100)
│   ├── f1_qa_outputs_101_200/                         # Refined question answer pairs json files (101-200)
│   ├── f1_qa_outputs_201_300/                         # Refined question answer pairs json files (201-300)
│   ├── f1_qa_outputs_301_400/                         # Refined question answer pairs json files (301-400)
│   ├── f1_qa_outputs_401_500/                         # Refined question answer pairs json files (401-500)
│   └── f1_qa_outputs_501_512/                         # Refined question answer pairs json files (501-512)
├── docs/                                              # Documentation
│   └── README.md                                      # Initial repository setup for F1 Conversational AI project
├── models/                                            # Model storage and management
│   ├── base/                                          # Pre-trained base models
│   │   ├── falcon-7b-instruct/                        # Model card of Falcom-7B-Instruct model
│   │   ├── mistral-7b/                                # Model card of Mistral-7B-v0.1 model
│   │   └── phi-3-mini/                                # Model card of Phi-3-Mini-4K-Instruct model
│   └── README.md                                      # Downloaded language models for the F1 Conversational AI project
├── notebooks/                                         # Contains Jupyter notebooks for analysis and experimentation.
│   ├── alternateFiles/                                # Alternative files for loading QA dataset and formatting multiple choice questions
│   ├── 02a_mistral_7b_evaluation.ipynb                # Evaluates Mistral 7B on the F1 QA dataset with multiple choice questions
│   ├── 02b_phi3_mini_evaluation.ipynb                 # Evaluates Phi-3 Mini on the F1 QA dataset with multiple choice questions
│   ├── 02c_llama_2_7b_chat_hf_evaluation.ipynb        # Evaluates Llama 2 7B Instruct on the F1 QA dataset with multiple choice questions
│   ├── 02c_llama_3_1_8b_evaluation.ipynb              # Evaluates Llama 3.1 8B Instruct on the F1 QA dataset with multiple choice questions
│   ├── HUGGINGFACE_APPROACH.md                        # Outlines a faster and more reliable F1 QA evaluation workflow that now loads models directly from Hugging Face Hub instead of Google Drive.
│   ├── README.md                                      # ExpJupyter notebooks for analysis
│   ├── Readme_f1_lora_finetune.md                     # README file of f1_lora_finetune.ipynb file
│   ├── f1_data_aggregator.ipynb                       # Complete pipeline for aggregating various Formula 1 data sources
│   ├── f1_data_preprocessing.ipynb                    # Covers the tasks for preprocessing and cleaning the raw data collected in the previous steps
│   ├── f1_lora_finetune.ipynb                         # End-to-end pipeline for fine-tuning small open models on a Formula 1 QA dataset using LoRA, then evaluating them on a large multiple-choice benchmark
│   └── f1_qa_utils.py                                 # F1 QA Evaluation Utilities, reusable functions for loading QA dataset and formatting multiple choice questions
├── scripts/                                           # Phase-specific execution scripts
│   ├── eval_framework/                                # Python-based system designed to automatically evaluate and score AI models using a powerful LLM (like GPT-4) as a "judge"
│   │   ├── eval_set/                                  # Datasets for evaluation purpose
│   │   │   ├── answers.txt                            # Datasets for answers
│   │   │   ├── answers_small.txt                      # Small dataset for answers
│   │   │   ├── notes.txt                              # Datasets for notes
│   │   │   ├── notes_small.txt                        # Small dataset for notes
│   │   │   ├── questions.txt                          # Datasets for questions
│   │   │   └── questions_small.txt                    # Small dataset for questions
│   │   ├── README.md                                  # Explains about the evaluation framework
│   │   ├── __init__.py                                # Initiation py code
│   │   └── eval.py                                    # Defines an evaluation framework that uses an LLM to automatically score F1 AI models and save the results to a CSV
│   ├── BASELINE_EVALUATION_README.md                  # Provides instructions on how to run the evaluation scripts (tasks 2.2 and 2.3) to test the baseline performance and quality of three F1 AI models
│   ├── README.md                                      # Contains the implementation scripts for each phase of the F1 Conversational AI project
│   ├── baseline_evaluation.py                         # Loads multiple F1-focused AI models, generates their responses to a small set of evaluation questions, and saves all the prompts and responses to a JSON file
│   ├── download_models.py                             # Checks for sufficient disk space before downloading and saving three large language models (Mistral 7B, Llama 2 8B, and CodeLlama 7B) from Hugging Face to a local directory
│   ├── response_quality_evaluator.py                  # loads the latest baseline model responses, scores them using F1-specific keywords and comparisons to ground truth answers, and then saves a new JSON report summarizing the quality and performance of each model
│   ├── run_baseline_evaluation.py                     # A simple pipeline runner that sequentially executes the baseline model evaluation and then the response quality evaluation scripts
│   ├── setup_models.py                                # Uses the huggingface_hub library to download three specific language models (Mistral, Phi-3, and Falcon) to a local models/base directory
│   └── verify_models.py                               # Checks all subdirectories in the models/base folder to confirm that each one contains the necessary configuration and tokenizer files for the F1 AI models
├── tests/                                             # Unit and integration tests
│   ├── __init__.py                                    # Initiation py code
│   └── test_utils.py                                  # Uses pytest to run unit tests that check if configuration files and JSON/JSONL data utilities are loading and saving data correctly
├── utils/                                             # Utility functions and helpers
│   ├── __init__.py                                    # Initiation py code
│   ├── cli.py                                         # Uses the click library to create a command-line interface (CLI) with placeholder commands for setting up the project, curating data, training, and evaluating the F1 Conversational AI
│   ├── config.py                                      # Defines a Config class to manage loading YAML configuration files and .env environment variables for the F1 Conversational AI project
│   ├── data_utils.py                                  # Defines data utility functions for reading, writing, splitting, and validating JSON, JSONL, and CSV files, and for converting them into Hugging Face datasets
│   ├── logger.py                                      # Uses the loguru library to set up and configure a flexible logger that can output to both the console and a file in either text or JSON format
│   └── model_utils.py                                 # provides utility functions for loading, quantizing, applying LoRA, and saving Hugging Face transformer models for the F1 Conversational AI project
├── .gitignore                                         # A .gitignore file for a Python machine learning project, designed to ignore large datasets, model checkpoints, experiment logs, virtual environments, IDE settings, and sensitive credential files
├── README.md                                          # This file, containing the complete implementation of a research project focused on fine-tuning large language models for Formula 1 conversational AI applications
├── requirements.txt                                   # Lists all the Python libraries and their minimum versions required for the F1 Conversational AI project, covering everything from deep learning and data processing to API deployment
├── setup.py                                           # Used to package the f1-conversational-ai project, installing its dependencies from requirements.txt and creating a command-line tool called f1-ai
└── test_model_loading.py                              # A test that sequentially loads three different AI models onto the CPU to verify they can be imported and can generate a short text response
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd f1-conversational-ai
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your API keys and paths
   ```

### Dependencies

Key dependencies include:

- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model training
- `datasets` - Dataset handling
- `beautifulsoup4` - Web scraping
- `fastf1` - F1 data access
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `wandb` - Experiment tracking

## Project Phases

### Phase 1: Data Curation

**Objective**: Collect, clean, and prepare F1-specific training data

**Key Components**:

- Web scraping scripts for F1 news, statistics, and historical data
- Kaggle dataset integration for structured F1 data
- Data preprocessing and cleaning pipelines
- GPT-4 assisted instruction-response dataset generation

**Scripts**:

- `notebooks/f1_data_aggregator.ipynb`
- `notebooks/f1_data_preprocessing.ipynb `

### Phase 2: Baseline Modeling

**Objective**: Evaluate pre-trained model performance on F1 tasks

**Key Components**:

- Inference scripts for Gemma-7B and LLaMA 2-7B
- Curated F1-specific prompt evaluation set
- Performance metrics collection

**Scripts**:

- `scripts/phase2_baseline/inference_gemma_base.py`
- `scripts/phase2_baseline/inference_llama_base.py`
- `scripts/phase2_baseline/evaluate_baseline.py`

### Phase 3: Core Experimentation

**Objective**: Implement and execute fine-tuning experiments

**Key Components**:

- Full-parameter fine-tuning implementation
- LoRA fine-tuning implementation
- Training configuration management
- Model checkpointing and artifact storage

**Scripts**:

- `scripts/phase3_training/train_full.py`
- `scripts/phase3_training/train_lora.py`
- `scripts/phase3_training/training_config.py`

### Phase 4: Evaluation

**Objective**: Comprehensive model evaluation using LLM-as-judge

**Key Components**:

- Automated evaluation framework
- LLM-as-judge implementation
- Performance metrics collection
- System resource monitoring

**Scripts**:

- `scripts/phase4_evaluation/llm_judge.py`
- `scripts/phase4_evaluation/evaluate_models.py`
- `scripts/phase4_evaluation/metrics_collection.py`

### Phase 5: Reporting & Deliverables

**Objective**: Synthesize findings and create deliverables

**Key Components**:

- Results visualization and analysis
- Performance comparison reports
- Live demonstration preparation
- Final documentation

**Scripts**:

- `scripts/phase5_reporting/generate_visualizations.py`
- `scripts/phase5_reporting/create_report.py`
- `scripts/phase5_reporting/prepare_demo.py`

## Usage

### Running Individual Phases

1. **Data Curation**

   ```bash
   python scripts/phase1_data_curation/main.py
   ```

2. **Baseline Evaluation**

   ```bash
   python scripts/phase2_baseline/main.py
   ```

3. **Training**

   ```bash
   python scripts/phase3_training/train_full.py --config config/training_config.yaml
   python scripts/phase3_training/train_lora.py --config config/lora_config.yaml
   ```

4. **Evaluation**

   ```bash
   python scripts/phase4_evaluation/main.py
   ```

5. **Reporting**
   ```bash
   python scripts/phase5_reporting/main.py
   ```

### Configuration

Configuration files are stored in the `config/` directory:

- `training_config.yaml` - Training hyperparameters
- `data_config.yaml` - Data processing settings
- `evaluation_config.yaml` - Evaluation parameters
- `.env` - Environment variables and API keys

## Data Management

### Data Sources

- **Web Scraping**: F1 news sites, official F1 statistics
- **Kaggle**: Structured F1 datasets
- **Generated**: GPT-4 assisted instruction-response pairs

### Data Storage

- Large datasets are stored in cloud storage (Google Cloud Storage)
- Local data directory contains processed and curated datasets
- Raw data is preserved for reproducibility

## Model Management

### Base Models

- Gemma-7B (Google)
- LLaMA 2-7B (Meta)

### Fine-tuned Models

- Full-parameter fine-tuned versions
- LoRA fine-tuned versions
- Model artifacts stored in `models/artifacts/`

## Evaluation Framework

### Metrics

- **Qualitative**: LLM-as-judge evaluation scores
- **Quantitative**: BLEU, ROUGE, perplexity
- **System**: GPU usage, training time, memory consumption

### Evaluation Process

1. Automated test set generation
2. Model inference on test prompts
3. LLM-as-judge scoring
4. Performance metrics calculation
5. Results visualization and analysis

## Results

### Expected Deliverables

- Comprehensive performance comparison report
- Training efficiency analysis
- Model quality assessment
- Live demonstration of conversational capabilities
- Reproducible experimental setup

### Output Locations

- `results/reports/` - Generated reports
- `results/visualizations/` - Charts and plots
- `results/demos/` - Demonstration materials

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Testing

Run the test suite:

```bash
pytest tests/
```

## Documentation

Additional documentation is available in the `docs/` directory:

- API documentation
- Configuration guides
- Troubleshooting guides
- Development setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformers library
- FastF1 for F1 data access
- The open-source ML community for tools and frameworks

## Contact

For questions or collaboration, please contact the project team.

---

**Note**: This project is for academic research purposes. Please ensure compliance with model licensing terms and data usage policies.
