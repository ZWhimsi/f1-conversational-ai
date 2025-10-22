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
├── data/                           # Data storage and processing
│   ├── raw/                       # Raw scraped and downloaded data
│   ├── processed/                 # Cleaned and preprocessed data
│   └── curated/                   # Final instruction-response training datasets
├── models/                        # Model storage and management
│   ├── base/                      # Pre-trained base models
│   ├── checkpoints/               # Training checkpoints
│   └── artifacts/                 # Final trained model artifacts
├── scripts/                       # Phase-specific execution scripts
│   ├── phase1_data_curation/     # Data collection and preprocessing
│   ├── phase2_baseline/          # Baseline model evaluation
│   ├── phase3_training/          # Fine-tuning implementation
│   ├── phase4_evaluation/        # Model evaluation framework
│   └── phase5_reporting/         # Results synthesis and reporting
├── notebooks/                     # Jupyter notebooks for analysis
├── experiments/                   # Experiment tracking and logs
├── config/                        # Configuration files
├── utils/                         # Utility functions and helpers
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
└── results/                       # Outputs and deliverables
    ├── visualizations/            # Charts and plots
    ├── reports/                   # Generated reports
    └── demos/                     # Live demonstration materials
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

- `scripts/phase1_data_curation/scrape_f1_data.py`
- `scripts/phase1_data_curation/process_kaggle_data.py`
- `scripts/phase1_data_curation/generate_training_data.py`

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
