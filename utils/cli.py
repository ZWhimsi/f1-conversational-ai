"""
Command-line interface for the F1 Conversational AI project.
"""

import click
from pathlib import Path
from typing import Optional
from utils.logger import setup_logger, get_logger
from utils.config import config


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', default=None, help='Log file path')
@click.option('--config-dir', default='config', help='Configuration directory')
def cli(log_level: str, log_file: Optional[str], config_dir: str):
    """F1 Conversational AI CLI."""
    setup_logger(log_level=log_level, log_file=log_file)
    global config
    config = config.__class__(config_dir)


@cli.command()
@click.option('--data-config', default='data_config.yaml', help='Data configuration file')
@click.option('--output-dir', default='data/curated', help='Output directory')
def curate_data(data_config: str, output_dir: str):
    """Run data curation pipeline."""
    logger = get_logger("data_curation")
    logger.info("Starting data curation pipeline")
    
    # TODO: Implement data curation pipeline
    click.echo("Data curation pipeline not yet implemented")


@cli.command()
@click.option('--model-name', required=True, help='Model name or path')
@click.option('--test-file', required=True, help='Test data file')
@click.option('--output-file', required=True, help='Output file for results')
def evaluate_baseline(model_name: str, test_file: str, output_file: str):
    """Evaluate baseline model performance."""
    logger = get_logger("baseline_evaluation")
    logger.info(f"Evaluating baseline model: {model_name}")
    
    # TODO: Implement baseline evaluation
    click.echo("Baseline evaluation not yet implemented")


@cli.command()
@click.option('--config-file', required=True, help='Training configuration file')
@click.option('--model-type', type=click.Choice(['full', 'lora']), required=True, help='Training type')
@click.option('--output-dir', required=True, help='Output directory for model')
def train(config_file: str, model_type: str, output_dir: str):
    """Train model with specified configuration."""
    logger = get_logger("training")
    logger.info(f"Starting {model_type} training")
    
    # TODO: Implement training pipeline
    click.echo(f"{model_type.title()} training not yet implemented")


@cli.command()
@click.option('--eval-config', default='evaluation_config.yaml', help='Evaluation configuration file')
@click.option('--output-dir', default='results/evaluation', help='Output directory')
def evaluate(eval_config: str, output_dir: str):
    """Run model evaluation."""
    logger = get_logger("evaluation")
    logger.info("Starting model evaluation")
    
    # TODO: Implement evaluation pipeline
    click.echo("Model evaluation not yet implemented")


@cli.command()
@click.option('--output-dir', default='results/reports', help='Output directory for reports')
def generate_report(output_dir: str):
    """Generate final report."""
    logger = get_logger("reporting")
    logger.info("Generating final report")
    
    # TODO: Implement report generation
    click.echo("Report generation not yet implemented")


@cli.command()
def setup():
    """Setup project environment."""
    logger = get_logger("setup")
    logger.info("Setting up project environment")
    
    # Create necessary directories
    directories = [
        "data/raw", "data/processed", "data/curated",
        "models/base", "models/checkpoints", "models/artifacts",
        "logs", "results/evaluation", "results/reports", "results/visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    click.echo("Project environment setup complete")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
