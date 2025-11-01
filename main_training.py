"""
Main Training Script
Orchestrates curriculum learning training workflow
"""

from bert_trainer import BertTrainer
from config import Config, ExperimentConfig


def train_single_stage(exp_config):
    """
    Train a single curriculum learning stage.
    
    Args:
        exp_config: ExperimentConfig object with experiment settings
    """
    print(f"\n{'='*80}")
    print(f"Training Experiment {exp_config.exp_num}, Stage {exp_config.stage}")
    print(f"{'='*80}\n")
    
    # Initialize trainer
    trainer = BertTrainer(
        training_dataset_path=exp_config.get_dataset_path(),
        model_name=exp_config.model_name,
        labels=Config.DIALECT_LABELS,
        threshold=exp_config.threshold,
        exp_num=exp_config.exp_num,
        stage=exp_config.stage
    )
    
    trainer.save_dir = exp_config.get_output_dir()
    
    # Train
    trainer.train(
        num_train_epochs=exp_config.epochs,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        per_device_train_batch_size=exp_config.batch_size,
        per_device_eval_batch_size=exp_config.batch_size,
    )
    
    # Evaluate
    print(f"\nEvaluating Stage {exp_config.stage}...")
    trainer.evaluate(dev_path=str(Config.DEV_PATH))
    
    print(f"\nStage {exp_config.stage} completed.\n")


def train_curriculum_sequence(exp_num, start_stage=1, end_stage=16, epochs=1, batch_size=24):
    """
    Train a complete curriculum learning sequence.
    
    Args:
        exp_num: Experiment number
        start_stage: Starting stage number
        end_stage: Ending stage number
        epochs: Number of epochs per stage
        batch_size: Batch size for training
    """
    print(f"\n{'#'*80}")
    print(f"Starting Curriculum Learning Sequence - Experiment {exp_num}")
    print(f"Stages: {start_stage} to {end_stage}")
    print(f"{'#'*80}\n")
    
    # Train initial stage (stage 0) from pretrained model
    if start_stage == 1:
        exp_config = ExperimentConfig(
            exp_num=exp_num,
            stage=1,
            model_name=Config.DEFAULT_MODEL,
            threshold=Config.DEFAULT_THRESHOLD,
            batch_size=batch_size,
            epochs=epochs,
            use_previous_stage_model=False
        )
        train_single_stage(exp_config)
        start_stage = 2
    
    # Train subsequent stages using previous stage models
    for stage in range(start_stage, end_stage + 1):
        exp_config = ExperimentConfig(
            exp_num=exp_num,
            stage=stage,
            threshold=Config.DEFAULT_THRESHOLD,
            batch_size=batch_size,
            epochs=epochs,
            use_previous_stage_model=True
        )
        train_single_stage(exp_config)
    
    print(f"\n{'#'*80}")
    print(f"Curriculum Learning Sequence Completed - Experiment {exp_num}")
    print(f"{'#'*80}\n")


def train_standalone_experiment(
    exp_num,
    dataset_path=None,
    model_name=None,
    stage=0,
    epochs=1,
    batch_size=24,
    threshold=0.3
):
    """
    Train a standalone experiment (not part of curriculum sequence).
    
    Args:
        exp_num: Experiment number
        dataset_path: Path to training dataset (None for default stage 1)
        model_name: Model name or path (None for default)
        stage: Stage number for organization
        epochs: Number of training epochs
        batch_size: Batch size
        threshold: Classification threshold
    """
    if dataset_path is None:
        dataset_path = Config.get_stage_path(1)
    
    if model_name is None:
        model_name = Config.DEFAULT_MODEL
    
    trainer = BertTrainer(
        training_dataset_path=dataset_path,
        model_name=model_name,
        labels=Config.DIALECT_LABELS,
        threshold=threshold,
        exp_num=exp_num,
        stage=stage
    )
    
    trainer.save_dir = f'./exp_{exp_num}'
    
    trainer.train(
        num_train_epochs=epochs,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    
    trainer.evaluate(dev_path=str(Config.DEV_PATH))


if __name__ == "__main__":
    # Example usage:
    
    # Option 1: Train a complete curriculum sequence
    # train_curriculum_sequence(exp_num=28, start_stage=0, end_stage=15, epochs=2, batch_size=24)
    
    # Option 2: Train a standalone experiment
    # train_standalone_experiment(exp_num=27, epochs=1, batch_size=24)
    
    # Option 3: Train a single stage
    # exp_config = ExperimentConfig(exp_num=28, stage=0, batch_size=24, epochs=2)
    # train_single_stage(exp_config)
    
    pass
