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


def train_curriculum_sequence(exp_num, start_stage=1, end_stage=18, epochs=1, batch_size=24, order=None):
    """
    Train a complete curriculum learning sequence.
    
    Args:
        exp_num: Experiment number
        start_stage: Starting stage number
        end_stage: Ending stage number
        epochs: Number of epochs per stage
        batch_size: Batch size for training
        order: List specifying the order of stages (if None, defaults to sequential order)
    """
    print(f"\n{'#'*80}")
    print(f"Starting Curriculum Learning Sequence - Experiment {exp_num}")
    print(f"Stages: {start_stage} to {end_stage}")
    print(f"{'#'*80}\n")

    if order is None:
        order = list(range(start_stage, end_stage + 1))
        
    # Train initial stage from pretrained model
    exp_config = ExperimentConfig(
        exp_num=exp_num,
        stage=order[0],
        model_name=Config.DEFAULT_MODEL,
        threshold=Config.DEFAULT_THRESHOLD,
        batch_size=batch_size,
        epochs=epochs,
        use_previous_stage_model=False
    )
    train_single_stage(exp_config)

    # Train subsequent stages using previous stage models
    for i in range(1, len(order)):
        exp_config = ExperimentConfig(
            exp_num=exp_num,
            stage=order[i],
            threshold=Config.DEFAULT_THRESHOLD,
            batch_size=batch_size,
            epochs=epochs,
            use_previous_stage_model=True,
            prev_stage=order[i-1]
        )
        train_single_stage(exp_config)
    
    print(f"\n{'#'*80}")
    print(f"Curriculum Learning Sequence Completed - Experiment {exp_num}")
    print(f"{'#'*80}\n")

