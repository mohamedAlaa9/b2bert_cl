"""
Configuration Module
Central location for paths, parameters, and experiment settings
"""

from pathlib import Path


class Config:
    """Configuration class for curriculum learning experiments."""
    
    # Base paths
    BASE_DIR = Path("/home/ali.mekky/Documents/NLP/Project")
    DATASET_DIR = BASE_DIR / "Cross-Country-Dialectal-Arabic-Identification"
    NADI_DIR = BASE_DIR / "NADI2024/subtask1"
    
    # Dataset paths
    MAIN_DATASET_PATH = DATASET_DIR / "CL_stages" / "NADIcombined_cleaned_MULTI_LABEL_MODIFIED_FINAL.csv"
    DEV_PATH = NADI_DIR / "dev" / "NADI2024_subtask1_dev2.tsv"
    OUTPUT_FILE = DATASET_DIR / "output.txt"
    
    # Curriculum learning stage paths
    STAGE_DIR = DATASET_DIR / "CL_stages"
    
    @classmethod
    def get_stage_path(cls, stage_num):
        """Get path for a specific curriculum stage."""
        return str(cls.STAGE_DIR / f"stage_{stage_num}.csv")
    
    @classmethod
    def get_all_stage_paths(cls):
        """Get paths for all curriculum learning stages."""
        stage_numbers = list(range(1, 16)) + [18]
        return [cls.get_stage_path(i) for i in stage_numbers]
    
    # Model configuration
    DEFAULT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    ALTERNATIVE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    
    # Dialect labels
    DIALECT_LABELS = [
        'Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
        'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
        'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen'
    ]
    
    # Training hyperparameters (defaults)
    DEFAULT_THRESHOLD = 0.3
    DEFAULT_BATCH_SIZE = 24
    DEFAULT_EPOCHS = 2
    DEFAULT_LEARNING_RATE = 5e-5
    DEFAULT_WARMUP_STEPS = 500
    DEFAULT_PATIENCE = 2
    DEFAULT_DROPOUT = 0.3
    
    # Evaluation settings
    EVAL_SUBSET_INDEXES = [0, 2, 4, 10, 13, 14, 15, 17]


class ExperimentConfig:
    """Configuration for a specific experiment."""
    
    def __init__(
        self,
        exp_num,
        stage=0,
        model_name=None,
        threshold=None,
        batch_size=None,
        epochs=None,
        use_previous_stage_model=False
    ):
        """
        Initialize experiment configuration.
        
        Args:
            exp_num: Experiment number
            stage: Current curriculum stage
            model_name: Model to use (None for default)
            threshold: Classification threshold (None for default)
            batch_size: Batch size (None for default)
            epochs: Number of epochs (None for default)
            use_previous_stage_model: Whether to load model from previous stage
        """
        self.exp_num = exp_num
        self.stage = stage
        self.threshold = threshold or Config.DEFAULT_THRESHOLD
        self.batch_size = batch_size or Config.DEFAULT_BATCH_SIZE
        self.epochs = epochs or Config.DEFAULT_EPOCHS
        
        # Determine model name
        if use_previous_stage_model and stage > 0:
            self.model_name = self._get_previous_stage_model_path()
        else:
            self.model_name = model_name or Config.DEFAULT_MODEL
    
    def _get_previous_stage_model_path(self):
        """Get path to model from previous curriculum stage."""
        return f"/home/ali.mekky/Documents/NLP/B2Bert_refactored_code/exp_{self.exp_num}/stage_{self.stage - 1}"
    
    def get_dataset_path(self):
        """Get dataset path for current stage."""
        stage_num = self.stage if self.stage < 16 else 18
        return Config.get_stage_path(stage_num)
    
    def get_output_dir(self):
        """Get output directory for this experiment."""
        return f"./exp_{self.exp_num}"
    
    def get_stage_output_dir(self):
        """Get output directory for current stage."""
        return f"./exp_{self.exp_num}/stage_{self.stage}"
