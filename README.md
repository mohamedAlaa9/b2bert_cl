# Curriculum Learning for Arabic Dialect Classification

## Overview
This project implements curriculum learning for multi-label Arabic dialect classification using BERT models. The code has been reorganized for better modularity, maintainability, and clarity while preserving all original implementation details.

## Project Structure

```
.
├── bert_trainer.py              # Main trainer class with model training logic
├── data_utils.py                # Data preparation and curriculum stage creation
├── config.py                    # Configuration and path management
├── main_training.py             # High-level training workflows
├── prepare_data.py              # Data preparation scripts
├── curriculum_learning_notebook.ipynb  # Jupyter notebook with examples
└── README.md                    # This file
```

## Module Descriptions

### `bert_trainer.py`
Core training module containing:
- **`TweetDataset`**: PyTorch dataset for tweet data
- **`CustomTrainer`**: Custom Hugging Face trainer with contiguous parameter handling
- **`BertTrainer`**: Main trainer class with methods for:
  - Model initialization and configuration
  - Data loading and preprocessing
  - Training and evaluation
  - Metrics computation
  - Prediction generation

**Key Features:**
- Automatic train/validation split (90/10)
- Multi-label classification support
- Configurable dropout and layer freezing
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics

### `data_utils.py`
Data preparation utilities:
- **`balance_countries()`**: Balance single-dialect samples across all countries
- **`create_curriculum_stage()`**: Create a single curriculum stage by combining samples
- **`prepare_all_curriculum_stages()`**: Generate all curriculum stages at once
- **`load_and_prepare_dataset()`**: Load and prepare dataset with dialect_sum calculation

### `config.py`
Centralized configuration:
- **`Config`**: Static configuration with paths, labels, and default hyperparameters
- **`ExperimentConfig`**: Per-experiment configuration with stage management

**Benefits:**
- Single source of truth for paths and parameters
- Easy modification of hyperparameters
- Support for both pretrained and checkpoint models

### `main_training.py`
High-level training workflows:
- **`train_single_stage()`**: Train one curriculum stage
- **`train_curriculum_sequence()`**: Train complete curriculum sequence
- **`train_standalone_experiment()`**: Train standalone experiment

### `prepare_data.py`
Data preparation scripts:
- **`prepare_curriculum_data()`**: Create all curriculum stages from main dataset
- **`analyze_dataset_distribution()`**: Analyze dataset statistics
- **`create_stage_1_balanced()`**: Create balanced stage 1 dataset

## Usage Examples

### 1. Train a Standalone Experiment

```python
from main_training import train_standalone_experiment

train_standalone_experiment(
    exp_num=27,
    epochs=1,
    batch_size=24,
    threshold=0.3
)
```

### 2. Train Complete Curriculum Sequence

```python
from main_training import train_curriculum_sequence

train_curriculum_sequence(
    exp_num=28,
    start_stage=0,
    end_stage=15,
    epochs=2,
    batch_size=24
)
```

### 3. Train Single Stage with Custom Config

```python
from config import ExperimentConfig
from main_training import train_single_stage

exp_config = ExperimentConfig(
    exp_num=28,
    stage=5,
    batch_size=24,
    epochs=2,
    use_previous_stage_model=True
)

train_single_stage(exp_config)
```

### 4. Prepare Curriculum Data

```python
from prepare_data import prepare_curriculum_data

prepare_curriculum_data(
    dataset_path="path/to/dataset.csv",
    output_dir="path/to/output",
    computed_filter=True
)
```

### 5. Direct Trainer Usage (Low-Level)

```python
from bert_trainer import BertTrainer
from config import Config

trainer = BertTrainer(
    training_dataset_path=Config.get_stage_path(1),
    model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix",
    labels=Config.DIALECT_LABELS,
    threshold=0.3,
    exp_num=27
)

trainer.train(
    num_train_epochs=2,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24
)

trainer.evaluate(dev_path=str(Config.DEV_PATH))
```

## Hyperparameters (Unchanged from Original)

All hyperparameters are preserved exactly as in the original implementation:

- **Dropout rate**: 0.3
- **Frozen layers**: First 8 BERT encoder layers
- **Learning rate**: 5e-5
- **Weight decay**: 0.01
- **Warmup steps**: 500
- **Max sequence length**: 128
- **Train/val split**: 90/10 with random_state=42
- **Early stopping patience**: 2
- **FP16**: Enabled
- **LR scheduler**: Linear

## Curriculum Learning Stages

The curriculum is organized by `dialect_sum` (number of dialects per sample):
- **Stage 1**: Single-dialect samples (balanced across 18 countries)
- **Stage 2-15**: Progressively add samples with more dialects
- **Stage 18**: Maximum dialect complexity

Each stage includes balanced samples from all previous stages.

## Key Improvements in Organization

### 1. **Separation of Concerns**
   - Training logic → `bert_trainer.py`
   - Data preparation → `data_utils.py`
   - Configuration → `config.py`
   - Workflows → `main_training.py`

### 2. **Code Reusability**
   - Modular functions for common operations
   - Configurable experiment settings
   - Reusable data preparation utilities

### 3. **Maintainability**
   - Clear function/class documentation
   - Logical code organization
   - Constants defined in one place
   - Descriptive variable names

### 4. **Readability**
   - Consistent code style
   - Meaningful function names
   - Reduced code duplication
   - Clear workflow examples

## What Was NOT Changed

To preserve exact reproducibility:
- ✅ All model architectures
- ✅ All hyperparameters
- ✅ All random seeds (42)
- ✅ Data preprocessing logic
- ✅ Training algorithms
- ✅ Evaluation metrics
- ✅ Loss functions
- ✅ Tokenization settings
- ✅ Optimizer settings
- ✅ Curriculum construction logic

## Dependencies

```
torch
transformers
pandas
numpy
scikit-learn
```

## Output Structure

```
exp_<exp_num>/
├── stage_<stage_num>/         # Saved models per stage
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
├── results/                   # Training checkpoints
└── logs/                      # TensorBoard logs
```

## Notes

- All file paths in `config.py` should be adjusted to match your environment
- The `preprocess.py` module with `final_eliminations()` function is expected to be available
- GPU training is automatically enabled if available
- TensorBoard logs are written to `exp_<num>/logs/`

## Migration from Original Code

To migrate from the original notebook:

1. **Replace cell 3-4** (standalone experiments) with:
   ```python
   from main_training import train_standalone_experiment
   train_standalone_experiment(exp_num=27, epochs=1, batch_size=24)
   ```

2. **Replace cell 7** (data preparation) with:
   ```python
   from prepare_data import prepare_curriculum_data
   prepare_curriculum_data(dataset_path, output_dir)
   ```

3. **Replace cell 9** (curriculum loop) with:
   ```python
   from main_training import train_curriculum_sequence
   train_curriculum_sequence(exp_num=28, start_stage=1, end_stage=15)
   ```

## Contact

For questions about the implementation, refer to the original code or contact the development team.
