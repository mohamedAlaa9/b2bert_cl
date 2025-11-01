"""
Data Preparation Script
Creates curriculum learning stages from the main dataset
"""

import pandas as pd
from data_utils import (
    load_and_prepare_dataset,
    prepare_all_curriculum_stages,
    DIALECT_COLUMNS
)
from config import Config


def prepare_curriculum_data(
    dataset_path,
    output_dir,
    stage_levels=None,
    computed_filter=True
):
    """
    Prepare all curriculum learning stages from the main dataset.
    
    Args:
        dataset_path: Path to main dataset CSV
        output_dir: Directory to save stage files
        stage_levels: List of stage levels (None for default)
        computed_filter: Whether to filter 'Computed' == 'yes' rows
    """
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(dataset_path, computed_filter)
    
    print(f"Dataset loaded: {len(dataset)} rows")
    print(f"Dialect sum distribution:\n{dataset['dialect_sum'].value_counts().sort_index()}\n")
    
    print("Creating curriculum learning stages...")
    stage_paths = prepare_all_curriculum_stages(dataset, output_dir, stage_levels)
    
    print(f"\nSuccessfully created {len(stage_paths)} curriculum stages.")
    return stage_paths


def analyze_dataset_distribution(dataset_path):
    """
    Analyze and print dataset distribution statistics.
    
    Args:
        dataset_path: Path to dataset CSV
    """
    print("Analyzing dataset distribution...\n")
    
    dataset = load_and_prepare_dataset(dataset_path, computed_filter=True)
    
    print(f"Total samples: {len(dataset)}")
    print(f"\nDialect sum distribution:")
    print(dataset['dialect_sum'].value_counts().sort_index())
    
    print(f"\nSamples with dialect_sum == 1:")
    rows_with_sum_1 = dataset[dataset['dialect_sum'] == 1]
    print(f"Total: {len(rows_with_sum_1)}")
    
    print("\nDistribution by country (for dialect_sum == 1):")
    for dialect in DIALECT_COLUMNS:
        count = len(rows_with_sum_1[rows_with_sum_1[dialect] == 1])
        print(f"  {dialect}: {count}")
    
    print(f"\nSamples with dialect_sum == 2: {len(dataset[dataset['dialect_sum'] == 2])}")
    print(f"Samples with dialect_sum >= 3: {len(dataset[dataset['dialect_sum'] >= 3])}")


def create_stage_1_balanced(dataset_path, output_path, n_samples_per_country=None):
    """
    Create a balanced stage 1 dataset with equal samples from each country.
    
    Args:
        dataset_path: Path to main dataset
        output_path: Path to save balanced stage 1
        n_samples_per_country: Samples per country (None for auto-balance)
    """
    from data_utils import balance_countries
    
    print("Creating balanced stage 1 dataset...")
    
    dataset = load_and_prepare_dataset(dataset_path, computed_filter=True)
    
    if n_samples_per_country is None:
        # Auto-determine based on smallest country
        rows_with_sum_1 = dataset[dataset['dialect_sum'] == 1]
        min_count = min([
            len(rows_with_sum_1[rows_with_sum_1[dialect] == 1]) 
            for dialect in DIALECT_COLUMNS
        ])
        n_samples_per_country = min_count
    
    print(f"Balancing with {n_samples_per_country} samples per country...")
    
    balanced_data = balance_countries(n_samples_per_country, dataset)
    balanced_data.to_csv(output_path, index=False)
    
    print(f"Balanced stage 1 saved to: {output_path}")
    print(f"Total samples: {len(balanced_data)}")
    print(f"Distribution:\n{balanced_data[DIALECT_COLUMNS].sum()}")


if __name__ == "__main__":
    # Example usage:
    
    # Option 1: Analyze dataset distribution
    # analyze_dataset_distribution(str(Config.MAIN_DATASET_PATH))
    
    # Option 2: Prepare all curriculum stages
    # prepare_curriculum_data(
    #     dataset_path=str(Config.MAIN_DATASET_PATH),
    #     output_dir=str(Config.STAGE_DIR),
    #     computed_filter=True
    # )
    
    # Option 3: Create balanced stage 1
    # create_stage_1_balanced(
    #     dataset_path=str(Config.MAIN_DATASET_PATH),
    #     output_path=Config.get_stage_path(1)
    # )
    
    pass
