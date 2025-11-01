"""
Data Utilities for Curriculum Learning
Handles dataset balancing and stage preparation
"""

import pandas as pd
from sklearn.utils import resample


# Dialect column definitions
DIALECT_COLUMNS = [
    'Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
    'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
    'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen'
]


def balance_countries(n_samples, dataset):
    """
    Balance dataset across all dialect countries for single-dialect samples.
    
    Args:
        n_samples: Number of samples to draw for each dialect
        dataset: DataFrame containing the full dataset
        
    Returns:
        DataFrame with balanced samples across all dialects
    """
    # Filter rows with dialect_sum equal to 1
    rows_with_sum_1 = dataset[dataset['dialect_sum'] == 1]
    
    # Create an empty DataFrame to store balanced rows
    balanced_rows = pd.DataFrame()
    
    # Balance rows across all dialects
    for dialect in DIALECT_COLUMNS:
        # Select rows where the specific dialect column is 1
        dialect_rows = rows_with_sum_1[rows_with_sum_1[dialect] == 1]
        
        # Check if enough rows are available for resampling
        if len(dialect_rows) >= n_samples:
            resampled_rows = resample(
                dialect_rows, 
                replace=False, 
                n_samples=n_samples, 
                random_state=42
            )
        else:
            # Use replacement if there are not enough rows
            resampled_rows = resample(
                dialect_rows, 
                replace=True, 
                n_samples=n_samples, 
                random_state=42
            )
        
        balanced_rows = pd.concat([balanced_rows, resampled_rows], ignore_index=True)
    
    return balanced_rows


def create_curriculum_stage(dataset, stage_level):
    """
    Create a curriculum learning stage by combining samples with different
    dialect_sum values up to the specified stage level.
    
    Args:
        dataset: Full dataset with 'dialect_sum' column
        stage_level: Maximum dialect_sum value for this stage
        
    Returns:
        DataFrame containing samples for the curriculum stage
    """
    # Get rows where dialect_sum equals the current stage level
    current_rows = dataset[dataset['dialect_sum'] == stage_level]
    print(f"Initial rows for dialect_sum = {stage_level}: {len(current_rows)}")
    
    n = len(current_rows)
    
    # Loop through smaller values of dialect_sum and combine
    for j in range(1, stage_level):
        if j == 1:
            # Balance countries for dialect_sum = 1
            resampled_rows = balance_countries(n // 18, dataset)
        else:
            # Get rows for dialect_sum = j
            rows_with_sum_j = dataset[dataset["dialect_sum"] == j]
            
            if len(rows_with_sum_j) > n:
                resampled_rows = resample(
                    rows_with_sum_j, 
                    replace=True, 
                    n_samples=n, 
                    random_state=42
                )
            else:
                resampled_rows = rows_with_sum_j
        
        # Combine with current rows
        current_rows = pd.concat([current_rows, resampled_rows], ignore_index=True)
        print(f"Dialect_sum counts after combining with sum {j}:")
        print(current_rows['dialect_sum'].value_counts())
    
    return current_rows


def prepare_all_curriculum_stages(dataset, output_dir, stage_levels=None):
    """
    Prepare all curriculum learning stages and save them to CSV files.
    
    Args:
        dataset: Full dataset with 'dialect_sum' column
        output_dir: Directory to save stage CSV files
        stage_levels: List of stage levels to create. If None, uses default range.
        
    Returns:
        List of paths to created stage files
    """
    if stage_levels is None:
        stage_levels = list(range(1, 16)) + [18]
    
    stage_paths = []
    
    for stage_level in stage_levels:
        stage_data = create_curriculum_stage(dataset, stage_level)
        output_path = f"{output_dir}/stage_{stage_level}.csv"
        stage_data.to_csv(output_path, index=False)
        stage_paths.append(output_path)
        print(f"Saved stage {stage_level} to {output_path}\n")
    
    return stage_paths


def load_and_prepare_dataset(dataset_path, computed_filter=True):
    """
    Load dataset and prepare it with dialect_sum column.
    
    Args:
        dataset_path: Path to the dataset CSV file
        computed_filter: Whether to filter for rows with 'Computed' == 'yes'
        
    Returns:
        Processed DataFrame with dialect_sum column
    """
    dataset = pd.read_csv(dataset_path)
    
    if computed_filter and 'Computed' in dataset.columns:
        dataset = dataset[dataset['Computed'] == 'yes']
    
    # Calculate dialect_sum if not present
    if 'dialect_sum' not in dataset.columns:
        dataset['dialect_sum'] = dataset[DIALECT_COLUMNS].sum(axis=1)
    
    return dataset
