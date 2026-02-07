"""
Data Utilities for Curriculum Learning
Handles dataset balancing and stage preparation
"""

import pandas as pd
from sklearn.utils import resample
from pathlib import Path
import numpy as np
# Dialect column definitions
DIALECT_COLUMNS = [
    'Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
    'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
    'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen'
]
ALDI_THRESHOLDS = {
    4: 0.77,  # Stage 1: only highest scores
    3: 0.44,  # Stage 2: 0.44–1.0 (stages 2 + 1)
    2: 0.11,  # Stage 3: 0.11–1.0 (3 + 2 + 1)
    1: 0.0,   # Stage 4: 0.00–1.0 (all)
}


def balance_countries(n_samples, dataset, seed=42):
    """
    Balance dataset across all dialect countries for single-dialect samples.
    
    Args:
        n_samples: Number of samples to draw for each dialect
        dataset: DataFrame containing the full dataset
        
    Returns:
        DataFrame with balanced samples across all dialects
    """
    # Nothing to sample when n_samples is zero/None; keep contract by returning
    # an empty frame with matching columns.
    if n_samples is None or n_samples < 1:
        return pd.DataFrame(columns=dataset.columns)

    n_samples = int(n_samples)

    # Filter rows with dialect_sum equal to 1
    rows_with_sum_1 = dataset[dataset['dialect_sum'] == 1]
    
    # Create an empty DataFrame to store balanced rows
    balanced_rows = pd.DataFrame()
    
    # Balance rows across all dialects
    for dialect in DIALECT_COLUMNS:
        # Select rows where the specific dialect column is 1
        dialect_rows = rows_with_sum_1[rows_with_sum_1[dialect] == 1]

        # Skip dialects with no available rows
        if dialect_rows.empty:
            continue
        
        # Check if enough rows are available for resampling
        if len(dialect_rows) >= n_samples:
            resampled_rows = resample(
                dialect_rows, 
                replace=False, 
                n_samples=n_samples, 
                random_state=seed
            )
        else:
            # Use replacement if there are not enough rows
            resampled_rows = resample(
                dialect_rows, 
                replace=True, 
                n_samples=n_samples, 
                random_state=seed
            )
        
        balanced_rows = pd.concat([balanced_rows, resampled_rows], ignore_index=True)
    
    return balanced_rows

def assign_aldi_bin(score: float) -> int:
    """
    Map a continuous aldi_score in [0, 1] to a base bin 1-4.

    Bin 4: [0.77, 1.00]
    Bin 3: [0.44, 0.77)
    Bin 2: [0.11, 0.44)
    Bin 1: [0.00, 0.11)
    """
    if pd.isna(score):
        return np.nan

    # thresholds sorted high → low
    for bin_id, thr in sorted(ALDI_THRESHOLDS.items(),
                              key=lambda kv: kv[1],
                              reverse=True):
        if score >= thr:
            return bin_id

    return np.nan

def create_cl_aldi_stage(dataset: pd.DataFrame, stage_id: int, stage_levels) -> pd.DataFrame:
    """
    Cumulative curriculum stage based on ALDI *bins* in custom order 1,4,2,3.

    Base bins (by aldi_score):
        Bin 4: 0.77-1.00
        Bin 3: 0.44-0.77
        Bin 2: 0.11-0.44
        Bin 1: 0.00-0.11

    Args:
        dataset: DataFrame containing an 'aldi_score' column with scores in [0, 1]
        stage_id: Curriculum stage index in {1, 2, 3, 4}

    Returns:
        DataFrame for the given cumulative ALDI stage.
    """
    if 'aldi_score' not in dataset.columns:
        raise ValueError("Dataset must contain an 'aldi_score' column.")

    if stage_id not in {1, 2, 3, 4}:
        raise ValueError(f"stage_id must be in {{1, 2, 3, 4}}, got {stage_id}.")

    # Make sure we have a base-bin column
    if 'aldi_bin' not in dataset.columns:
        dataset = dataset.copy()
        dataset['aldi_bin'] = dataset['aldi_score'].apply(assign_aldi_bin)

    # Which bins are included at this curriculum step?
    idx = stage_levels.index(stage_id)

    bins_for_stage = stage_levels[:idx+1]

    stage_data = dataset[dataset['aldi_bin'].isin(bins_for_stage)].copy()

    print(f"Stage {stage_id}: using bins {bins_for_stage}")
    print(f"  -> {len(stage_data)} samples")
    print("  Bin counts in this stage:")
    print(stage_data['aldi_bin'].value_counts().sort_index())

    return stage_data



def create_cl_cardinality_stage(dataset, stage_level, stage_levels, seed=42):
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
    idx = stage_levels.index(stage_level)

    for j in range(idx):
        if stage_levels[j] == 1:
            # Balance countries for dialect_sum = 1
            resampled_rows = balance_countries(n // 18, dataset, seed=seed)
        else:
            # Get rows for dialect_sum = j
            rows_with_sum_j = dataset[dataset["dialect_sum"] == stage_levels[j]]
            
            if len(rows_with_sum_j) > n:
                resampled_rows = resample(
                    rows_with_sum_j, 
                    replace=True, 
                    n_samples=n, 
                    random_state=seed
                )
            else:
                resampled_rows = rows_with_sum_j
        
        # Combine with current rows
        current_rows = pd.concat([current_rows, resampled_rows], ignore_index=True)
    print(current_rows['dialect_sum'].value_counts())
    
    return current_rows


def prepare_all_curriculum_stages(dataset, output_dir, stage_levels=None, cl_method=1, seed=42):
    """
    Prepare all curriculum learning stages and save them to CSV files.
    
    Args:
        dataset: Full dataset with 'dialect_sum' column
        output_dir: Directory to save stage CSV files
        stage_levels: List of stage levels to create. If None, uses default range.
        
    Returns:
        List of paths to created stage files
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    
    stage_paths = []
    
    for stage_level in stage_levels:
        if cl_method == 1:
            stage_data = create_cl_cardinality_stage(dataset, stage_level, stage_levels, seed=seed)
        else:
            stage_data = create_cl_aldi_stage(dataset, stage_level, stage_levels)
        output_path = f"{output_dir}/stage_{stage_level}.csv"
        stage_data.to_csv(output_path, index=False)
        stage_paths.append(output_path)
        print(f"Saved stage {stage_level} to {output_path}\n")
    
    return stage_paths


def load_and_prepare_dataset(dataset_path, computed_filter=False):
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
