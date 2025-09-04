from typing import Any
import pandas as pd
import anndata as ad

def load_data(file_path: str) -> Any:
    """
    Load and preprocess data from an .h5ad file.

    Parameters:
    - file_path: str, path to the .h5ad file.

    Returns:
    - ad.AnnData object containing the loaded data.
    """
    data = ad.read_h5ad(file_path)
    # Perform any necessary preprocessing here
    return data

def preprocess_data(data: ad.AnnData) -> ad.AnnData:
    """
    Preprocess the loaded AnnData object.

    Parameters:
    - data: ad.AnnData, the data to preprocess.

    Returns:
    - ad.AnnData object after preprocessing.
    """
    # Example preprocessing steps
    data.raw = data
    data = data.raw.to_adata()  # Convert raw data to AnnData if needed
    # Additional preprocessing can be added here
    return data