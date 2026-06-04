import pandas as pd
from typing import Dict, Any, Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def extract_metadata_info(metadata_frame: pd.DataFrame, index: int) -> Dict[str, Any]:
    """
    Extracts metadata (image_id, image_path, is_labeled, nframes) for a specific index from a DataFrame.

    Parameters
    ----------
    metadata_frame : pandas.DataFrame
        DataFrame containing metadata.
    index : int
        Row index to extract.

    Returns
    -------
    dict[str, Any]
        Dictionary with image ID, path, label status, and frame count.
    """
    row_data = metadata_frame.iloc[index]
        
    metadata = {
        "image_id": str(row_data["id"]),
        "image_path": str(row_data["file_path"]),
        "is_labeled": bool(row_data["is_labeled"]),
        "nframes": int(row_data["nframes"])
    }
    
    if not metadata["is_labeled"]:
        logger.warning("Index=%d is unlabeled.", index)
    return metadata


def filter_detections_by_imageid(detections_df: pd.DataFrame, target_image_id: str) -> Union[pd.DataFrame, None]:
    """
    Filters detections for a specific image ID.

    Parameters
    ----------
    detections_df : pandas.DataFrame
        DataFrame containing detection results.
    target_image_id : str
        Image ID to filter by.

    Returns
    -------
    pandas.DataFrame or None
        Filtered detections for the given image ID, or None if not found.
    """
    if "image_id" not in detections_df.columns:
        raise ValueError("Column 'image_id' is missing in the DataFrame!")

    filtered_df = detections_df[detections_df["image_id"] == target_image_id]
    
    if filtered_df.empty:
        logger.warning("No detections found for image ID: %s", target_image_id)
        return None
    return filtered_df


def extract_sngs_id(metadata_frame: pd.DataFrame) -> str:
    """
    Extracts the SNGS ID from the metadata DataFrame.

    Parameters
    ----------
    metadata_frame : pandas.DataFrame
        DataFrame containing the 'file_path' column.

    Returns
    -------
    str
        The SNGS ID extracted from the file path.
    """
    if metadata_frame.empty or "file_path" not in metadata_frame:
        raise ValueError("The metadata_frame must contain a non-empty 'file_path' column")

    filepath = str(metadata_frame["file_path"].iloc[0]).strip()
    parts = filepath.split('/')
    return parts[-3]