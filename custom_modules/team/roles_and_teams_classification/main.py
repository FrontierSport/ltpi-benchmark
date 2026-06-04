import torch
from collections import deque
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import logging

from .src.parsing.config_parser import load_config, find_by_sngs
from .src.parsing.df_parsers import (
    extract_metadata_info,
    filter_detections_by_imageid,
    extract_sngs_id
)
from .src.utils import build_remapping, compute_mean_hsv
from .src.modeling.mobilenet import MobileNetEmbedding
from .src.modeling.postprocess import predict_class, classify_other_by_hsv_and_position

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_device() -> torch.device:
    """
    Prepares a PyTorch device, preferring GPU if available.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a MobileNetEmbedding model with pretrained weights.

    Parameters
    ----------
    model_path : str
        Path to the pretrained model file.
    device : torch.device
        Device to load the model on.

    Returns
    -------
    torch.nn.Module
        Model loaded with pretrained weights and moved to the specified device.
    """
    model = MobileNetEmbedding(embedding_size=128, pretrained=True, freeze_layers=True, num_classes=3)
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)
    model = model.to(device)
    # logger.info("Model successfully loaded to device %s", device)
    return model


def get_transform() -> transforms.Compose:
    """
    Returns a torchvision transform pipeline for image preprocessing.

    The pipeline resizes images, converts them to tensors, and normalizes them.

    Returns
    -------
    torchvision.transforms.Compose
        Composed transform for image preprocessing.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 64)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def initialize_other_profiles(max_profiles: int = 100) -> dict[str, deque]:
    """
    Initializes storage for previous detections of goalkeepers and referees.

    Parameters
    ----------
    max_profiles : int, optional
        Maximum number of entries to store per profile, by default 100.

    Returns
    -------
    dict[str, deque]
        Dictionary with deques for 'right_goalkeeper', 'left_goalkeeper', and 'referee'.
    """
    return {
        "right_goalkeeper": deque(maxlen=max_profiles),
        "left_goalkeeper": deque(maxlen=max_profiles),
        "referee": deque(maxlen=max_profiles)
    }


def process_frame(image_path: str, target_df: pd.DataFrame):
    """
    Processes a frame to extract crops, bounding boxes, and absolute positions.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    target_df : pandas.DataFrame
        DataFrame containing bounding box and position information for each object.

    Returns
    -------
    tuple
        - List of PIL.Image crops.
        - List of bounding boxes in (x1, y1, x2, y2) format.
        - List of absolute positions as [x, y].
    """
    image = Image.open(image_path).convert("RGB")
    crops, bboxes, abs_poses = [], [], []

    for _, row in target_df.iterrows():
        bbox = tuple(row["bbox_ltwh"])
        x, y, w, h = bbox
        bbox = (x, y, x + w, y + h)
        crop = image.crop(bbox)
        crops.append(crop)
        bboxes.append(bbox)
        abs_poses.append([
            row["bbox_pitch"]["x_bottom_middle"],
            row["bbox_pitch"]["y_bottom_middle"]
        ])

    return crops, bboxes, abs_poses


def main(metadata_df: pd.DataFrame, detections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline for classifying players and other roles in video frames.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        DataFrame containing frame metadata.
    detections_df : pandas.DataFrame
        DataFrame containing detection results per frame.

    Returns
    -------
    pandas.DataFrame
        Updated detections DataFrame with a new 'classified_role' column.
    """
    device = prepare_device()
    sngs_id = extract_sngs_id(metadata_df)
    config = load_config("src/config.yaml")
    config_main_info = find_by_sngs(config, sngs_id)
    logger.info("Device and config prepared!")

    model_path = config_main_info["model_params_id"]
    teams_remapping = build_remapping(config_main_info["remapping"], config_main_info["time_now"])
    model = load_model(model_path, device)
    transform = get_transform()
    logger.info("Model and transform pipeline ready!")

    other_hsv_profiles = initialize_other_profiles()

    detections_df["classified_role"] = None

    for idx in tqdm(range(len(metadata_df)), desc="Processing frames"):
        metadata_info = extract_metadata_info(metadata_df, idx)
        target_df = filter_detections_by_imageid(detections_df, metadata_info["image_id"])

        if target_df.empty:
            continue

        crops, _, abs_poses = process_frame(metadata_info["image_path"], target_df)
        labels = predict_class(crops, model, transform, device) if crops else []

        for i, (label, abs_pose, crop) in enumerate(zip(labels, abs_poses, crops)):
            abs_x, abs_y = abs_pose

            if label in ["first_team", "second_team"]:
                label = teams_remapping[label]
            elif label == "other":
                mean_hsv = compute_mean_hsv(crop)
                refined_label = classify_other_by_hsv_and_position(
                    mean_hsv, abs_x, abs_y, other_hsv_profiles
                )
                other_hsv_profiles[refined_label].append((mean_hsv, (abs_x, abs_y)))
                label = refined_label

            detections_df.loc[target_df.index[i], "classified_role"] = label
    logger.info("Processing completed for all frames.")
    return detections_df
