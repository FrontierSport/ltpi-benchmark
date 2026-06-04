import logging

import pandas as pd
import torch
from ast import literal_eval
from PIL import Image

from tracklab.pipeline.videolevel_module import VideoLevelModule
from custom_modules.team.roles_and_teams_classification.src.utils import (
    build_remapping,
    compute_mean_hsv,
)
from custom_modules.team.roles_and_teams_classification.main import (
    load_model,
    get_transform,
    initialize_other_profiles,
)

from custom_modules.team.roles_and_teams_classification.src.modeling.postprocess import (
    predict_class,
    classify_other_by_hsv_and_position,
)

log = logging.getLogger(__name__)


def parse_config(config) -> dict:

    matches_list = config.get("matches_ids", [])

    transformed_matches = {}
    keys_to_parse = [
        "folder",
        "sngs_inside",
        "remapping",
        "sngs_times",
        "pure_colors",
        "model_params_id",
    ]

    for match in matches_list:
        match_id = match.get("id")

        if match_id is None:
            continue

        for key in keys_to_parse:
            if key in match and isinstance(match[key], str):
                try:
                    match[key] = literal_eval(match[key])
                except (ValueError, SyntaxError):
                    pass

        transformed_matches[match_id] = match
    return transformed_matches


def find_by_sngs(config_data: dict[int, dict], sngs: str) -> dict:
    """
    Finds remapping dictionary, model id for ClearML and gametime in configuration file.

    Parameters
    ----------
    config_data : dict[int, dict]
        Сonfigurations with SNGSs slit and etc.
    sngs : str
        Target SNGS key in format 'SNGS-<NUM>'.

    Returns
    -------
    dict
        Info with remapping, model params ID, and time, or empty if not found.
    """
    for match_id, match_info in config_data.items():
        sngs_list = match_info.get("sngs_inside", [])
        sngs_times = match_info.get("sngs_times", [])
        if (sngs in sngs_list) and (sngs in sngs_times):
            main_info = {
                "remapping": match_info.get("remapping"),
                "model_params_id": match_info.get("model_params_id"),
                "time_now": sngs_times[sngs],
            }
            return main_info
    raise ValueError(f"Model not found for {sngs}")


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
        if "bbox_pitch" in row and isinstance(row["bbox_pitch"], dict):
            abs_poses.append(
                [
                    row["bbox_pitch"]["x_bottom_middle"],
                    row["bbox_pitch"]["y_bottom_middle"],
                ]
            )
        else:
            abs_poses.append([None, None])

    return crops, bboxes, abs_poses


# mapping from model predictions to teams
pred2team = {
    "left-team": "left",
    "left_goalkeeper": "left",
    "right-team": "right",
    "right_goalkeeper": "right",
    "referee": None,
    None: None,
}

# mapping from model predictions to roles
pred2role = {
    "right-team": "player",
    "left-team": "player",
    "referee": "referee",
    "left_goalkeeper": "goalkeeper",
    "right_goalkeeper": "goalkeeper",
    None: None,
}


class PretrainedTeams(VideoLevelModule):
    version = "1.2"
    input_columns = ["track_id"]
    output_columns = [
        "team_confidence",
        "team_detection",
        "role_detection",
        "role_confidence",
    ]

    def __init__(self, cfg, device, batch_size, use_probs, use_pitch, **kwargs):
        super().__init__()

        self.config = parse_config(cfg)
        self.transform = get_transform()
        self.other_hsv_profiles = initialize_other_profiles()
        self.device = device
        self.use_probs = use_probs
        self.use_pitch = use_pitch
        if self.use_pitch:
            self.input_columns.append('bbox_pitch')

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        video_id = metadatas["video_id"].iloc[0]
        config_main_info = find_by_sngs(self.config, f"SNGS-{int(video_id):03}")

        model_path = config_main_info["model_params_id"]
        teams_remapping = build_remapping(
            config_main_info["remapping"], config_main_info["time_now"]
        )
        model = load_model(model_path, self.device)

        detections["team_detection"] = None
        detections["team_confidence"] = None
        detections["role_detection"] = None
        detections["role_confidence"] = None

        # Iterate over images
        for image_id, image_detections in detections.groupby("image_id"):
            image_path = metadatas.loc[image_id, "file_path"]

            if image_detections.empty:
                continue
            
            crops, _, abs_poses = process_frame(image_path, image_detections)
            labels, probs = (
                predict_class(crops, model, self.transform, self.device)
                if crops
                else []
            )

            assert len(labels) == len(crops) == len(probs)

            # Iterate over detections
            for i, (label, abs_pose, crop, prob) in enumerate(
                zip(labels, abs_poses, crops, probs)
            ):
                abs_x, abs_y = abs_pose

                if label in ["first_team", "second_team"]:
                    label = teams_remapping[label]
                    conf = prob if self.use_probs else 1
                elif label == "other" and self.use_pitch and abs_x is not None and abs_y is not None:
                    mean_hsv = compute_mean_hsv(crop)
                    refined_label = classify_other_by_hsv_and_position(
                        mean_hsv, abs_x, abs_y, self.other_hsv_profiles
                    )
                    self.other_hsv_profiles[refined_label].append(
                        (mean_hsv, (abs_x, abs_y))
                    )
                    label = refined_label
                    conf = prob if self.use_probs else 1
                else:
                    label = None
                    conf = 0

                detections.loc[image_detections.index[i], "team_detection"] = pred2team[
                    label
                ]
                detections.loc[image_detections.index[i], "team_confidence"] = conf
                detections.loc[image_detections.index[i], "role_detection"] = pred2role[
                    label
                ]
                detections.loc[image_detections.index[i], "role_confidence"] = conf

        return detections
