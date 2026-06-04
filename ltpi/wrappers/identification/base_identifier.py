import torch
import pandas as pd

from tracklab.pipeline import VideoLevelModule

import logging

log = logging.getLogger(__name__)

team2id = {'right': 0,
           'left': 1}


class BaseIdentifier(VideoLevelModule):
    version = "1.0aa"
    input_columns = [
        "track_id",
        "image_id",
        "jersey_number",
        "team",
    ]
    output_columns = ["pred_id"]

    def __init__(self, device, tracking_dataset):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections['pred_id'] = detections.apply(self.identify, axis=1)
        # detections['pred_id'] = detections['pred_id'].astype(int)

        return detections

    @staticmethod
    def identify(row):
        if pd.isna(row['jersey_number']) or pd.isna(row['team']):
            return None
        else:
            return team2id[row['team']] * 100 + int(row['jersey_number'])
