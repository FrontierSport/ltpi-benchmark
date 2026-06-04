import logging

import numpy as np
import pandas as pd
import torch

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.attribute_voting import select_highest_voted_att

log = logging.getLogger(__name__)


class MajorityVoteTracklet(VideoLevelModule):
    version = '1.0'
    input_columns = ['track_id']
    output_columns = []

    def __init__(self, cfg, device, tracking_dataset=None):
        self.attributes = cfg.attributes
        for attribute in self.attributes:
            self.input_columns.append(f"{attribute}_detection")
            self.input_columns.append(f"{attribute}_confidence")
            self.output_columns.append(attribute)

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        detections[self.output_columns] = np.nan

        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            for attribute in self.attributes:
                attribute_detection = tracklet[f"{attribute}_detection"]
                attribute_confidence = tracklet[f"{attribute}_confidence"]
                attribute_value = [
                    select_highest_voted_att(attribute_detection, attribute_confidence)
                ] * len(tracklet)
                detections.loc[tracklet.index, attribute] = attribute_value

        return detections
