import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
# from centroids_reid.config import cfg as centroids_cfg
# from centroids_reid.custom_ctl_model import ReIDBackbone
# from centroids_reid.datasets.transforms import ReidTransforms
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from strhub.models.utils import load_from_checkpoint
from tqdm import tqdm

from custom_modules.jersey.jersey_pipeline import helpers
from custom_modules.jersey.jersey_pipeline import legibility_classifier as lc
from custom_modules.jersey.jersey_pipeline.jersey_number_dataset import \
    data_transforms
from custom_modules.jersey.jersey_pipeline.networks import (
    LegibilityClassifier, LegibilityClassifier34,
    LegibilityClassifierTransformer, LegibilitySimpleClassifier)
from custom_modules.jersey.jersey_pipeline.str import run_inference_tracklab
# from tracklab.pipeline import VideoLevelModule
from tracklab.pipeline import DetectionLevelModule
# from tracklab.utils.download import process_dataset_path, process_model_path
from tracklab.utils.collate import Unbatchable, default_collate

tracklab_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")

log = logging.getLogger(__name__)


@dataclass
class CFG:
    # centroids_model_conf: str
    # centroids_model_weights: str
    # centroids_threshold: float
    # centroids_rounds: int
    legibility_model: str
    legibility_model_arch: str
    arch: str
    mode: str
    pose_config: str
    pose_checkpoint: str
    str_checkpoint: str
    charset_test: str


class JerseyPipeline(DetectionLevelModule):
    '''
    Differences from usual JerseyPipeline:
    1) Refactored as DetectionLevelModule so it can run on long videos without memory
       errors
    2) Centroids reid filtering is removed as it is not implemented in current 
       production pipeline and requires full tracks (which only available in 
       VideoLevelModule)
    '''
    version = "1.0"
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, config, batch_size, num_cores, device, tracking_dataset):
        super().__init__(batch_size)
        self.device = device
        self.num_cores = num_cores

        self.num2split = {"1": "train", "2": "valid", "3": "test", "4": "challenge"}
        self.dataset_path = tracking_dataset.dataset_path
        self.batch_size = batch_size
        # self.centroids_threshold = config.centroids_threshold
        # self.centroids_rounds = config.centroids_rounds

        # load Centroid Reid model
        # centroids_cfg.merge_from_file(
        #     os.path.join(tracklab_dir, config.centroids_model_conf)
        # )
        # opts = [
        #     "MODEL.PRETRAIN_PATH",
        #     config.centroids_model_weights,
        #     "MODEL.PRETRAINED",
        #     True,
        #     "TEST.ONLY_TEST",
        #     True,
        #     "MODEL.RESUME_TRAINING",
        #     False,
        # ]
        # centroids_cfg.merge_from_list(opts)

        # self.model_centroids = ReIDBackbone(centroids_cfg.MODEL.PRETRAIN_PATH)

        # self.model_centroids = self.model_centroids.to(device)
        # self.model_centroids.eval()

        # load legibility classification model
        state_dict = torch.load(
            config.legibility_model, map_location=device
        )
        if config.legibility_model_arch == "resnet18":
            self.model_ft = LegibilityClassifier()
        elif config.legibility_model_arch == "vit":
            self.model_ft = LegibilityClassifierTransformer()
        else:
            self.model_ft = LegibilityClassifier34()

        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        self.model_ft.load_state_dict(state_dict)
        self.model_ft = self.model_ft.to(device)
        self.model_ft.eval()

        # inititalize legibility model transformations
        self.transform = data_transforms[config.mode][config.arch]

        # initialize pose model
        # print(os.path.join(tracklab_dir, config.pose_config))
        # raise Exception()
        self.pose_model = init_pose_model(
            os.path.join(tracklab_dir, config.pose_config),
            config.pose_checkpoint,
            device=device,
        )

        # initialize str model
        self.model_str = (
            load_from_checkpoint(
                config.str_checkpoint,
                charset_test=config.charset_test,
            )
            .eval()
            .to(device)
        )
        self.hp = self.model_str.hparams

        # transforms_base = ReidTransforms(centroids_cfg)
        # self.val_transforms = transforms_base.build_transforms(is_train=False)

    def crop_image(self, image_id: str, detections_tmp, val_transforms, metadatas):
        dets_by_im = detections_tmp.loc[detections_tmp["image_id"] == image_id]

        image = cv2.imread(
            metadatas.loc[image_id]['file_path']
        )

        crops = []
        for i, row in dets_by_im.iterrows():
            bbox = row["bbox_ltwh"]
            l, t, r, b = (
                np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                .round()
                .astype(int)
            )
            crop = image[t:b, l:r]

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)

            crop = crop[:, :, ::-1].astype(np.float32)
            crop /= 255.0
            crop = np.transpose(crop, (2, 0, 1))
            crop = torch.from_numpy(crop)

            crops.append(
                {"index": i, "crop": crop, "crop_transformed": val_transforms(crop)}
            )

        return crops
    
    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r].astype(np.float32)
        crop /= 255.0
        crop = np.transpose(crop, (2, 0, 1))
        crop = torch.from_numpy(crop)
        crop = Unbatchable([crop])

        batch = {
            "img": crop,
            "detection_id": detection.name
        }

        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # initialize result columns with "number not recognized"
        detections["jersey_number_confidence"] = 0
        detections["jersey_number_detection"] = None

        results = lc.run_tracklab(
            batch['img'], self.model_ft, transform=self.transform, threshold=0.9
        )
        legible = list(np.nonzero(results))[0]
        filtered_batch = [
                {"batch_id": batch['detection_id'][i].item(), "img": batch['img'][i]}
                for i in legible
            ]
        
        # find pose keypoints
        results = self.find_keypoints(filtered_batch)

        # crop torso from pose keypoints
        saved_crops = helpers.generate_crops_tracklab(results)

        # run jersey number recognition model
        results = run_inference_tracklab(
            self.model_str, saved_crops, self.hp.img_size
        )

        for res in results:
            # if str model return empty number for some reason
            if res["label"] != "":
                total_prob = 1
                for x in res["confidence"][:-1]:
                    total_prob = total_prob * float(x)
                detections.loc[res["batch_id"], "jersey_number_confidence"] = (
                    total_prob
                )
                detections.loc[res["batch_id"], "jersey_number_detection"] = res[
                    "label"
                ]
            else:
                detections.loc[res["batch_id"], "jersey_number_confidence"] = 0
                detections.loc[res["batch_id"], "jersey_number_detection"] = None

        return detections

    def find_keypoints(self, filtered_batch):
        results = []
        dataset = self.pose_model.cfg.data["test"]["type"]
        dataset_info = DatasetInfo(
            self.pose_model.cfg.data["test"].get("dataset_info", None)
        )

        for item in filtered_batch:
            crop = np.transpose(item["img"].numpy(), (1, 2, 0))
            crop *= 255
            crop = crop[:, :, ::-1]
            crop = crop.astype(np.uint8)

            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                crop,
                [{"bbox": np.array([0, 0, crop.shape[1], crop.shape[0]])}],
                bbox_thr=None,
                format="xywh",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None,
            )

            results.append(
                {
                    "batch_id": item["batch_id"],
                    "image": crop,
                    "keypoints": pose_results[0]["keypoints"].tolist(),
                }
            )

        return results
