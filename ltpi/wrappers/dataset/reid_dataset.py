import json
import logging
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from tracklab.datastruct import TrackingDataset, TrackingSet
from tracklab.utils.progress import progress
from tracklab.utils.coordinates import ltrb_to_ltwh

log = logging.getLogger(__name__)


class Reid(TrackingDataset):
    def __init__(
        self,
        dataset_path: str,
        nvid: int = -1,
        vids_dict: list = None,
        fps: int = None,
        *args,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)

        assert (
            self.dataset_path.exists()
        ), f"'{self.dataset_path}' directory does not exist. Please check the path or download the dataset following the instructions here: https://github.com/SoccerNet/sn-gamestate"

        sets = {}
        for split in ["train", "valid", "test", "challenge"]:
            if os.path.exists(self.dataset_path / split):
                sets[split] = load_set(
                    self.dataset_path / split, nvid, vids_dict.get(split, []), fps=fps
                )
            else:
                log.warning(
                    f"Warning: The '{split}' set does not exist in the SoccerNetGS dataset at '{self.dataset_path}'. "
                    f"Please check the path or download the dataset following the instructions here: https://github.com/soccerNet/sn-gamestate#manual-downloading-of-soccernet-gamestate"
                )

        # We pass 'nvid=-1', 'vids_dict=None' because video subsampling is already done in the load_set function
        super().__init__(self.dataset_path, sets, nvid=-1, vids_dict=None, *args, **kwargs)

    def process_trackeval_results(self, results, dataset_config, eval_config):
        combined_results = results["SUMMARIES"]["cls_comb_det_av"]
        combined_results["GS-HOTA"] = combined_results.pop("HOTA")
        # In all keys, replace the substring "HOTA" with "GS-HOTA"
        combined_results["GS-HOTA"] = {
            k.replace("HOTA", "GS-HOTA"): v
            for k, v in combined_results["GS-HOTA"].items()
        }
        log.info(
            f"SoccerNet Game State Reconstruction performance GS-HOTA = {combined_results['GS-HOTA']['GS-HOTA']}% (config: EVAL_SPACE={dataset_config['EVAL_SPACE']}, USE_JERSEY_NUMBERS={dataset_config['USE_JERSEY_NUMBERS']}, USE_TEAMS={dataset_config['USE_TEAMS']}, USE_ROLES={dataset_config['USE_ROLES']}, EVAL_DIST_TOL={dataset_config['EVAL_DIST_TOL']})"
        )
        log.info(
            f"Have a look at 'tracklab/tracklab/configs/dataset/soccernet_gs.yaml' for more details about the GS-HOTA metric and the evaluation configuration."
        )
        return combined_results

    def save_for_eval(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        save_folder: str,
        bbox_column_for_eval="bbox_ltwh",
        save_classes=False,
        is_ground_truth=False,
        save_zip=True,
    ):
        if is_ground_truth:
            return
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        detections = self.soccernet_encoding(detections.copy(), supercategory="object")
        camera_metadata = self.soccernet_encoding(
            image_metadatas.copy(), supercategory="camera"
        )
        pitch_metadata = self.soccernet_encoding(
            image_metadatas.copy(), supercategory="pitch"
        )
        predictions = pd.concat(
            [detections, camera_metadata, pitch_metadata], ignore_index=True
        )
        zf_save_path = save_path.parents[1] / f"{save_path.parent.name}.zip"
        for id, video in video_metadatas.iterrows():
            file_path = save_path / f"{video['name']}.json"
            video_predictions_df = predictions[
                predictions["video_id"] == str(id)
            ].copy()
            if not video_predictions_df.empty:
                video_predictions_df.sort_values(by="id", inplace=True)
                video_predictions = [
                    {
                        k: int(v) if k == "track_id" else v
                        for k, v in m.items()
                        if np.all(pd.notna(v))
                    }
                    for m in video_predictions_df.to_dict(orient="records")
                ]
                with file_path.open("w") as fp:
                    json.dump({"predictions": video_predictions}, fp, indent=2)
                if save_zip:
                    with zipfile.ZipFile(
                        zf_save_path, "a", compression=zipfile.ZIP_DEFLATED
                    ) as zf:
                        zf.write(
                            file_path, arcname=f"{save_path.name}/{file_path.name}"
                        )

    @staticmethod
    def soccernet_encoding(dataframe: pd.DataFrame, supercategory):
        dataframe["supercategory"] = supercategory
        dataframe = dataframe.replace({np.nan: None})
        if supercategory == "object":
            # Remove detections that don't have mandatory columns
            # Detections with no track_id will therefore be removed and not count as FP at evaluation
            # Костыль на случай, если столбца bbox_pitch нет в выходе пайплайна,
            # например, если в пайплайне только детекция и трекинг
            if "bbox_pitch" in dataframe.columns:
                dataframe.dropna(
                    subset=[
                        "track_id",
                        "bbox_ltwh",
                        "bbox_pitch",
                    ],
                    how="any",
                    inplace=True,
                )
            else:
                dataframe.dropna(
                    subset=[
                        "track_id",
                        "bbox_ltwh",
                    ],
                    how="any",
                    inplace=True,
                )
            dataframe = dataframe.rename(
                columns={"bbox_ltwh": "bbox_image", "jersey_number": "jersey"}
            )
            dataframe["track_id"] = dataframe["track_id"]
            dataframe["attributes"] = [
                {
                    "role": x.get("role"),
                    "jersey": x.get("jersey"),
                    "team": x.get("team"),
                }
                for n, x in dataframe.iterrows()
            ]
            dataframe["id"] = dataframe.index
            dataframe = dataframe[
                dataframe.columns.intersection(
                    [
                        "id",
                        "image_id",
                        "video_id",
                        "track_id",
                        "supercategory",
                        "category_id",
                        "attributes",
                        "bbox_image",
                        "bbox_pitch",
                    ]
                )
            ]

            dataframe["bbox_image"] = dataframe["bbox_image"].apply(
                transform_bbox_image
            )
        elif supercategory == "camera":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 6
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "01")
            dataframe = dataframe[
                dataframe.columns.intersection(
                    [
                        "id",
                        "image_id",
                        "video_id",
                        "supercategory",
                        "category_id",
                        "parameters",
                        "relative_mean_reproj",
                        "accuracy@5",
                    ]
                )
            ]
        elif supercategory == "pitch":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 5
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "00")
            dataframe = dataframe[
                dataframe.columns.intersection(
                    [
                        "id",
                        "image_id",
                        "video_id",
                        "supercategory",
                        "category_id",
                        "lines",
                    ]
                )
            ]
        dataframe["video_id"] = dataframe["video_id"].apply(str)
        dataframe["image_id"] = dataframe["image_id"].apply(str)
        dataframe["id"] = dataframe["id"].apply(str)
        dataframe = dataframe.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        return dataframe


def transform_bbox_image(row):
    row = row.astype(float)
    return {"x": row[0], "y": row[1], "w": row[2], "h": row[3]}


def video_dir_to_dfs(args):
    dataset_path = args["dataset_path"]
    video_folder = args["video_folder"]
    split = args["split"]
    split_id = ["train", "valid", "test", "challenge"].index(split) + 1
    detections_df = None
    video_folder_path = os.path.join(dataset_path, video_folder)
    if os.path.isdir(video_folder_path):
        # Read the gamestate.json file
        markup_path = os.path.join(video_folder_path, "subtracks.csv")
        # roster_path = os.path.join(video_folder_path, "roster.csv")
        markup_data = pd.read_csv(markup_path)
        # print(markup_data)
        # raise Exception()
        # roster_data = pd.read_csv(roster_path)
        markup_data['bbox_ltwh'] = markup_data.apply(lambda row: ltrb_to_ltwh([row['x1'], row['y1'], row['x2'], row['y2']]), axis=1)
        markup_data = markup_data.rename(columns={'subtrack_id': 'track_id'})
        
        det_cols = ['image_id', 'bbox_ltwh', 'track_id', 'gt_track_id', 'team', 'jersey_number']
        if 'visibility' in markup_data.columns:
            det_cols.append('visibility')
        detections_df = markup_data[det_cols]
        
        detections_df["video_id"] = video_folder
        detections_df['bbox_conf'] = 1.0
        detections_df['image_id'] = detections_df['image_id'].apply(lambda x: str(split_id) + f'{int(video_folder):03d}' + f"{x:06d}")

        detections_df["person_id"] = detections_df["gt_track_id"].astype(
                str
            ) + detections_df["video_id"].astype(str)

        img_folder_path = os.path.join(
            video_folder_path, 'img1'
        )

        video_metadata = {
            "id": video_folder,
            "name": video_folder,
            "nframes": len(os.listdir(img_folder_path)),
            # "roster_0": roster_data[roster_data['team'] == 0]['jersey_number'].tolist(),
            # "roster_1": roster_data[roster_data['team'] == 1]['jersey_number'].tolist()
        }

        img_metadata_df = pd.DataFrame(
            {
                "frame": list(range(len(os.listdir(img_folder_path)))),
                # "id": sorted(detections_df['image_id'].unique()),
                "id": [
                    str(split_id) + f'{int(video_folder):03d}' + filename.split('_')[-1].split('.')[0]
                    for filename in sorted(os.listdir(img_folder_path))
                ],
                "file_path": [
                    os.path.join(img_folder_path, filename)
                    for filename in sorted(os.listdir(img_folder_path))
                ],
                "video_id": video_folder,
            }
        )

        # Check that all images from detections is represented in metadatas
        # assert len(set(detections_df['image_id']).difference(set(img_metadata_df['id']))) == 0

        return {
            "video_metadata": video_metadata,
            "image_metadata": img_metadata_df,
            "detections": detections_df,
            "annotations_pitch_camera": {},
            "video_level_categories": {},
        }

def load_set(dataset_path, nvid=-1, vids_filter_set=None, fps=None):
    video_metadatas_list = []
    image_metadata_list = []
    annotations_pitch_camera_list = []
    detections_list = []
    categories_list = []
    split = os.path.basename(dataset_path)  # Get the split name from the dataset path
    video_list = os.listdir(dataset_path)
    video_list.sort()

    if vids_filter_set is not None and len(vids_filter_set) > 0:
        missing_videos = set(vids_filter_set) - set(video_list)
        if missing_videos:
            log.warning(
                f"Warning: The following videos provided in config 'dataset.vids_dict' do not exist in {split} set: {missing_videos}"
            )

        video_list = [video for video in video_list if video in vids_filter_set]

    if nvid > 0:
        video_list = video_list[:nvid]

    assert (
        len(video_list) != 0
    ), f"After applying filtering, no videos left in the '{split}' set, please fix the 'dataset.vids_dict' config."

    pool = Pool()
    args = [
        {
            "dataset_path": dataset_path,
            "video_folder": video_folder,
            "split": split,
            "fps": fps,
        }
        for video_folder in video_list
    ]
    for result in progress(
        pool.imap_unordered(video_dir_to_dfs, args),
        total=len(args),
        desc=f"Loading SoccerNetGS '{split}' set videos",
    ):
        if result is not None:
            video_metadatas_list.append(result["video_metadata"])
            image_metadata_list.append(result["image_metadata"])
            detections_list.append(result["detections"])
            annotations_pitch_camera_list.append(result["annotations_pitch_camera"])
            categories_list += result["video_level_categories"]

    # Concatenate dataframes
    video_metadata = pd.DataFrame(video_metadatas_list)
    image_metadata = pd.concat(image_metadata_list, ignore_index=True)
    detections = pd.concat(detections_list, ignore_index=True)

    detections["person_id"] = pd.factorize(detections["person_id"])[0]

    detections['role'] = 'player'

    # Use video_id, image_id, track_id as unique id
    detections = detections.sort_values(
        by=["video_id", "image_id", "gt_track_id"], ascending=[True, True, True]
    )
    detections["id"] = (
        detections["video_id"].astype(str)
        + "_"
        + detections["image_id"].astype(str)
        + "_"
        + detections["gt_track_id"].astype(str)
    )

    # print(len(detections))
    # print(len(detections['id'].unique()))
    # assert len(detections) == len(detections['id'].unique())

    image_gt = image_metadata.copy().set_index("id", drop=False)

    detections.set_index("id", drop=False, inplace=True)
    image_metadata.set_index("id", drop=False, inplace=True)
    video_metadata.set_index("id", drop=False, inplace=True)

    # Reorder columns in dataframes
    video_metadata_columns = [
        "name",
        "nframes",
    ]
    video_metadata_columns.extend(
        set(video_metadata.columns) - set(video_metadata_columns)
    )
    video_metadata = video_metadata[video_metadata_columns]
    image_metadata_columns = [
        "frame",
        "file_path",
    ]
    image_metadata_columns.extend(
        set(image_metadata.columns) - set(image_metadata_columns)
    )
    image_metadata = image_metadata[image_metadata_columns]
    detections_column_ordered = [
        "image_id",
        "video_id",
        "track_id",
        "bbox_ltwh",
        "bbox_conf",
        "person_id"
    ]
    if 'visibility' in detections.columns:
        detections_column_ordered.append('visibility')
    detections_column_ordered.extend(
        set(detections.columns) - set(detections_column_ordered)
    )
    detections = detections[detections_column_ordered]
    detections["bbox_conf"] = 1

    return TrackingSet(
        video_metadata,
        image_metadata,
        detections,
        image_gt,
    )
