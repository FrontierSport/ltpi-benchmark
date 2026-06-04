import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import cv2

from .reid_dataset import Reid

log = logging.getLogger(__name__)


class ReidVideoDataset(Reid):
    def __init__(
        self,
        dataset_path: str,
        source_path: str,
        nvid: int = -1,
        vids_dict: list = None,
        fps: int = None,
        video_filename: str = "video.mp4",
        force_extract: bool = False,
        *args,
        **kwargs,
    ):
        self.source_path = Path(source_path)
        self.video_filename = video_filename
        self.force_extract = force_extract

        assert (
            self.source_path.exists()
        ), f"'{self.source_path}' directory does not exist. Please check the source_path."

        # Prepare cache structure before calling parent __init__
        prepare_cache(self.source_path, Path(dataset_path), video_filename, force_extract)

        # Initialize parent class with cached dataset path
        super().__init__(
            dataset_path=dataset_path,
            nvid=nvid,
            vids_dict=vids_dict,
            fps=fps,
            *args,
            **kwargs,
        )

def prepare_cache(source_path, cache_path, video_filename, force_extract):
    """Create cache structure with symlinks and extracted frames."""
    for split in ["train", "valid", "test", "challenge"]:
        source_split = source_path / split
        if not source_split.exists():
            continue

        cache_split = cache_path / split
        cache_split.mkdir(parents=True, exist_ok=True)

        video_folders = sorted([
            f for f in source_split.iterdir()
            if f.is_dir()
        ])

        for video_folder in video_folders:
            prepare_video_folder(video_folder, cache_split, video_filename, force_extract)


def prepare_video_folder(source_folder, cache_split, video_filename, force_extract):
    """Prepare single video folder in cache."""
    folder_name = source_folder.name
    cache_folder = cache_split / folder_name
    cache_folder.mkdir(parents=True, exist_ok=True)

    # Create symlinks for annotation files
    create_annotation_symlinks(source_folder, cache_folder)

    # Extract frames if needed
    video_path = source_folder / video_filename
    img_folder = cache_folder / "img1"

    if video_path.exists():
        if should_extract_frames(img_folder, force_extract):
            extract_frames(video_path, img_folder)
        else:
            log.debug(f"Using cached frames for {folder_name}")
    else:
        # Check if img1 already exists in source (original format)
        source_img = source_folder / "img1"
        if source_img.exists() and not img_folder.exists():
            img_folder.symlink_to(source_img.resolve())
            log.debug(f"Linked existing img1 for {folder_name}")
        elif not img_folder.exists() or not any(img_folder.glob("*.jpg")):
            raise FileNotFoundError(
                f"No video or frames found for '{folder_name}'.\n"
                f"  Expected video: {video_path}\n"
                f"  Or frames in: {source_img}\n"
                f"Please add the video file or extracted frames."
            )


def create_annotation_symlinks(source_folder, cache_folder):
    """Create symlinks for annotation files."""
    annotation_files = [
        "roster.csv",
        "substitutions.csv",
        "subtracks.csv"
    ]

    for filename in annotation_files:
        source_file = source_folder / filename
        cache_file = cache_folder / filename

        if source_file.exists() and not cache_file.exists():
            cache_file.symlink_to(source_file.resolve())
            log.debug(f"Created symlink: {cache_file} -> {source_file}")


def should_extract_frames(img_folder, force_extract):
    """Check if frames need to be extracted."""
    if force_extract:
        return True

    if not img_folder.exists():
        return True

    # Check if folder has any jpg files
    jpg_files = list(img_folder.glob("*.jpg"))
    return len(jpg_files) == 0


def extract_frames(video_path, output_folder, frame_format="{:06d}.jpg"):
    """Extract all frames from video file."""
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Extracting {total_frames} frames from {video_path.name}")

    with tqdm(total=total_frames) as pb:
        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = output_folder / frame_format.format(frame_idx)
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1
            pb.update()

    cap.release()
    log.info(f"Extracted {frame_idx - 1} frames to {output_folder}")
