from typing import Dict

import cv2
import numpy as np


def build_remapping(remapping: Dict[str, Dict[str, str]], time_now: str) -> Dict[str, str]:
    """
    Builds a mapping from FS team names to real team names (RL) for a given time.

    Parameters
    ----------
    remapping : dict
        Dictionary mapping time keys to team name remappings.
    time_now : str
        Current time key to select the remapping.

    Returns
    -------
    dict
        Dictionary mapping FS team names to real team names (RL).
    """
    if time_now not in remapping:
        curr_remapping = remapping.get("1")
        if not curr_remapping or len(curr_remapping) < 2:
            raise ValueError("remapping['1'] must contain at least two elements.")

        rl_teams = list(curr_remapping.keys())
        fs_teams = list(curr_remapping.values())

        return {
            fs_teams[0]: rl_teams[1],
            fs_teams[1]: rl_teams[0],
        }

    curr_remapping = remapping[time_now]
    return {v: k for k, v in curr_remapping.items()}


def compute_mean_hsv(pil_img: np.ndarray) -> np.ndarray:
    """
    Computes the mean HSV color of a PIL image.

    Parameters
    ----------
    pil_img : PIL.Image.Image
        Input image in RGB format.

    Returns
    -------
    np.ndarray
        Mean HSV values as a 3-element array.
    """
    img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    return mean_hsv