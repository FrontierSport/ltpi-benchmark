import torch
import numpy as np
from typing import Any


def classify_other_by_position(x: float, y: float) -> str:
    """
    Classifies a player as goalkeeper or referee based on x-coordinate.

    Parameters
    ----------
    x : float
        X-coordinate of the player on the field.
    y : float
        Y-coordinate of the player on the field (unused).

    Returns
    -------
    str
        'left_goalkeeper', 'right_goalkeeper', or 'referee' based on position.
    """
    if x <= -40:
        return "left_goalkeeper"
    if x >= 40:
        return "right_goalkeeper"
    return "referee"
    

def predict_class(
    image_patches: list[Any],
    model: torch.nn.Module,
    transform: Any,
    device: torch.device
) -> list[str]:
    """
    Predicts class labels for a list of image patches using a PyTorch model.

    Parameters
    ----------
    image_patches : list
        List of image patches to classify.
    model : torch.nn.Module
        Trained PyTorch model for classification.
    transform : callable
        Transformation function to apply to each image patch.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    list of str
        Predicted class labels for each image patch ('first_team', 'second_team', 'other').
    """
    if not image_patches:
        return []

    idx2label = {0: "first_team", 1: "second_team", 2: "other"}

    tensors = [transform(patch) for patch in image_patches]
    batch_tensor = torch.stack(tensors).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(batch_tensor)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        pred_idxs = logits.argmax(dim=-1).tolist()
        pred_probs = torch.nn.functional.softmax(logits).max(dim=-1).values
        
    return [idx2label.get(idx, "unknown") for idx in pred_idxs], pred_probs.tolist()


def classify_other_by_hsv_and_position(
    mean_hsv: np.ndarray,
    abs_x: float,
    abs_y: float,
    profiles: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    distance_threshold: float = 30.0,
    w_hsv: float = 0.7,
    w_pos: float = 0.3
) -> str:
    """
    Classifies a player based on HSV color and position, falling back to position-only.

    Parameters
    ----------
    mean_hsv : np.ndarray
        Mean HSV color of the player.
    abs_x : float
        X-coordinate on the field.
    abs_y : float
        Y-coordinate on the field.
    profiles : dict
        Dictionary mapping class names to lists of (HSV, position) samples.
    distance_threshold : float, optional
        Maximum allowed distance for HSV+position matching, by default 30.0.
    w_hsv : float, optional
        Weight for HSV distance, by default 0.7.
    w_pos : float, optional
        Weight for positional distance, by default 0.3.

    Returns
    -------
    str
        Predicted class name.
    """
    best_class = None
    best_dist = float("inf")
    current_pos = np.array([abs_x, abs_y], dtype=float)

    for cls_name, samples in profiles.items():
        if not samples:
            continue

        dists = []
        for hsv_ref, pos_ref in samples:
            hsv_ref = np.asarray(hsv_ref, dtype=float)
            pos_ref = np.asarray(pos_ref, dtype=float)

            d_hsv = np.linalg.norm(mean_hsv - hsv_ref)
            d_pos = np.linalg.norm(current_pos - pos_ref)
            dists.append(w_hsv * d_hsv + w_pos * d_pos)

        avg_dist = float(np.mean(dists))
        if avg_dist < best_dist:
            best_dist = avg_dist
            best_class = cls_name

    if best_class is not None and best_dist < distance_threshold:
        return best_class
    return classify_other_by_position(abs_x, abs_y)