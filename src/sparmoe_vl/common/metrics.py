"""Cross-modal retrieval metrics for the two main experiment tables."""

from typing import Dict, Sequence

import torch
from torch import Tensor


def retrieval_recalls(
    image_features: Tensor,
    text_features: Tensor,
    caption_image_indices: Sequence[int],
) -> Dict[str, float]:
    """Compute bidirectional COCO recall at 1, 5, and 10."""

    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("feature tensors must have shape [samples, embedding_dim]")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("image and text feature dimensions must match")
    if text_features.shape[0] != len(caption_image_indices):
        raise ValueError("caption_image_indices must contain one index per text")
    if image_features.shape[0] == 0 or text_features.shape[0] == 0:
        raise ValueError("retrieval features must not be empty")

    image_indices = torch.as_tensor(caption_image_indices, dtype=torch.long)
    if torch.any(image_indices < 0) or torch.any(image_indices >= image_features.shape[0]):
        raise ValueError("caption image indices are out of range")
    counts = torch.bincount(image_indices, minlength=image_features.shape[0])
    if torch.any(counts == 0):
        raise ValueError("every image must have at least one ground-truth caption")

    similarities = image_features.float() @ text_features.float().T
    results = {}
    for k in (1, 5, 10):
        image_to_text = 0
        for image_index in range(image_features.shape[0]):
            predicted = similarities[image_index].argsort(descending=True)[:k]
            if torch.any(image_indices[predicted] == image_index):
                image_to_text += 1

        text_to_image = 0
        for text_index, image_index in enumerate(image_indices.tolist()):
            predicted = similarities[:, text_index].argsort(descending=True)[:k]
            if torch.any(predicted == image_index):
                text_to_image += 1

        results[f"I2T_R{k}"] = 100.0 * image_to_text / image_features.shape[0]
        results[f"T2I_R{k}"] = 100.0 * text_to_image / text_features.shape[0]
    return results
