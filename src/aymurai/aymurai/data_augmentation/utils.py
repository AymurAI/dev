import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset


def compute_label_counts(
    dataset: Dataset,
    code2label: dict[int, str],
) -> dict[str, int]:
    """Computes the number of times each label appears in the dataset.

    Args:
        dataset (Dataset): dataset to compute the label counts.
        code2label (dict[int, str]): mapping from code (int) to label (str).

    Returns:
        dict[str, int]: dictionary with the label counts.
    """
    counts = []
    for example in tqdm(dataset, total=len(dataset)):
        labels = [code2label[code] for code in example["tags"]]
        labels = [re.sub(r"[BI]-", "", label) for label in labels]
        labels, count = np.unique(labels, return_counts=True)

        counts.append({l: c for l, c in zip(labels, count)})
    counts = pd.DataFrame(counts)
    counts = counts.sum().astype(int)
    return counts.to_dict()


def compute_label_weights(
    dataset: Dataset,
    code2label: dict[int, str],
    ignore_labels: list[str] = ["O", "PER", "FECHA"],
) -> dict[str, float]:
    """Computes weights for each label based on the number of times it appears in the dataset.

    Args:
        dataset (Dataset): dataset to compute the label weights.
        code2label (dict[int, str]): mapping from code (int) to label (str).
        ignore_labels (list[str], optional): labels to ignore, typically the ones with the highest frequency.
            Defaults to ["O", "PER", "FECHA"].

    Returns:
        dict[str, float]: dictionary with the label weights.
    """
    counts = compute_label_counts(dataset, code2label)
    counts = pd.Series(counts)
    counts = counts.drop(ignore_labels)

    label_weights = counts.sum() / counts
    label_weights /= label_weights.min()
    label_weights = label_weights.to_dict()
    return label_weights
