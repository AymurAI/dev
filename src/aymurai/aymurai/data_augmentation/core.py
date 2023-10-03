import re
from itertools import groupby

from joblib import hash
from datasets import Dataset
from more_itertools import unzip, flatten

from aymurai.data_augmentation.anonymizer_entities import faker

from .utils import compute_label_weights

FORMAT_FUNCTIONS = [
    lambda x: x,
    str.lower,
    str.upper,
    str.title,
    str.capitalize,
]


class DataAugmenter:
    """Data augmenter class. It replaces entities with fake data."""

    def __init__(
        self,
        code2label: dict,
        augmentation_functions: dict,
        random_state: int | None = None,
    ) -> None:
        """Initialize the data augmenter.

        Args:
            code2label (dict): mapping from code (int) to label (str).
            augmentation_functions (dict): dictionary with the augmentation functions for each entity.
            random_state (int | None, optional): random state. Defaults to None.
        """
        self.random_state = random_state
        self.augmentation_functions = augmentation_functions
        self.code2label = code2label
        self.label2code = {v: k for k, v in self.code2label.items()}

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value: int | None):
        faker.seed_instance(value)
        self._random_state = value

    def _get_tokens_and_tags(
        self, text: str, entity: str
    ) -> tuple[list[str], list[int]]:
        """Get tokens and tags for a given text and entity.

        Args:
            text (str): text to be tokenized.
            entity (str): entity to be tagged.

        Returns:
            tuple[list[str], list[int]]: tokens and tags.
        """
        tokens = text.split()
        tags = [self.label2code.get(f"I-{entity}")] * len(tokens)
        tags[0] = self.label2code.get(f"B-{entity}")
        return tokens, tags

    def augment_sample(self, sample: dict) -> dict:
        """Augment a sample by replacing entities with fake data.

        Args:
            sample (dict): sample to be augmented. Must contain the following keys:
                - 'tokens': list of tokens.
                - 'tags': list of tags.
                - 'hash': hash of the original sample.

        Returns:
            dict: augmented sample.
        """
        sample_tokens = sample["tokens"]
        sample_tags = sample["tags"]
        original_hash = sample["hash"]
        sample_labels = [
            re.sub(r"^[BI]-", "", self.code2label.get(tag, "O")) for tag in sample_tags
        ]

        # initialize empty augmented tokens and tags
        augmented_tokens = []
        augmented_tags = []

        # loop over entities and group the tokens and tags
        for entity, group in groupby(
            zip(sample_tokens, sample_tags, sample_labels), key=lambda x: x[2]
        ):
            tokens, tags, _ = unzip(group)

            # if the entity is not in the augmentation functions,
            # append the original tokens and tags
            if entity in ["O", "TEXTO_ANONIMIZAR"]:
                augmented_tokens.append(list(tokens))
                augmented_tags.append(list(tags))
                continue

            # get the replacement tokens and tags
            replacement_tokens, replacement_tags = self._get_tokens_and_tags(
                self.augmentation_functions.get(entity)(), entity
            )

            augmented_tokens.append(list(replacement_tokens))
            augmented_tags.append(list(replacement_tags))

        # flatten the augmented tokens and tags
        augmented_tokens = list(flatten(augmented_tokens))
        augmented_tags = list(flatten(augmented_tags))

        # apply a random format function to the augmented tokens
        format_function = faker.random_element(FORMAT_FUNCTIONS)
        augmented_tokens = list(map(format_function, augmented_tokens))

        # compute the number of labels
        n_labels = len([tag for tag in augmented_tags if tag > 0])

        # create the augmented sample
        augmented_sample = {
            "n_labels": n_labels,
            "tokens": augmented_tokens,
            "tags": augmented_tags,
            "original_hash": original_hash,
            "hash": hash(" ".join(augmented_tokens)),
        }

        return augmented_sample

    def augment_dataset(
        self,
        dataset: Dataset,
        weighted: bool = True,
        frac: float = 1.0,
        ignore_labels: list[str] = ["O", "PER", "FECHA"],
    ) -> Dataset:
        """Augment a dataset by replacing entities with fake data.

        Args:
            dataset (Dataset): dataset to be augmented.
            weighted (bool, optional): whether to weight the samples. Defaults to True.
            frac (float, optional): fraction of the dataset to be sampled. Defaults to 1.0.
            ignore_labels (list[str], optional): labels to ignore. Defaults to ["O", "PER", "FECHA"].

        Returns:
            Dataset: augmented dataset.
        """
        if weighted:
            # compute label weights
            label_weights = compute_label_weights(
                dataset=dataset,
                code2label=self.code2label,
                ignore_labels=ignore_labels,
            )

            # add weight field to dataset
            def get_weight(example):
                labels = [self.code2label[code] for code in example["tags"]]
                labels = [re.sub(r"[BI]-", "", label) for label in labels]
                weights = [label_weights.get(label, 0) for label in labels]

                example["weight"] = max(weights)
                return example

            dataset = dataset.map(get_weight)
        else:
            # add weight field to dataset equal to 1
            dataset = dataset.map(lambda sample: sample | {"weight": 1})

        # resample
        if weighted:
            resampled = dataset.to_pandas().sample(
                frac=frac,
                weights=dataset["weight"],
                replace=True,
                random_state=self.random_state,
            )
            resampled = Dataset.from_pandas(resampled)
            dataset = resampled.remove_columns(["__index_level_0__"])

        # remove internal weight field
        dataset = dataset.remove_columns(["weight"])

        # apply augment function
        dataset = dataset.map(self.augment_sample)

        return dataset
