from copy import deepcopy

from aymurai.meta.types import DataItem
from aymurai.utils.misc import get_element
from aymurai.meta.pipeline_interfaces import Transform

from .patterns import regex_patterns
from .utils import find_subcategories


class RegexSubcategorizer(Transform):
    NEED_CONTEXT = ["PERSONA_ACUSADA_NO_DETERMINADA"]

    def __call__(self, item: DataItem) -> DataItem:
        item = deepcopy(item)

        ents = get_element(item, levels=["predictions", "entities"]) or []

        for category, patterns in regex_patterns.items():
            use_context = True if category in self.NEED_CONTEXT else False
            ents = find_subcategories(ents, category, patterns, use_context=use_context)

        item["predictions"]["entities"] = ents

        return item
