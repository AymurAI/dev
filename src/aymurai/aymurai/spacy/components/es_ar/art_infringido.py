# codeblock generated by devtools
# source: /workspace/notebooks/dev/patterns/03-sections-spot/30-art-infringido.ipynb

import re
from copy import deepcopy

import spacy
from spacy.tokens import Span
from more_itertools import unique_everseen

from aymurai.spacy.utils import format_entity
from aymurai.meta.pipeline_interfaces import Transform
from aymurai.spacy.components.regex import EnhancedRegexMatcher
from aymurai.spacy.components.es_ar.articles.patterns import (
    ABBRS,
    CODES,
    ART_PATTERN_MULTI_MOD,
    ART_PATTERN_MULTI_PREFIX,
)


class SpacyRulerArtInfringido(Transform):
    LAYER0_PREFIX = r"(SOBRE|s/|s\\)\s*"
    LAYER0_SUFFIX = r"\s*(\"|\n|N[uú]mero)"
    PAT_COND1_SUFFIX = r"\s*(\"|\n|N[uú]mero:?)"
    PAT_COND1_PREFIX = r"(-\s+)"

    def __init__(self):
        global __nlp
        __nlp = spacy.blank("es")
        self.matcher_layer_0 = EnhancedRegexMatcher(__nlp.vocab)
        self.matcher_layer_1 = EnhancedRegexMatcher(__nlp.vocab)
        self.matcher_layer_0.add(
            "LAYER_0", patterns=[f"{self.LAYER0_PREFIX}.*?{self.LAYER0_SUFFIX}"]
        )
        self.matcher_layer_1.add(
            "ART_INFRINGIDO",
            patterns=[
                r"[\d\.]{2,}",
                ART_PATTERN_MULTI_PREFIX,
                ART_PATTERN_MULTI_MOD,
            ],
        )
        # self.matcher_layer_1.add(
        #     "CONDUCTA1",
        #     patterns=[
        #         f"{self.PAT_COND1_PREFIX}.*?{self.PAT_COND1_SUFFIX}",
        #     ],
        # )
        self.matcher_layer_1.add("CODIGO_O_LEY", patterns=CODES)
        self.matcher_layer_1.add(
            "CODIGO_O_LEY", patterns=[f"{abbr}(\s?CABA)?" for abbr in ABBRS]
        )
        self.matcher_layer_1.add(
            "CODIGO_O_LEY", patterns=["(?i)ley(es)?(( y|,)? ([\d\.]+))+"]
        )

    def clean_span(self, span):
        match span.label_:
            case "CONDUCTA":
                pre = __nlp.make_doc(re.sub(f"^{self.PAT_COND1_PREFIX}", "", span.text))
                post = __nlp.make_doc(
                    re.sub(f"{self.PAT_COND1_SUFFIX}$", "", span.text)
                )
                start = len(span) - len(pre)
                end = len(span) - len(post)
                span = Span(
                    span.doc,
                    start=span.start + start,
                    end=span.end - end,
                    label="CONDUCTA",
                )

        return span

    def __call__(self, item):
        item = deepcopy(item)
        if "entities" not in item["data"]:
            item["data"]["entities"] = []

        fragment = item["data"]["doc.text"][:700]
        doc = __nlp(fragment)

        matches = []
        matches_layer_0 = self.matcher_layer_0(doc)
        for label0, start0, end0, score0 in matches_layer_0:
            span = doc[start0:end0]
            doc0 = span.as_doc()

            matches_layer_1 = self.matcher_layer_1(doc0)
            for label1, start1, end1, score1 in matches_layer_1:
                start1 += start0
                end1 += start0
                matches += [(label1, start1, end1, score1)]
                if label1 == "ART_INFRINGIDO":
                    matches += [
                        ("CONDUCTA", end1 + 1, start0 + len(doc0) + 1, (0, 0, 0))
                    ]

        matches = sorted(matches, key=lambda x: (sum(x[3]), x[1]))
        matches = unique_everseen(matches, key=lambda x: x[0])
        matches = list(matches)

        for label, start, end, score in matches:
            span = Span(doc, start=start, end=end, label=label)
            span = self.clean_span(span)
            item["data"]["entities"] += [format_entity(span)]

        # item["data"]["doc.text"] = fulltext

        # item["data"]["entities"] = [
        #     self.clean_art(ent) for ent in item["data"]["entities"]
        # ]

        return item
