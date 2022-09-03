import conllu
import numpy as np
from .catboost_clf import CatBoostClf
from .feature_extractor import FeatureExtractor
from isanlp import PipelineCommon
from isanlp.annotation_rst import DiscourseUnit


class AnnotationCONLLConverter:
    """Converts isanlp-style annotation to raw CONLL-U format."""

    def __init__(self):
        self._unifeatures = (
            "Abbr",
            "Animacy",
            "Aspect",
            "Case",
            "Clusivity",
            "Definite",
            "Degree",
            "Evident",
            "Foreign",
            "Gender",
            "Mood",
            "NounClass",
            "NumType",
            "Number",
            "Person",
            "Polarity",
            "Polite",
            "Poss",
            "PronType",
            "Reflex",
            "Tense",
            "Typo",
            "VerbForm",
            "Voice",
            "Variant",
        )

    def __call__(self, doc_id: str, annotation: dict):
        assert "sentences" in annotation.keys()
        assert "tokens" in annotation.keys()
        assert "lemma" in annotation.keys()
        assert "postag" in annotation.keys()
        assert "morph" in annotation.keys()
        assert "syntax_dep_tree" in annotation.keys()

        _postag_key = "ud_postag" if "ud_postag" in annotation.keys() else "postag"

        yield "# newdoc id = " + doc_id
        for j, sentence in enumerate(annotation["sentences"]):
            for i, token_number in enumerate(range(sentence.begin, sentence.end)):
                parent_number = annotation["syntax_dep_tree"][j][i].parent
                if parent_number != -1:
                    parent_number += 1

                yield_string = "\t".join(
                    list(
                        map(
                            self._prepare_value,
                            [
                                i + 1,  # ID
                                annotation["tokens"][token_number].text,  # FORM
                                annotation["lemma"][j][i],  # LEMMA
                                annotation["postag"][j][i]
                                if annotation["postag"][j][i]
                                else "X",  # UPOS
                                "_",  # XPOS
                                self._to_universal_features(
                                    annotation["morph"][j][i]
                                ),  # FEATS
                                parent_number,  # HEAD
                                annotation["syntax_dep_tree"][j][i].link_name,  # DEPREL
                                "_",  # DEPS
                            ],
                        )
                    )
                )
                yield yield_string
            yield "\n"

    def _prepare_value(self, value):
        if type(value) == str:
            return value
        elif value:
            return str(value)
        elif value == 0:
            return "0"
        return "_"

    def _to_universal_features(self, morph_annot):
        if not morph_annot:
            return None

        return "|".join(
            [
                feature + "=" + morph_annot.get(feature)
                for feature in self._unifeatures
                if morph_annot.get(feature)
            ]
        )


class ClauseSegmenterProcessor:
    def __init__(self, model_dir_path):
        self._model_dir_path = model_dir_path
        self._conll_converter = AnnotationCONLLConverter()
        self._feature_extractor = FeatureExtractor()
        self._model = CatBoostClf(model_dir_path)

    @staticmethod
    def pipeline(model_path: str):
        pipeline = PipelineCommon(
            [
                (
                    Processor(model_dir_path=model_path),
                    [
                        "text",
                        "tokens",
                        "sentences",
                        "lemma",
                        "morph",
                        "postag",
                        "syntax_dep_tree",
                    ],
                    {0: "clauses"},
                )
            ],
            name="default",
        )
        return pipeline

    def __call__(
        self,
        annot_text,
        annot_tokens,
        annot_sentences,
        annot_lemma,
        annot_morph,
        annot_postag,
        annot_syntax_dep_tree,
    ):

        annot = {
            "text": annot_text,
            "tokens": annot_tokens,
            "sentences": annot_sentences,
            "lemma": annot_lemma,
            "morph": annot_morph,
            "postag": annot_postag,
            "syntax_dep_tree": annot_syntax_dep_tree,
        }

        converted_annot = ""
        for line in self._conll_converter(doc_id="0", annotation=annot):
            converted_annot += line + "\n"

        sentences = conllu.parse(converted_annot)
        features = self._feature_extractor(sentences)
        predictions = np.argwhere(np.array(self._model.predict(features)) == 1)[:, 0]
        return self._build_discourse_units(annot_text, annot_tokens, predictions)

    @staticmethod
    def _convert_annot(annot):
        _conll_converter = AnnotationCONLLConverter()
        converted_annot = ""
        for line in _conll_converter(doc_id="0", annotation=annot):
            converted_annot += line + "\n"
        return converted_annot

    def _build_discourse_units(self, text, tokens, numbers):
        """
        :param text: original text
        :param list tokens: isanlp.annotation.Token
        :param numbers: positions of tokens predicted as EDU left boundaries (beginners)
        :return: list of DiscourseUnit
        """

        edus = []
        start_id = 0

        if numbers.shape[0]:
            for i in range(0, len(numbers) - 1):
                new_edu = DiscourseUnit(
                    start_id + i,
                    start=tokens[numbers[i]].begin,
                    end=tokens[numbers[i + 1]].begin - 1,
                    text=text[tokens[numbers[i]].begin : tokens[numbers[i + 1]].begin],
                    relation="elementary",
                    nuclearity="_",
                )
                edus.append(new_edu)

            if numbers.shape[0] == 1:
                i = -1

            new_edu = DiscourseUnit(
                start_id + i + 1,
                start=tokens[numbers[-1]].begin,
                end=tokens[-1].end,
                text=text[tokens[numbers[-1]].begin : tokens[-1].end],
                relation="elementary",
                nuclearity="_",
            )
            edus.append(new_edu)

        return edus
