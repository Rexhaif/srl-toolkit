from __future__ import annotations

import string

import razdel
import transformers as tr
from pymystem3 import Mystem

from srl_toolkit.ruleset import Rule, Ruleset


class SrlLabeler:
    def __init__(self, rulesets: list[Ruleset]) -> None:
        self.rulesets = rulesets

    def __call__(self, pas: dict[str, any]) -> dict[str, any]:
        """
        Applies the rulesets to the predicate-argument pairs.
        """
        _pas = pas["predicate_arguments"].copy()
        result = []
        for pa in _pas:
            labeled_pa = pa.copy()
            for ruleset in self.rulesets:
                labeled_pa, applied = ruleset(pa)
                if applied:
                    break
            result.append(labeled_pa)
        return {"labeled": result}


class NeuralLabeler:
    def __init__(self, model_name: str, good_lemmas: list[str]) -> None:
        self.pipeline = tr.pipeline(
            "token-classification", model=model_name, aggregation_strategy="simple"
        )
        self.good_lemmas = good_lemmas
        self.mystem = Mystem(entire_input=False)

    def __map_to_word_idx(
        self, spans: list[tuple[int, int]], start: int, end: int
    ) -> list[int]:
        """
        Map to word index
        """
        word_indices = []
        for i, (i_start, i_end) in enumerate(spans):
            if i_start == start:
                word_indices.append(i)

            if i_end == end:
                word_indices.append(i)

        return list(sorted(set(word_indices)))

    def __deduplicate_predictions(self, predictions: list[dict]) -> list[dict]:
        """
        Deduplicate predictions
        """
        new_predictions = {}
        used_idxs = set()
        for label in predictions:
            for prediction in predictions[label]:
                if len(prediction["idxs"]) == 1 and prediction["idxs"][0] in used_idxs:
                    continue
                used_idxs.update(prediction["idxs"])
                if label not in new_predictions:
                    new_predictions[label] = []
                new_predictions[label].append(prediction)

        if "predicate" not in new_predictions:
            return {}
        else:
            return new_predictions

    def __postprocess_predictions(
        self, text: str, predictions: list[dict]
    ) -> list[dict]:
        """
        Postprocess predictions
        """
        response = {}
        tokenized_text = list(razdel.tokenize(text))
        tokenized_text = list(
            filter(lambda x: x.text not in string.punctuation, tokenized_text)
        )
        words = [token.text for token in tokenized_text]
        spans = [(token.start, token.stop) for token in tokenized_text]
        _reconstructed_text = " ".join(words)
        lemmas = self.mystem.lemmatize(_reconstructed_text)

        for prediction in predictions:
            label = prediction["entity_group"].lower()
            word_idxs = self.__map_to_word_idx(
                spans, prediction["start"], prediction["end"]
            )

            if len(word_idxs) == 0:
                continue
            if label not in response:
                response[label] = []

            if label == "predicate":
                lemma = lemmas[word_idxs[0]]
                if self.good_lemmas and lemma not in self.good_lemmas:
                    continue

            response[label].append(
                {
                    "idxs": word_idxs,
                    "text": " ".join(words[i] for i in word_idxs),
                    "score": float(prediction["score"]),
                }
            )
        return self.__deduplicate_predictions(response)

    def __call__(self, clauses: list[str]) -> list[dict[str, any]]:
        response = []
        predictions = self.pipeline(clauses)
        for i in range(len(clauses)):

            response.append(
                {
                    "text": clauses[i],
                    "predictions": self.__postprocess_predictions(
                        clauses[i], predictions[i]
                    ),
                }
            )

        return response
