from abc import ABC, abstractmethod
from ast import arg
from asyncio.log import logger

from diskcache import Cache
from isanlp.pipeline_common import PipelineCommon
from isanlp.processor_udpipe import ProcessorUDPipe
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp.ru.processor_mystem import ProcessorMystem
from xxhash import xxh64
import time
import logging

from srl_toolkit.ruleset import Ruleset, Rule

from .clause_segmenter import ClauseSegmenterProcessor
from .pa_extractor import ArgumentExtractor, PredicateExtractor
from rich import inspect


logger = logging.getLogger(__name__)

class CachedExtractor(ABC):
    def __init__(self, cache_dir: str = "~/.cache/srl_toolkit"):
        self.cache = Cache(cache_dir)

    @property
    def classname(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _extract(self, text: str) -> dict:
        pass

    def __call__(self, text: str) -> dict:
        key: str = f"{self.classname}:{text}"
        key: bytes = xxh64(key).digest()
        if key in self.cache:
            return self.cache[key]
        else:
            result = self._extract(text)
            self.cache[key] = result
            return result


class ClauseExtractor(CachedExtractor):
    def __init__(
        self,
        udpipe_path: str,
        cb_path: str,
        cache_dir: str = "~/.cache/srl_toolkit",
    ):
        super().__init__(cache_dir)
        _t1 = time.time()
        model, inputs, outputs = ClauseSegmenterProcessor.for_pipeline(cb_path)
        _t2 = time.time() - _t1
        logger.debug(f"Loaded model for {self.classname} in {_t2:.2f} seconds")
        _t1 = time.time()
        self.pipeline = PipelineCommon(
            [
                (
                    ProcessorUDPipe(udpipe_path),
                    ["text"],
                    {
                        "sentences": "sentences",
                        "tokens": "tokens",
                        "lemma": "lemma",
                        "syntax_dep_tree": "syntax_dep_tree",
                        "postag": "ud_postag",
                    },
                ),
                (
                    ProcessorMystem(delay_init=False),
                    ["tokens", "sentences"],
                    {"postag": "postag"},
                ),
                (
                    ConverterMystemToUd(),
                    ["postag"],
                    {"morph": "morph", "postag": "postag"},
                ),
                (model, inputs, outputs),
            ]
        )
        _t2 = time.time() - _t1
        logger.debug(f"Loaded pipeline for {self.classname} in {_t2:.2f} seconds")

    def _extract(self, text: str) -> dict:
        result = self.pipeline(text)
        clauses = [x.text for x in result["clauses"]]
        return {"clauses": clauses}


class PredicateArgumentExtractor(CachedExtractor):
    def __init__(self, udpipe_path: str, prepostion_search_radius: int = 3, cache_dir: str = "~/.cache/srl_toolkit"):
        super().__init__(cache_dir)
        _t1 = time.time()
        self.pipeline = PipelineCommon(
            [
                (
                    ProcessorUDPipe(udpipe_path),
                    ["text"],
                    {
                        "tokens": "tokens",
                        "lemma": "lemma",
                        "postag": "postag",
                        "morph": "morph",
                        "syntax_dep_tree": "syntax_dep_tree",
                    },
                )
            ]
        )
        self.predicate_extractor = PredicateExtractor()
        self.argument_extractor = ArgumentExtractor()
        _t2 = time.time() - _t1
        logger.debug(f"Loaded pipeline for {self.classname} in {_t2:.2f} seconds")
        self.prepostion_search_radius = prepostion_search_radius

    def __get_preposition(self, word_idx: int, tokens, syntax_dep_tree, postag):
        for i in range(1, self.prepostion_search_radius + 1):
            if word_idx - i >= 0 \
                and syntax_dep_tree[word_idx - i].parent == word_idx \
                and postag[word_idx - i] == "ADP":
                return tokens[word_idx - i].text.lower()
        return None

    def _extract(self, text: str) -> dict:
        parse = self.pipeline(text)
        predicates = [
            (parse["tokens"][idx].text, idx)
            for idx in self.predicate_extractor(parse["postag"][0])
        ]
        result = []
        for predicate, position in predicates:
            arguments = self.argument_extractor(
                position,
                parse["postag"][0],
                parse["morph"][0],
                parse["lemma"][0],
                parse["syntax_dep_tree"][0],
            )
            _arguments = []
            for idx in arguments:
                word = {
                    "text": parse["tokens"][idx].text,
                    "lemma": parse["lemma"][0][idx],
                    "morph": parse["morph"][0][idx],
                    "postag": parse["postag"][0][idx],
                }
                # search for prepositions
                word["preposition"] = self.__get_preposition(
                    idx, parse["tokens"], parse["syntax_dep_tree"][0], parse["postag"][0]
                )

                _arguments.append(word)
            predicate_dict = {
                "text": predicate,
                "lemma": parse["lemma"][0][position],
                "morph": parse["morph"][0][position],
                "postag": parse["postag"][0][position],
            }
            predicate_dict["preposition"] = self.__get_preposition(
                position, parse["tokens"], parse["syntax_dep_tree"][0], parse["postag"][0]
            )
            result.append({"predicate": predicate_dict, "arguments": _arguments})
            

        return {"predicate_arguments": result}
