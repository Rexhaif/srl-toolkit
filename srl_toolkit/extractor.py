from abc import ABC, abstractmethod
from xxhash import xxh64
from isanlp.pipeline_common import PipelineCommon
from isanlp.processor_udpipe import ProcessorUDPipe
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp.ru.processor_mystem import ProcessorMystem
from diskcache import Cache

from .clause_segmenter import ClauseSegmenterProcessor
from .pa_extractor import PredicateExtractor, ArgumentExtractor


class CachedExtractor(ABC):
    def __init__(
        self,
        cache_dir: str = "~/.cache/srl_toolkit"
    ):
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
        model, inputs, outputs = ClauseSegmenterProcessor.for_pipeline(cb_path)
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

    def _extract(self, text: str) -> dict:
        result = self.pipeline(text)
        clauses = [x.text for x in result['clauses']]
        return {"clauses": clauses}


class PredicateArgumentExtractor(CachedExtractor):

    def __init__(self, udpipe_path: str, cache_dir: str = "~/.cache/srl_toolkit"):
        super().__init__(cache_dir)
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
                    }
                )
            ]
        )
        self.predicate_extractor = PredicateExtractor()
        self.argument_extractor = ArgumentExtractor()

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
            arguments = [parse["tokens"][idx].text for idx in arguments]
            result.append({"predicate": predicate, "arguments": arguments})

        return {'predicate_arguments': result}

