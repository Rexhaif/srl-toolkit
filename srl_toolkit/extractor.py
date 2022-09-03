from abc import ABC, abstractmethod
from typing import Callable

from isanlp.pipeline_common import PipelineCommon
from isanlp.processor_udpipe import ProcessorUDPipe
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp.ru.processor_mystem import ProcessorMystem
from joblib import Memory

from .clause_segmenter import ClauseSegmenterProcessor


class CachedExtractor(ABC):
    def __init__(
        self,
        cache_dir: str = "~/.cache/srl_toolkit",
        mmap_mode: str = "w+",
        compress: int | bool = None,
        bytes_limit: int = 10485760,
    ):
        self.memory = Memory(
            location=cache_dir,
            mmap_mode=mmap_mode,
            compress=compress,
            bytes_limit=bytes_limit,
        )

        self._extract = self.memory.cache(self._extract)

    @abstractmethod
    def _extract(self, text: str) -> dict:
        pass

    
    def __call__(self, text: str) -> dict:
        return self._extract(text)


class ClauseExtractor(CachedExtractor):
    def __init__(
        self,
        udpipe_path: str,
        cb_path: str,
        cache_dir: str = "~/.cache/srl_toolkit",
        mmap_mode: str = "r+",
        compress: int | bool = None,
        bytes_limit: int = 10485760,
    ):
        super().__init__(cache_dir, mmap_mode, compress, bytes_limit)
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
