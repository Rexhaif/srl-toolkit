from asyncio.log import logger
from srl_toolkit.extractor import PredicateArgumentExtractor, ClauseExtractor
from rich.logging import RichHandler
from rich import print_json
import json
import logging

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    pa_extractor = PredicateArgumentExtractor(
        udpipe_path="./resources/russian-syntagrus-ud-2.5-191206.udpipe"
    )
    clause_extractor = ClauseExtractor(
        udpipe_path="./resources/russian-syntagrus-ud-2.5-191206.udpipe",
        cb_path="./resources/catboost_model.cbm"
    )
    text: str = "Мама мыла раму, а папа, сидя на стуле, читал газету."
    clauses = clause_extractor(text)
    for clause in clauses['clauses']:
        logger.info(f"Clause = {clause}")
        pas = pa_extractor(clause)
        logger.info(f"Predicate-argument pairs = {pas}")