from asyncio.log import logger
from srl_toolkit.extractor import PredicateArgumentExtractor, ClauseExtractor
from srl_toolkit.labeler import SrlLabeler
from srl_toolkit.ruleset import Ruleset, Rule
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
    rulesets = [
        Ruleset(
            predicate_rule=Rule(
                pattern={
                    "postag": "VERB",
                }
            ),
            argument_rules={
                "локатив": Rule(
                    pattern={
                        "postag": "NOUN",
                        "Case": "Loc",
                        "preposition": "на",
                    }
                )
            }
        )
    ]
    labeler = SrlLabeler(rulesets)
    text: str = "Мама прыгала на раме."
    clauses = clause_extractor(text)
    for clause in clauses['clauses']:
        logger.info(f"Clause = {clause}")
        pas = pa_extractor(clause)
        logger.info(f"Predicate-argument pairs = {pas}")
        labeled_pas = labeler(pas)
        logger.info(f"Labeled predicate-argument pairs = {labeled_pas}")

    logger.info(f"Rulesets = {rulesets[0].to_dict()}")