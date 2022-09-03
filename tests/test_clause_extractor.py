from srl_toolkit.extractor import ClauseExtractor
import pytest


@pytest.fixture
def clause_extractor():
    return ClauseExtractor(
        udpipe_path="./resources/russian-syntagrus-ud-2.5-191206.udpipe",
        cb_path="./resources/catboost_clf.cbm",
    )

def test_init(clause_extractor):

    assert clause_extractor is not None


def test_extract(clause_extractor):
    text = "Мама мыла раму, а папа курил сигарету."
    result = clause_extractor(text)

    assert len(result["clauses"]) == 2

    assert result is not None