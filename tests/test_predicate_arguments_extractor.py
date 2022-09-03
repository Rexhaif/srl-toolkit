import pytest
from srl_toolkit.extractor import PredicateArgumentExtractor


@pytest.fixture
def extractor():
    return PredicateArgumentExtractor(udpipe_path="./resources/russian-syntagrus-ud-2.5-191206.udpipe")


def test_init(extractor):
    assert extractor is not None

def test_extraction(extractor):
    text = "Мама мыла раму."
    result = extractor(text)

    assert result is not None
    assert 'predicate_arguments' in result
    assert len(result['predicate_arguments']) == 1
    assert result['predicate_arguments'][0] == {
        'predicate': 'мыла',
        'arguments': [
            'Мама',
            'раму'
        ]
    }