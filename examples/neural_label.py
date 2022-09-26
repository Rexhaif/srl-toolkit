from srl_toolkit.labeler import NeuralLabeler
from srl_toolkit.extractor import ClauseExtractor

clause_extractor = ClauseExtractor(
    udpipe_path="./resources/russian-syntagrus-ud-2.5-191206.udpipe",
    cb_path="./resources/catboost_model.cbm"
)

labeler = NeuralLabeler(
    model_name="Rexhaif/rubert-base-srl-seqlabeling",
    good_lemmas=None
)

text: str = "Мама рассердилась на папу"

clauses = clause_extractor(text)
print(clauses)
labeled_pas = labeler(clauses['clauses'])
print(labeled_pas)