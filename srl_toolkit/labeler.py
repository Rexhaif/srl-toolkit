from __future__ import annotations
from srl_toolkit.ruleset import Rule, Ruleset

class SrlLabeler:

    def __init__(self, rulesets: list[Ruleset]) -> None:
        self.rulesets = rulesets

    def __call__(self, pas: dict[str, any]) -> dict[str, any]:
        """
        Applies the rulesets to the predicate-argument pairs.
        """
        _pas = pas['predicate_arguments'].copy()
        result = []
        for pa in _pas:
            labeled_pa = pa.copy()
            for ruleset in self.rulesets:
                labeled_pa, applied = ruleset(pa)
                if applied:
                    break
            result.append(labeled_pa)
        return {"labeled": result}
        