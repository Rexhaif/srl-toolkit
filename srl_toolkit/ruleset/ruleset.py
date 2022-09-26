from __future__ import annotations


class Rule:
    def __init__(self, pattern: dict[str, str]):
        self.pattern = pattern

    def __call__(self, word: dict[str, str]) -> bool:
        """
        Checks if the word matches the rule.
        """
        _word = word.copy()
        morph = _word.pop("morph")
        for key, value in morph.items():
            _word[key] = value

        for k, v in self.pattern.items():
            if k not in _word:
                return False
            if isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
                if _word[k] not in v:
                    return False
            else:
                if _word[k] != v:
                    return False
        return True

    def __repr__(self):
        return f"Rule({self.pattern})"

    def __str__(self):
        return f"Rule({self.pattern})"

    def to_dict(self):
        return {"pattern": self.pattern}


class Ruleset:
    def __init__(self, predicate_rule: Rule, argument_rules: dict[str, list[Rule]]):
        self.predicate_rule = predicate_rule
        self.argument_rules = argument_rules

    def __call__(self, pa: dict[str, any]) -> tuple[dict[str, any], bool]:
        """
        Applies the ruleset to the predicate-argument pair.
        """
        applied = False
        if self.predicate_rule(pa["predicate"]):
            for role, rules in self.argument_rules.items():
                for rule in rules:
                    for arg in pa["arguments"]:
                        if rule(arg):
                            arg["role"] = role
                            applied = True
                        else:
                            arg["role"] = None
        return pa, applied

    def __repr__(self):
        return f"Ruleset({self.predicate_rule}, {self.argument_rules})"

    def __str__(self):
        return f"Ruleset({self.predicate_rule}, {self.argument_rules})"

    def to_dict(self):
        return {
            "predicate_rule": self.predicate_rule.to_dict(),
            "argument_rules": {
                role: [x.to_dict() for x in rule_list]
                for role, rule_list in self.argument_rules.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return cls(
            predicate_rule=Rule(**data["predicate_rule"]),
            argument_rules={
                role: [Rule(**rule) for rule in rule_list]
                for role, rule_list in data["argument_rules"].items()
            },
        )
