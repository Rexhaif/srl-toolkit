import collections
import multiprocessing

import pandas as pd


class FeatureExtractor:
    def __init__(self):
        self.categorical_features = [
            "lemma",
            "upos",
            "animacy",
            "aspect",
            "case",
            "degree",
            "gender",
            "number",
            "person",
            "tense",
            "verbform",
            "voice",
            "deprel",
        ]

        self.categorical_features += [
            "parent_" + feature_name for feature_name in self.categorical_features
        ]

        self.other_features = [
            "word_index",
            "parent_index",
            "is_capitalized",
            "is_upper",
            "n_children",
        ]

        self.ancestor_categorical_features = [
            "lemma",
            "upos",
            "animacy",
            "aspect",
            "case",
            "degree",
            "gender",
            "number",
            "person",
            "tense",
            "verbform",
            "voice",
            "deprel",
        ]
        self.ancestor_other_features = ["is_capitalized", "is_upper", "n_children"]

        self.out_features = [
            "lemma",
            "upos",
            "animacy",
            "aspect",
            "case",
            "degree",
            "gender",
            "number",
            "person",
            "tense",
            "verbform",
            "voice",
            "deprel",
            "parent_lemma",
            "parent_upos",
            "parent_animacy",
            "parent_aspect",
            "parent_case",
            "parent_degree",
            "parent_gender",
            "parent_number",
            "parent_person",
            "parent_tense",
            "parent_verbform",
            "parent_voice",
            "parent_deprel",
            "lemma_prev_1",
            "upos_prev_1",
            "animacy_prev_1",
            "aspect_prev_1",
            "case_prev_1",
            "degree_prev_1",
            "gender_prev_1",
            "number_prev_1",
            "person_prev_1",
            "tense_prev_1",
            "verbform_prev_1",
            "voice_prev_1",
            "deprel_prev_1",
            "parent_lemma_prev_1",
            "parent_upos_prev_1",
            "parent_animacy_prev_1",
            "parent_aspect_prev_1",
            "parent_case_prev_1",
            "parent_degree_prev_1",
            "parent_gender_prev_1",
            "parent_number_prev_1",
            "parent_person_prev_1",
            "parent_tense_prev_1",
            "parent_verbform_prev_1",
            "parent_voice_prev_1",
            "parent_deprel_prev_1",
            "lemma_prev_2",
            "upos_prev_2",
            "animacy_prev_2",
            "aspect_prev_2",
            "case_prev_2",
            "degree_prev_2",
            "gender_prev_2",
            "number_prev_2",
            "person_prev_2",
            "tense_prev_2",
            "verbform_prev_2",
            "voice_prev_2",
            "deprel_prev_2",
            "parent_lemma_prev_2",
            "parent_upos_prev_2",
            "parent_animacy_prev_2",
            "parent_aspect_prev_2",
            "parent_case_prev_2",
            "parent_degree_prev_2",
            "parent_gender_prev_2",
            "parent_number_prev_2",
            "parent_person_prev_2",
            "parent_tense_prev_2",
            "parent_verbform_prev_2",
            "parent_voice_prev_2",
            "parent_deprel_prev_2",
            "is_capitalized",
            "is_upper",
            "position",
            "distance_to_end",
            "n_children",
            "is_capitalized_prev_1",
            "is_upper_prev_1",
            "n_children_prev_1",
            "is_capitalized_prev_2",
            "is_upper_prev_2",
            "n_children_prev_2",
        ]

    def __call__(self, sentences):
        self.sentences = sentences
        df = self.to_dataframe(sentences)
        features = FeatureExtractor._parallelize_dataframe(df, self._all_features)
        # features = FeatureExtractor._add_ancestor_features(features, 2,
        #                                                    self.ancestor_categorical_features,
        #                                                    self.ancestor_other_features, sentences)
        # features['same_common_ancestors'] = 1 * (features.index_ancestor_2_1 == features.index_ancestor_1_0)
        return features[self.out_features]

    def to_dataframe(self, conll_data: list):
        data = []
        for sentence_id, sentence in enumerate(conll_data):
            childer_counter = collections.Counter(
                word_data["head"] for word_data in sentence
            )

            for word_index, word_data in enumerate(sentence):
                try:
                    features = {
                        "sentence_id": sentence_id,
                        "word_id": word_index,
                        "lemma": word_data["lemma"],
                        "upos": word_data["upostag"],
                        "animacy": "",
                        "aspect": "",
                        "case": "",
                        "degree": "",
                        "gender": "",
                        "number": "",
                        "person": "",
                        "tense": "",
                        "verbform": "",
                        "voice": "",
                        "deprel": "",
                        "is_capitalized": int(word_data["form"][0].isupper()),
                        "is_upper": int(word_data["form"].isupper()),
                        "position": word_index / (len(sentence) - 1)
                        if len(sentence) > 1
                        else 0,
                        "distance_to_end": len(sentence) - word_index - 1,
                        "n_children": childer_counter[word_index],
                        "begin_segment": 0,
                        "parent_index": word_data["head"],
                        "parent_lemma": sentence[word_data["head"] - 1]["lemma"]
                        if word_data["head"] > -1
                        else "NA",
                        "parent_upos": sentence[word_data["head"] - 1]["upostag"]
                        if word_data["head"] > -1
                        else "NA",
                        "parent_animacy": "" if word_data["head"] > -1 else "NA",
                        "parent_aspect": "" if word_data["head"] > -1 else "NA",
                        "parent_case": "" if word_data["head"] > -1 else "NA",
                        "parent_degree": "" if word_data["head"] > -1 else "NA",
                        "parent_gender": "" if word_data["head"] > -1 else "NA",
                        "parent_number": "" if word_data["head"] > -1 else "NA",
                        "parent_person": "" if word_data["head"] > -1 else "NA",
                        "parent_tense": "" if word_data["head"] > -1 else "NA",
                        "parent_verbform": "" if word_data["head"] > -1 else "NA",
                        "parent_voice": "" if word_data["head"] > -1 else "NA",
                        "parent_deprel": "" if word_data["head"] > -1 else "NA",
                    }
                    if word_data["feats"] is not None:
                        for key in word_data["feats"]:
                            features[key.lower()] = word_data["feats"][key]

                    if word_data["deprel"] is not None:
                        features["deprel"] = word_data["deprel"]

                    if word_data["head"] > -1:
                        parent_data = sentence[word_data["head"] - 1]
                        if parent_data["feats"] is not None:
                            for key in parent_data["feats"]:
                                features["parent_" + key.lower()] = parent_data[
                                    "feats"
                                ][key]
                        if parent_data["deprel"] is not None:
                            features["parent_deprel"] = parent_data["deprel"]
                    data.append(features)

                except IndexError:
                    print(sentence_id, word_data)

        dataframe = pd.DataFrame(data).set_index(["sentence_id", "word_id"])
        dataframe.sort_index(inplace=True)
        dataframe["word_index"] = dataframe.index.get_level_values("word_id").values
        return dataframe

    @staticmethod
    def _add_previous_word_features(dataframe, n, categorical_features, other_features):
        def prev_feature_name(feature_name, k):
            return f"{feature_name}_prev_{k}"

        all_features = categorical_features + other_features

        expanded_dataframe = dataframe.copy()
        for i in range(1, n + 1):
            for feature in categorical_features:
                expanded_dataframe[prev_feature_name(feature, i)] = "NA"
            for feature in other_features:
                expanded_dataframe[prev_feature_name(feature, i)] = -1

        for row in dataframe.itertuples():
            sentence_id = row.Index[0]
            word_index = row.Index[1]
            for i in range(1, n + 1):
                if word_index < i:
                    continue
                for feature in all_features:
                    expanded_dataframe.at[
                        row.Index, prev_feature_name(feature, i)
                    ] = expanded_dataframe.at[(sentence_id, word_index - i), feature]

        return expanded_dataframe

    def _func_previous(self, d):
        return FeatureExtractor._add_previous_word_features(
            d, 2, self.categorical_features, self.other_features
        )

    @staticmethod
    def _lowest_common_ancestor(sentence_data, word_index1, word_index2):
        def ancestor_indices(sentence_data, word_index):
            ancestors = [word_index]
            while sentence_data[word_index]["head"] > -1:
                word_index = sentence_data[word_index]["head"]
                if word_index != -1:
                    word_index -= 1

                ancestors.append(word_index)

            return ancestors

        ancestors1 = ancestor_indices(sentence_data, word_index1)
        ancestors2 = ancestor_indices(sentence_data, word_index2)
        ancestors2 = {
            ancestor: distance for distance, ancestor in enumerate(ancestors2)
        }

        for distance1, ancestor in enumerate(ancestors1):
            if ancestor in ancestors2:
                return ancestor, distance1, ancestors2[ancestor]

        return (None, None, None)

    @staticmethod
    def _add_ancestor_features(
        dataframe, n, ancestor_categorical_features, ancestor_other_features, conll_data
    ):
        def prev_feature_name(feature_name, k):
            return feature_name if k == 0 else f"{feature_name}_prev_{k}"

        def ancestor_feature_name(feature_name, k, m, l=""):
            return f"{feature_name}{l}_ancestor_{k}_{m}"

        for i in range(n, 0, -1):
            for feature in ancestor_categorical_features:
                dataframe[ancestor_feature_name(feature, i, i - 1)] = "NA"
            for feature in ancestor_other_features:
                dataframe[ancestor_feature_name(feature, i, i - 1)] = -1
            dataframe[ancestor_feature_name("distance", i, i - 1, i)] = -1
            dataframe[ancestor_feature_name("distance", i, i - 1, i - 1)] = -1
            dataframe[ancestor_feature_name("index", i, i - 1)] = -1

        all_features = ancestor_categorical_features + ancestor_other_features
        for row in dataframe.itertuples():
            sentence_id = row.Index[0]

            for i in range(n, 0, -1):
                word_index1 = getattr(row, prev_feature_name("word_index", i))
                word_index2 = getattr(row, prev_feature_name("word_index", i - 1))

                if word_index1 < 0 or word_index2 < 0:
                    continue

                word_index1 -= 1
                word_index2 -= 1

                (
                    ancestor,
                    distance1,
                    distance2,
                ) = FeatureExtractor._lowest_common_ancestor(
                    conll_data[sentence_id], word_index1, word_index2
                )

                if ancestor:

                    for feature in all_features:
                        dataframe.at[
                            row.Index, ancestor_feature_name(feature, i, i - 1)
                        ] = dataframe.at[(sentence_id, ancestor), feature]
                    dataframe.at[
                        row.Index, ancestor_feature_name("distance", i, i - 1, i)
                    ] = distance1
                    dataframe.at[
                        row.Index, ancestor_feature_name("distance", i, i - 1, i - 1)
                    ] = distance2
                    dataframe.at[
                        row.Index, ancestor_feature_name("index", i, i - 1)
                    ] = ancestor

        return dataframe

    def _func_anc(self, d):
        features = FeatureExtractor._add_ancestor_features(
            d,
            2,
            self.ancestor_categorical_features,
            self.ancestor_other_features,
            self.sentences,
        )
        return features

    def _all_features(self, d):
        features = FeatureExtractor._add_previous_word_features(
            d, 2, self.categorical_features, self.other_features
        )
        # features = FeatureExtractor._add_ancestor_features(features, 2,
        #                                                    self.ancestor_categorical_features,
        #                                                    self.ancestor_other_features,
        #                                                    self.sentences)
        return features

    @staticmethod
    def _parallelize_dataframe(df, func):
        num_cores = 8
        num_partitions = num_cores

        def iter_by_group(df, column, num_groups):
            groups = []
            for i, group in df.groupby(column):
                groups.append(group)
                if len(groups) == num_groups:
                    yield pd.concat(groups)
                    groups = []
            if groups:
                yield pd.concat(groups)

        df_split = list(iter_by_group(df, "sentence_id", num_partitions))

        pool = multiprocessing.Pool(num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df
