import numpy as np
import sklearn.linear_model as sklm


class RandomLRHash:

    def __init__(
            self,
            debug=False,
            tune_hp=False,
            C=1.0,
            penalty='l1',
    ):
        self.debug = debug
        self.tune_hp = tune_hp
        assert not tune_hp, 'tune at higher level, not here'

        self.solver = 'liblinear'
        self.penalty = penalty
        self.C = C
        self.fit_intercept = True
        self.class_weight = 'balanced'

    def __infer_bits__(self, events_ref_set, labels, events, parameters):

        clf_obj = sklm.LogisticRegression(**parameters).fit(
            X=events_ref_set,
            y=labels,
        )

        z = clf_obj.predict(events)
        z = z.astype(np.bool)

        return z

    def compute_hashcode_bit(self,
            events,
            events_ref_set,
            subset1,
            subset2,
    ):
        assert events.shape[1] == events_ref_set.shape[1]

        # representing the two subsets in terms of binary labels,
        # so as to formulate it as a binary classification problem
        superset_size = events_ref_set.shape[0]
        labels = np.zeros(superset_size, dtype=np.int)
        labels[subset1] = 0
        labels[subset2] = 1

        parameters = {
            'solver': self.solver,
            'penalty': self.penalty,
            'C': self.C,
            'fit_intercept': self.fit_intercept,
            'class_weight': self.class_weight,
        }

        z = self.__infer_bits__(
            events_ref_set=events_ref_set,
            labels=labels,
            events=events,
            parameters=parameters,
        )

        return z
