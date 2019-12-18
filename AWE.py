from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

WEIGHT_CALCULATION = (
   "proportional_to_mse",
)


class AWE(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, weighting_method="proportional_to_mse"):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self._weighting_method = weighting_method
        self.ensemble_ = None
        self.weights_ = None
        self.classes_ = None
        self.X_ = None
        self.y_ = None

    def _prune(self):
        number_to_prune = len(self.ensemble_) - self.n_estimators
        for i in range(number_to_prune):
            self.ensemble_.pop(np.argmin(self.weights_))
            self.weights_ = np.delete(self.weights_, np.argmin(self.weights_))

    @staticmethod
    def _mean_squared_error(y_true, predicted_proba):
        corr_proba = predicted_proba[range(len(predicted_proba)), y_true]
        return sum((1 - corr_proba)**2) / len(y_true)

    @staticmethod
    def _random_mean_squared_error(y):
        return sum([sum(y == u) / len(y) * (1 - (sum(y == u) / len(y)))**2 for u in np.unique(y)])

    def _get_weigth_for_candidate(self, candidate_clf):
        if self._weighting_method == "proportional_to_mse":
            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5)
            mse = []
            for train_index, test_index in sss.split(self.X_, self.y_):
                candidate_clf.fit(self.X_[train_index], self.y_[train_index])
                mse.append(self._random_mean_squared_error(self.y_[test_index]) - self._mean_squared_error(self.y_[test_index], candidate_clf.predict_proba(self.X_[test_index])))
                candidate_clf.fit(self.X_[test_index], self.y_[test_index])
                mse.append(self._random_mean_squared_error(self.y_[train_index]) - self._mean_squared_error(self.y_[train_index], candidate_clf.predict_proba(self.X_[train_index])))
            return sum(mse) / len(mse)
        else:
            raise NotImplementedError

    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        candidate_clf = clone(self.base_estimator)
        candidate_clf.fit(X, y)

        self.ensemble_ = [candidate_clf]
        self.weights_ = np.array([1])
        self.classes_ = np.unique(y)

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes

            self.ensemble_ = []
            self.weights_ = np.array([])


        """Partial fitting"""


        # Preparing and training new candidate
        if classes is not None:
            self.classes_ = classes
        elif self.classes_ is None:
            raise Exception('Classes not specified')

        candidate_clf = clone(self.base_estimator)
        candidate_weight = self._get_weigth_for_candidate(candidate_clf)
        candidate_clf.fit(X, y)

        self._set_weights()

        self.ensemble_.append(candidate_clf)

        self.weights_ = np.append(self.weights_, np.array([candidate_weight]))

        # Post-pruning
        if len(self.ensemble_) > self.n_estimators:
            self._prune()

        # Weights normalization
        self.weights_ = self.weights_ / np.sum(self.weights_)

    def _set_weights(self):

        """Wang's weights"""
        if self._weighting_method == "proportional_to_mse":
            mse_rand = self._random_mean_squared_error(self.y_)
            mse_members = np.array([self._mean_squared_error(self.y_, member_clf.predict_proba(self.X_), )
                                    for member_clf in self.ensemble_])
            self.weights_ = mse_rand - mse_members

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        return sum(prediction == y) / len(y)

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Weight support before acumulation
        weighted_support = (
               self.ensemble_support_matrix(X) * self.weights_[:, np.newaxis, np.newaxis]
        )

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        return acumulated_weighted_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]

