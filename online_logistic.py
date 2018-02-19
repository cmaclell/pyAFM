from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from numbers import Number

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils import shuffle

from util import log_one_plus_exp_vect
from util import invlogit_vect


class OnlineLogistic(BaseEstimator, ClassifierMixin):

    def __init__(self, fit_intercept=True, bounds=None, l2=1.0, beta1=0.9,
                 beta2=0.999, alpha=1, eps=1e-8):
        """
        Implements Adam, but supports logistic regression, adds box
        constraints, and l2 regularization.
        """
        self.fit_intercept = fit_intercept
        self.bounds = bounds
        self.l2 = l2

        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.eps = eps

        self.w = None
        self.classes_ = set([0, 1])

    def partial_fit(self, X, y):
        """
        Train the Logistic model, X and y are numpy arrays.
        """
        X, y = check_X_y(X, y)
        # ,accept_sparse=['csr', 'csc']) # not sure how to handle sparse

        if not set(y).issubset(self.classes_):
            raise ValueError("y values must be 0 or 1.")

        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        if self.w is None:

            self.count = np.zeros(X.shape[1])
            self.w = np.zeros(X.shape[1])

            if self.bounds is None:
                self.lower_ = np.array([float("-inf") for v in self.w])
                self.upper_ = np.array([float("inf") for v in self.w])
                # self.bounds_ = [(None, None) for v in self.w]
            elif isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                self.lower_ = np.array([self.bounds[0] if self.bounds[0] is not
                                        None else float('-inf') for v in
                                        self.w])
                self.upper_ = np.array([self.bounds[1] if self.bounds[1] is not
                                        None else float('inf') for v in
                                        self.w])
                # self.bounds_ = [self.bounds for v in self.w]
            elif self.fit_intercept and len(self.bounds) == len(self.w) - 1:
                self.lower_ = np.concatenate(([float('-inf')],
                                             [b[0] if b[0] is not None else
                                                 float('-inf') for b in
                                                 self.bounds]))
                self.upper_ = np.concatenate(([float('-inf')],
                                             [b[1] if b[1] is not None else
                                                 float('inf') for b in
                                                 self.bounds]))
                # self.bounds_ = np.concatenate(([(None, None)], self.bounds))
            else:
                self.lower_ = np.array([b[0] if b[0] is not None else
                                        float('-inf') for b in self.bounds])
                self.upper_ = np.array([b[1] if b[1] is not None else
                                        float('inf') for b in self.bounds])
                # self.bounds_ = self.bounds
            if (len(self.upper_) != len(self.w) or
                    len(self.lower_) != len(self.w)):
                raise ValueError("Bounds must be the same length as the w")

            if isinstance(self.l2, Number):
                self.l2_ = [self.l2 for v in self.w]
            elif self.fit_intercept and len(self.l2) == len(self.w) - 1:
                self.l2_ = np.insert(self.l2, 0, 0)
            else:
                self.l2_ = self.l2

            if len(self.l2_) != len(self.w):
                raise ValueError("L2 penalty must be the same length as the"
                                 "coef, be sure the intercept is accounted"
                                 "for.")

            # the intercept should never be regularized.
            if self.fit_intercept:
                self.lower_[0] = float('-inf')
                self.upper_[0] = float('inf')
                self.l2_[0] = 0.0

            # Adam parameters
            self.momentum = np.zeros(self.w.shape)
            self.learning_rate = np.zeros(self.w.shape)
            # self.G = np.zeros((self.w.shape[0], self.w.shape[0]))

        # Do checks on inputs and weights.
        if len(X[0]) != len(self.w):
            raise ValueError("X does not have the same dimensions as w.")

        # Get mask for only updating terms that are present
        mask = np.any(X, axis=0)
        #mask = np.array([True] * X.shape[1])

        # Do the RMS prop stuff to update w.
        grad = _ll_grad(self.w, X, y, self.l2_)

        self.method = 'Adam'

        if self.method == 'SGD':
            self.w[mask] -= self.alpha * grad[mask]

        elif self.method == 'SGD+Momentum':
            self.w[mask] -= (self.beta1 * self.momentum + self.alpha *
                             grad[mask])
            self.momentum[mask] = grad[mask]

        elif self.method == 'NAG':
            self.momentum[mask] = (self.beta1 * self.momentum[mask] +
                                   self.alpha *
                                   _ll_grad(self.w - self.beta1 *
                                            self.momentum[mask],
                                            X, y, self.l2_))
            self.w[mask] -= self.momentum[mask]

        elif self.method == 'AdaGrad':
            # print(np.dot(self.alpha / np.sqrt(self.G + self.eps), grad))
            self.w[mask] -= (self.alpha / np.sqrt(self.learning_rate +
                             self.eps) * grad)
            # self.w[mask] -= self.alpha / np.sqrt(self.G + self.eps) * grad
            self.learning_rate += np.square(grad)

        elif self.method == "AdaDelta":
            self.learning_rate[mask] = (self.beta2 * self.learning_rate[mask] +
                                        (1 - self.beta2) *
                                        np.square(grad[mask]))
            delta_w = (np.sqrt(self.momentum[mask] + self.eps) /
                       np.sqrt(self.learning_rate[mask] + self.eps) *
                       grad[mask])
            self.w[mask] -= delta_w

            self.momentum[mask] = (self.beta1 * self.momentum[mask] +
                                   (1 - self.beta1) * np.square(delta_w))

        elif self.method == "RMSProp":
            self.learning_rate[mask] = (self.beta2 * self.learning_rate[mask] +
                                        (1 - self.beta2) *
                                        np.square(grad[mask]))
            self.w[mask] -= (self.alpha / np.sqrt(self.learning_rate[mask] +
                             self.eps) * grad[mask])

        elif self.method == "Adam":
            self.momentum[mask] = (self.beta1 * self.momentum[mask] +
                                   (1 - self.beta1) * grad[mask])
            self.learning_rate[mask] = (self.beta2 * self.learning_rate[mask] +
                                        (1 - self.beta2) * np.square(grad[mask]))

            self.count[mask] += 1

            momentum_hat = (self.momentum[mask] /
                            (1 - np.power(self.beta1, self.count[mask])))
            learning_rate_hat = (self.learning_rate[mask] / (1 -
                                 np.power(self.beta2, self.count[mask])))

            self.w[mask] -= (self.alpha / np.sqrt(self.count[mask]) *
                             momentum_hat / (np.sqrt(learning_rate_hat) +
                             self.eps))

        # Clamp weights by bounds.
        self.w[mask] = np.maximum(self.w[mask], self.lower_[mask])
        self.w[mask] = np.minimum(self.w[mask], self.upper_[mask])

        if self.fit_intercept:
            self.coef_ = self.w[1:]
            self.intercept = self.w[0]
        else:
            self.coef_ = self.w[0:]

    def fit(self, X, y, minibatch_size=1):
        self.w = None

        for _ in range(len(y)):
            for i in range(0, X.shape[0], minibatch_size):
                X_mini = X[i:i + minibatch_size]
                y_mini = y[i:i + minibatch_size]
                self.partial_fit(X_mini, y_mini)

    def predict(self, X):
        """
        Returns the predicted class for each x in X, predicts 1 if probability
        is greater than or equal to 1.
        """
        y = np.array(self.predict_proba(X))
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        return y

    def predict_proba(self, X):
        """
        Returns the probability of class 1 for each x in X.
        """
        if self.w is None:
            return np.array([0] * len(X))
            # raise RuntimeError("You must train classifer before predicting"
            #                    "data!")

        X = check_array(X)
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        return invlogit_vect(np.dot(self.w, np.transpose(X)))

    def mean_squared_error(self, X, y):
        pred = self.predict_proba(X)
        sq_err = [(v-pred[i]) * (v-pred[i]) for i, v in enumerate(y)]
        return np.average(sq_err)


def _ll(w, X, y, l2):
    """
    Logistic Regression loglikelihood given, the weights w, the data X, and the
    labels y.
    """
    z = np.dot(w, np.transpose(X))
    ll = sum(np.subtract(log_one_plus_exp_vect(z), np.multiply(y, z)))
    ll += np.dot(np.divide(l2, 2), np.multiply(w, w))
    return ll


def _ll_grad(w, X, y, l2):
    """
    Logistic Regression loglikelihood gradient given, the weights w, the data
    X, and the labels y.
    """
    p = invlogit_vect(np.dot(w, np.transpose(X)))
    g = np.dot(np.transpose(X), np.subtract(y, p))
    g -= np.multiply(l2, w)
    return -1 * g
